"""
gpr_predict_grid.py
===================
Predict wind speed over a spatial grid for one city and one day.

Usage
-----
  Edit CITY and PRED_DATE at the top, then run.

Outputs
-------
  output/<city>/gpr_<CITY>_<DATE>.nc   — NetCDF4 with mean / std / ci95
  output/<city>/gpr_<CITY>_<DATE>_<HH>UTC.png  — pcolormesh plot per hour
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from linear_operator.operators import DiagLinearOperator
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.warp import transform as raster_transform
from rasterio.windows import from_bounds
import xarray as xr
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cmcrameri import cm as cmc

# =============================================================================
# ← EDIT THESE TWO LINES
# =============================================================================

CITY      = 'zurich'          # 'zurich' | 'milan' | 'rome'
PRED_DATE = '2018-06-15'      # any date in the city's ERA5 period

# =============================================================================
# PER-CITY CONFIG
# =============================================================================

CITY_CONFIGS = {
    'zurich': {
        'model_path':      'obs/models/gpr_zurich.pt',
        'lambda_p_tif':    '../tiffs/z0/zurich_lambda_p_30.tif',
        'mean_height_tif': '../tiffs/z0/zurich_mean_height_30.tif',
        'z0_tif':          '../tiffs/z0/zurich_30.tif',
        'zd_tif':          '../tiffs/zd/zurich_30.tif',
        'dtm_tif':         '../tiffs/dtm/zurich_30.tif',
        'era5_glob':       '../raw_data/era5/*2018*.nc',
        'era5_static_glob':'../raw_data/era5/*2018*.nc',
        'height_ag':       10.0,
        'output_dir':      'output/zurich',
    },
    'milan': {
        'model_path':      'obs/models/gpr_milan.pt',
        'lambda_p_tif':    '../tiffs/z0/milan_lambda_p_30.tif',
        'mean_height_tif': '../tiffs/z0/milan_mean_height_30.tif',
        'z0_tif':          '../tiffs/z0/milan_30.tif',
        'zd_tif':          '../tiffs/zd/milan_30.tif',
        'dtm_tif':         '../tiffs/dtm/milan_30.tif',
        'era5_glob':       '../raw_data/era5/*2018*.nc',
        'era5_static_glob':'../raw_data/era5/*2018*.nc',
        'height_ag':       10.0,
        'output_dir':      'output/milan',
    },
    'rome': {
        'model_path':      'obs/models/gpr_rome.pt',
        'lambda_p_tif':    '../tiffs/z0/rome_lambda_p_30.tif',
        'mean_height_tif': '../tiffs/z0/rome_mean_height_30.tif',
        'z0_tif':          '../tiffs/z0/rome_30.tif',
        'zd_tif':          '../tiffs/zd/rome_30.tif',
        'dtm_tif':         '../tiffs/dtm/rome_30.tif',
        'era5_glob':       '../raw_data/era5/*2022*.nc',
        'era5_static_glob':'../raw_data/era5/*2018*.nc',
        'height_ag':       10.0,
        'output_dir':      'output/rome',
    },
}

BATCH_SIZE = 8000   # reduce if you hit memory limits

# =============================================================================
# MODEL CLASSES — must be identical to gpr_wind.py
# =============================================================================

SOURCE_MAP      = {'station': 0, 'urban_tales': 1, 'street_network': 2}
N_SOURCES       = len(SOURCE_MAP)
MATERN_NU       = 2.5
_INIT_RAW_FLOOR = -3.7

KERNEL_COLS   = ['lambda_p', 'mean_height', 'elev_diff', 'height_ag',
                 'zust', 'era5_sinWD', 'era5_cosWD']
MEAN_COLS     = ['zust', 'height_ag', 'z0', 'zd', 'mean_height']
MORPH_FEATS   = ['lambda_p', 'mean_height', 'height_ag']
FORCING_FEATS = ['zust', 'era5_sinWD', 'era5_cosWD', 'elev_diff']

n_kernel     = len(KERNEL_COLS)
morph_dims   = [KERNEL_COLS.index(c) for c in MORPH_FEATS   if c in KERNEL_COLS]
forcing_dims = [KERNEL_COLS.index(c) for c in FORCING_FEATS if c in KERNEL_COLS]
zust_idx     = n_kernel + MEAN_COLS.index('zust')
z_idx        = n_kernel + MEAN_COLS.index('height_ag')
z0_idx       = n_kernel + MEAN_COLS.index('z0')
zd_idx       = n_kernel + MEAN_COLS.index('zd')
H_idx        = n_kernel + MEAN_COLS.index('mean_height')


class PerSourceNoiseLikelihood(FixedNoiseGaussianLikelihood):
    def __init__(self, noise, source_idx, n_sources, **kwargs):
        super().__init__(noise=noise, learn_additional_noise=False, **kwargs)
        self.register_buffer('source_idx', source_idx)
        self.register_parameter(
            'raw_noise_floors',
            torch.nn.Parameter(torch.full((n_sources,), _INIT_RAW_FLOOR))
        )

    @property
    def noise_floors(self):
        return F.softplus(self.raw_noise_floors)

    def marginal(self, function_dist, *args, **kwargs):
        per_point_floor = self.noise_floors[self.source_idx]
        total_noise     = self.noise + per_point_floor
        covar = (function_dist.lazy_covariance_matrix
                 + DiagLinearOperator(total_noise))
        return function_dist.__class__(function_dist.mean, covar)


class ZustMean(gpytorch.means.Mean):
    def __init__(self, zust_idx, z_idx, z0_idx, zd_idx, H_idx):
        super().__init__()
        self.zust_idx = zust_idx
        self.z_idx    = z_idx
        self.z0_idx   = z0_idx
        self.zd_idx   = zd_idx
        self.H_idx    = H_idx
        for name, val in [('kap_inv', 2.44), ('b', 0.), ('c', 0.), ('d', 1.),
                           ('e', 0.), ('f', 0.), ('alpha', 0.),
                           ('beta', 0.), ('gamma', 0.)]:
            self.register_parameter(
                name, torch.nn.Parameter(torch.tensor(val))
            )

    def forward(self, x):
        zust    = x[:, self.zust_idx]
        z       = x[:, self.z_idx]
        z0      = x[:, self.z0_idx].clamp(min=1e-4)
        zd      = x[:, self.zd_idx].clamp(min=1e-2)
        H       = x[:, self.H_idx].clamp(min=1e-2)
        log_arg = ((z - (zd + self.c)) / (z0 + self.d).clamp(min=1e-4)).clamp(min=1e-3)
        mean_above = self.kap_inv * (zust + self.b) * (torch.log(log_arg) + self.e) + self.f
        exp_arg    = (self.beta * (z - H) / H).clamp(-20., 20.)
        mean_below = self.alpha * torch.exp(exp_arg) + self.gamma
        return torch.where(z > H, mean_above, mean_below)


class WindGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 morph_dims, forcing_dims, zust_idx, z_idx,
                 z0_idx, zd_idx, H_idx):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZustMean(zust_idx, z_idx, z0_idx, zd_idx, H_idx)
        lsc = gpytorch.constraints.Interval(0.05, 3.0)
        self.k_morph = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=MATERN_NU, ard_num_dims=len(morph_dims),
                active_dims=torch.tensor(morph_dims), lengthscale_constraint=lsc,
            )
        )
        self.k_forcing = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=MATERN_NU, ard_num_dims=len(forcing_dims),
                active_dims=torch.tensor(forcing_dims), lengthscale_constraint=lsc,
            )
        )
        self.covar_module = self.k_morph + self.k_forcing

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# =============================================================================
# STEP 1 — Load static rasters (once)
# =============================================================================

cfg = CITY_CONFIGS[CITY]
os.makedirs(cfg['output_dir'], exist_ok=True)
date_str = pd.Timestamp(PRED_DATE).strftime('%Y-%m-%d')
print(f"\n{'='*60}")
print(f"City: {CITY.upper()}   Day: {date_str}")
print(f"{'='*60}")

print("\n[1] Loading static rasters …")

def read_raster(path):
    with rasterio.open(path) as src:
        data  = src.read(1).astype(np.float32)
        meta  = {'transform': src.transform, 'crs': src.crs,
                 'height': src.height, 'width': src.width, 'nodata': src.nodata}
    return data, meta

lp_2d,  meta = read_raster(cfg['lambda_p_tif'])
mh_2d,  _    = read_raster(cfg['mean_height_tif'])
z0_2d,  _    = read_raster(cfg['z0_tif'])
zd_2d,  _    = read_raster(cfg['zd_tif'])
dtm_2d, _    = read_raster(cfg['dtm_tif'])



def read_raster_intersected(paths: list[str]):
    """Read multiple rasters clipped to their geographic intersection."""

    # --- 1. collect bounds from all rasters ---
    srcs = [rasterio.open(p) for p in paths]
    # geographic intersection of bounding boxes
    left   = max(s.bounds.left   for s in srcs)
    bottom = max(s.bounds.bottom for s in srcs)
    right  = min(s.bounds.right  for s in srcs)
    top    = min(s.bounds.top    for s in srcs)


    # --- 2. read each raster through its own window ---
    arrays, metas = [], []
    for src in srcs:
        win  = from_bounds(left, bottom, right, top, src.transform)
        data = src.read(1, window=win).astype(np.float32)
        meta = {
            'transform': src.window_transform(win),
            'crs':       src.crs,
            'height':    data.shape[0],
            'width':     data.shape[1],
            'nodata':    src.nodata,
        }
        arrays.append(data)
        metas.append(meta)

        # for s in srcs: s.close()

    return arrays, metas


paths = [cfg['lambda_p_tif'], cfg['mean_height_tif'],
         cfg['z0_tif'],       cfg['zd_tif'],          cfg['dtm_tif']]

(lp_2d, mh_2d, z0_2d, zd_2d, dtm_2d), (meta, *_) = read_raster_intersected(paths)

nrows, ncols = meta['height'], meta['width']
valid_mask   = (np.isfinite(lp_2d) & np.isfinite(mh_2d) &
                np.isfinite(z0_2d) & np.isfinite(zd_2d) & np.isfinite(dtm_2d))
n_valid      = valid_mask.sum()
print(f"  Grid: {nrows}×{ncols}  valid cells: {n_valid}")

# Flat static arrays — extracted once
lp_flat      = lp_2d[valid_mask]
mh_flat      = mh_2d[valid_mask]
z0_flat      = z0_2d[valid_mask]
zd_flat      = zd_2d[valid_mask]
dtm_flat     = dtm_2d[valid_mask]
h_ag_flat    = np.full(n_valid, cfg['height_ag'], dtype=np.float32)

# Grid coordinates in raster CRS for NetCDF
col_coords = (np.arange(ncols) * meta['transform'].a
              + meta['transform'].c + meta['transform'].a / 2)
row_coords = (np.arange(nrows) * meta['transform'].e
              + meta['transform'].f + meta['transform'].e / 2)

# =============================================================================
# STEP 2 — Elevation correction (static scalar for city-centre ERA5 cell)
# =============================================================================

print("\n[2] Computing elevation correction …")

from pyproj import Transformer
tr_to_wgs = Transformer.from_crs(str(meta['crs']), 'EPSG:4326', always_xy=True)
rows_idx, cols_idx = np.where(valid_mask)
xs, ys = rasterio.transform.xy(meta['transform'], rows_idx, cols_idx)
lons_all, lats_all = tr_to_wgs.transform(np.array(xs), np.array(ys))
lat_c, lon_c = float(lats_all.mean()), float(lons_all.mean())

static_file = sorted(glob.glob(cfg['era5_static_glob']))[0]
with xr.open_dataset(static_file) as ds_s:
    era5_elev = float(
        ds_s['z'].sel(latitude=lat_c, longitude=lon_c, method='nearest')
        .isel(valid_time=0).values.item()
    ) / 9.80665

elev_diff_flat = (dtm_flat - era5_elev).astype(np.float32)
print(f"  ERA5 reference elevation: {era5_elev:.1f} m")
print(f"  elev_diff range: [{elev_diff_flat.min():.1f}, {elev_diff_flat.max():.1f}] m")

# Static kernel block — columns that don't change with time
# Order matches KERNEL_COLS: lambda_p, mean_height, elev_diff, height_ag, [zust, sinWD, cosWD added per step]
static_Xk_part = np.column_stack([
    lp_flat, mh_flat, elev_diff_flat, h_ag_flat
]).astype(np.float32)   # shape (n_valid, 4)

# Static mean block — columns that don't change with time
# Order matches MEAN_COLS: [zust, height_ag, z0, zd, mean_height]
static_Xm_part = np.column_stack([
    h_ag_flat, z0_flat, zd_flat, mh_flat
]).astype(np.float32)   # shape (n_valid, 4)   (zust added per step)

# =============================================================================
# STEP 3 — Load model
# =============================================================================

print("\n[3] Loading model …")

ckpt   = torch.load(cfg['model_path'], map_location='cpu', weights_only=False)
scaler = StandardScaler()
scaler.mean_  = ckpt['scaler_mean']
scaler.scale_ = ckpt['scaler_scale']

noise_tr   = ckpt['noise_tr']
source_idx = ckpt['source_idx']
n_cols     = len(KERNEL_COLS) + len(MEAN_COLS)
dummy_x    = torch.zeros(len(noise_tr), n_cols)
dummy_y    = torch.zeros(len(noise_tr))

likelihood = PerSourceNoiseLikelihood(
    noise=noise_tr, source_idx=source_idx, n_sources=N_SOURCES
)
model = WindGP(dummy_x, dummy_y, likelihood,
               morph_dims, forcing_dims,
               zust_idx, z_idx, z0_idx, zd_idx, H_idx)
model.load_state_dict(ckpt['model_state'])
likelihood.load_state_dict(ckpt['likelihood_state'])
model.eval(); likelihood.eval()
print(f"  Model loaded from {cfg['model_path']}")

# ── STEP 4 (revised) — load ERA5, slice to city bbox, load into RAM ──────────
print("\n[4] Loading ERA5 forcing …")

day_start = pd.Timestamp(PRED_DATE)#, tz='UTC')
day_end   = day_start + pd.Timedelta(hours=23)

pred_ts   = pd.Timestamp(PRED_DATE)
year      = pred_ts.year

# Find the single file for this year
era5_files = sorted(glob.glob(cfg['era5_glob']))
era5_file  = next(f for f in era5_files if str(year) in os.path.basename(f))

# Bounding box of the city grid (+ 1 ERA5 cell buffer so nearest never clips)
lon_min, lon_max = float(lons_all.min()), float(lons_all.max())
lat_min, lat_max = float(lats_all.min()), float(lats_all.max())

with xr.open_dataset(f"../raw_data/era5/ERA5_{PRED_DATE}.nc") as ds_era5:
    ds_day = ds_era5.sel(
        latitude  = slice(lat_max + 0.5, lat_min - 0.5),
        longitude = slice(lon_min - 0.5, lon_max + 0.5),
    ).load()

time_dim  = 'valid_time' 
times_day = pd.to_datetime(ds_day[time_dim].values)
n_times   = len(times_day)
print(f"  {n_times} hours: {times_day[0]} → {times_day[-1]}")
print(f"  ERA5 spatial subset: "
      f"{ds_day.latitude.size} lat × {ds_day.longitude.size} lon cells")

# ── Nearest-neighbour lookup: one index per valid grid cell ───────────────────
era5_lats = ds_day.latitude.values   # shape (nlat,)   N→S
era5_lons = ds_day.longitude.values  # shape (nlon,)

# argmin over ERA5 axis for each of the n_valid cells → shape (n_valid,)
lat_idx = np.argmin(np.abs(era5_lats[:, None] - lats_all[None, :]), axis=0)
lon_idx = np.argmin(np.abs(era5_lons[:, None] - lons_all[None, :]), axis=0)

# Extract all variables at once: shape (n_times, n_valid)
zust_grid  = ds_day['zust'].values[:, lat_idx, lon_idx].astype(np.float32)
u10_grid   = ds_day['u10'].values[:, lat_idx, lon_idx].astype(np.float32)
v10_grid   = ds_day['v10'].values[:, lat_idx, lon_idx].astype(np.float32)

era5_wd    = (180 + np.degrees(np.arctan2(u10_grid, v10_grid))) % 360
sinWD_grid = np.sin(np.deg2rad(era5_wd)).astype(np.float32)  # (n_times, n_valid)
cosWD_grid = np.cos(np.deg2rad(era5_wd)).astype(np.float32)

# Per-cell elevation correction (replaces the single city-centre scalar)
era5_z_grid = ds_day['z'].isel({time_dim: 0}).values[lat_idx, lon_idx] / 9.80665
elev_diff_flat = (dtm_flat - era5_z_grid.astype(np.float32))

print(f"  elev_diff range: [{elev_diff_flat.min():.1f}, {elev_diff_flat.max():.1f}] m")

# Rebuild static_Xk_part now that elev_diff is per-cell
static_Xk_part = np.column_stack(
    [lp_flat, mh_flat, elev_diff_flat, h_ag_flat]
).astype(np.float32)   # (n_valid, 4)

# ── STEP 5 (revised) — time loop uses pre-loaded per-cell arrays ─────────────
print(f"\n[5] Predicting ({n_valid} cells × {n_times} hours) …")

out_mean = np.full((n_times, nrows, ncols), np.nan, dtype=np.float32)
out_std  = np.full((n_times, nrows, ncols), np.nan, dtype=np.float32)

for t_idx in range(n_times):
    # All three are already (n_valid,) — no broadcasting needed
    zust_col  = zust_grid[t_idx]
    sinWD_col = sinWD_grid[t_idx]
    cosWD_col = cosWD_grid[t_idx]

    Xk = np.concatenate([
        static_Xk_part,
        zust_col [:, None],
        sinWD_col[:, None],
        cosWD_col[:, None],
    ], axis=1)                     # (n_valid, 7)

    Xm = np.concatenate([
        zust_col[:, None],
        static_Xm_part,
    ], axis=1)                     # (n_valid, 5)

    Xk_s = scaler.transform(Xk)
    X    = np.concatenate([Xk_s, Xm], axis=1).astype(np.float32)

    mean_flat = np.empty(n_valid, dtype=np.float32)
    std_flat  = np.empty(n_valid, dtype=np.float32)

    for start in range(0, n_valid, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_valid)
        x_t = torch.tensor(X[start:end])
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f = model(x_t)
        mean_flat[start:end] = np.maximum(f.mean.numpy(), 0.0)
        std_flat[start:end]  = f.stddev.numpy()

    out_mean[t_idx][valid_mask] = mean_flat
    out_std[t_idx][valid_mask]  = std_flat

    print(f"  {t_idx+1:2d}/{n_times}  {times_day[t_idx].strftime('%H:%M')} UTC  "
          f"mean={mean_flat.mean():.3f}  max={mean_flat.max():.3f} m/s")

# =============================================================================
# STEP 6 — Write NetCDF4
# =============================================================================

print("\n[6] Writing NetCDF4 …")

nc_name = f"gpr_{CITY}_{date_str.replace('-', '')}.nc"
nc_path = os.path.join(cfg['output_dir'], nc_name)

ds_out = xr.Dataset(
    {
        'wind_mean': xr.DataArray(
            out_mean, dims=['time', 'y', 'x'],
            attrs={'long_name': 'GPR mean wind speed', 'units': 'm/s'}
        ),
        'wind_std': xr.DataArray(
            out_std, dims=['time', 'y', 'x'],
            attrs={'long_name': 'GPR wind speed std-dev', 'units': 'm/s'}
        ),
        'wind_ci95': xr.DataArray(
            out_std * 1.96, dims=['time', 'y', 'x'],
            attrs={'long_name': 'GPR 95% CI half-width', 'units': 'm/s'}
        ),
    },
    coords={
        'time': times_day,
        'y':    ('y', row_coords, {'long_name': f'y coordinate ({meta["crs"]})', 'units': 'm'}),
        'x':    ('x', col_coords, {'long_name': f'x coordinate ({meta["crs"]})', 'units': 'm'}),
    },
    attrs={
        'city':       CITY,
        'date':       date_str,
        'height_ag':  cfg['height_ag'],
        'crs':        str(meta['crs']),
        'Conventions': 'CF-1.8',
    }
)
ds_out.to_netcdf(nc_path, format='NETCDF4')
print(f"  Saved → {nc_path}")

# =============================================================================
# STEP 7 — Quick pcolormesh plots for each hour
# =============================================================================

print("\n[7] Plotting …")

plot_dir = os.path.join(cfg['output_dir'], f"plots_{date_str.replace('-', '')}")
os.makedirs(plot_dir, exist_ok=True)

# Shared colour scale across all hours
vmin = np.nanpercentile(out_mean, 2)
vmax = np.nanpercentile(out_mean, 98)

X2d, Y2d = np.meshgrid(col_coords, row_coords)

for t_idx, ts in enumerate(times_day):
    fig, ax = plt.subplots(figsize=(7, 6))
    pcm = ax.pcolormesh(
        X2d, Y2d, out_mean[t_idx],
        cmap=cmc.batlow,
        vmin=vmin, vmax=vmax,
        shading='auto',
    )
    cbar = fig.colorbar(pcm, ax=ax, label='Wind speed (m/s)', shrink=0.8)
    ax.set_aspect('equal')
    ax.set_xlabel(f'x [{meta["crs"]}] (m)')
    ax.set_ylabel(f'y [{meta["crs"]}] (m)')
    ax.set_title(
        f"{CITY.capitalize()} — GPR wind speed\n"
        f"{ts.strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"h={cfg['height_ag']} m a.g.",
        fontsize=10,
    )
    plt.tight_layout()
    fname = os.path.join(plot_dir, f"ws_{ts.strftime('%H')}UTC.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()

print(f"  {n_times} plots saved to {plot_dir}/")

# =============================================================================
# HOW TO RE-PLOT FROM NetCDF (example snippet)
# =============================================================================

print(f"""
To re-plot any timestamp from the NetCDF:

  import xarray as xr
  import matplotlib.pyplot as plt
  from cmcrameri import cm as cmc
  import numpy as np

  ds  = xr.open_dataset('{nc_path}')
  ts  = ds.time[6]                         # pick any hour index
  arr = ds['wind_mean'].sel(time=ts)

  X, Y = np.meshgrid(ds.x.values, ds.y.values)
  fig, ax = plt.subplots()
  pcm = ax.pcolormesh(X, Y, arr.values, cmap=cmc.batlow, shading='auto')
  fig.colorbar(pcm, ax=ax, label='m/s')
  ax.set_title(str(ts.values))
  plt.show()
""")

print("Done.")