"""
gpr_wind_full.py
================
GPR for urban wind speed downscaling — full training on ALL data (no splits).
Reads obs/gpr_obs.pkl and obs/gpr_train_gpr.pkl produced by gpr_0_prep.py.

Trains on all stations + MNW + UrbanTales sims simultaneously.
Produces velocity profile plots showing how the learned mean function
varies wind speed with height for each station's morphological context.

Outputs
-------
  plots/gpr_wind/velocity_profile.png   — wind-speed profiles varying height_ag
  plots/gpr_wind/ts_<station>.png       — optional time series per station
  Hyperparameter table printed to console
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from linear_operator.operators import DiagLinearOperator
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from cmcrameri import cm as cmc

# =============================================================================
# CONFIG
# =============================================================================

OBS_PATH   = 'obs/gpr_obs.pkl'
TRAIN_PATH = 'obs/gpr_train_gpr.pkl'
PLOT_DIR   = 'plots/gpr_wind'
os.makedirs(PLOT_DIR, exist_ok=True)

N_ITER      = 2500
LR          = 0.1
PRINT_EVERY = 100
PATIENCE    = 200
MIN_DELTA   = 1e-4

MATERN_NU       = 2.5
_INIT_RAW_FLOOR = -3.7

SOURCE_MAP = {
    'station':        0,
    'urban_tales':    1,
    'street_network': 2,
}
N_SOURCES = len(SOURCE_MAP)

# Velocity profile config
PROFILE_HEIGHT_MIN  = 0.5    # m
PROFILE_HEIGHT_MAX  = 25.0   # m
PROFILE_HEIGHT_STEP = 0.25   # m
PROFILE_HOUR        = 15     # UTC hour used as reference for morphology snapshot

# =============================================================================
# PART 1 — Load
# =============================================================================

obs_obj   = pd.read_pickle(OBS_PATH)
train_obj = pd.read_pickle(TRAIN_PATH)

df_obs   = obs_obj['df'].copy()
df_train = train_obj['df'].copy()
meta     = train_obj['meta']

FEAT_COLS = meta['feat_cols']
TARGET    = meta['target_col']

MORPH_FEATS   = ['lambda_p', 'mean_height', 'height_ag']
FORCING_FEATS = ['zust', 'era5_sinWD', 'era5_cosWD', 'elev_diff']

# Kernel columns = all FEAT_COLS (standardised, seen by kernels only)
# Mean columns   = physical quantities passed raw to the mean function
KERNEL_COLS = FEAT_COLS
MEAN_COLS   = ['zust', 'height_ag', 'z0', 'zd', 'mean_height']

n_kernel = len(KERNEL_COLS)

# Indices into the concatenated [kernel | mean] input matrix
zust_idx = n_kernel + MEAN_COLS.index('zust')
z_idx    = n_kernel + MEAN_COLS.index('height_ag')
z0_idx   = n_kernel + MEAN_COLS.index('z0')
zd_idx   = n_kernel + MEAN_COLS.index('zd')
H_idx    = n_kernel + MEAN_COLS.index('mean_height')

# Kernel active_dims index into the full [kernel|mean] matrix
morph_dims   = [KERNEL_COLS.index(c) for c in MORPH_FEATS   if c in KERNEL_COLS]
forcing_dims = [KERNEL_COLS.index(c) for c in FORCING_FEATS if c in KERNEL_COLS]

print(f"k_morph   dims : {[KERNEL_COLS[i] for i in morph_dims]}")
print(f"k_forcing dims : {[KERNEL_COLS[i] for i in forcing_dims]}")

df_obs['Datum']   = pd.to_datetime(df_obs['Datum'])
df_train['Datum'] = pd.to_datetime(df_train['Datum'])

# All real station IDs (not sims or street network)
all_stations = sorted(
    df_train[df_train['source'] == 'station']['Standort']
    .dropna().unique().tolist()
)

print(f"Stations       : {all_stations}")
print(f"Features       : {FEAT_COLS}")
print(f"df_obs         : {len(df_obs)} rows  (all hours)")
print(f"df_train       : {len(df_train)} rows  "
      f"({(df_train['source']=='station').sum()} obs  "
      f"+ {(df_train['source']=='urban_tales').sum()} sim  "
      f"+ {(df_train['source']=='street_network').sum()} mnw)")

# =============================================================================
# PART 2 — Likelihood  (identical to gpr_wind.py)
# =============================================================================

class PerSourceNoiseLikelihood(FixedNoiseGaussianLikelihood):
    """
    Learnable noise floor per observation source type.
    σ²_total(i) = noise_var(i) [fixed] + noise_floor[source(i)] [learned]
    """
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


# =============================================================================
# PART 3 — GP model  (identical to gpr_wind.py)
# =============================================================================

class ZustMean(gpytorch.means.Mean):
    """
    Conditional log-law prior mean.

    Above canopy (z > H):
        m(x) = kap_inv * (zust + b) * (log((z - (zd+c)) / (z0+d)) + e) + f

    Within canopy (z <= H):
        m(x) = alpha * exp(beta * (z - H) / H) + gamma

    z0, zd, H read directly from raw (unstandardised) MEAN_COLS columns.
    All parameters learned via MLL.
    """
    def __init__(self, zust_idx, z_idx, z0_idx, zd_idx, H_idx):
        super().__init__()
        self.zust_idx = zust_idx
        self.z_idx    = z_idx
        self.z0_idx   = z0_idx
        self.zd_idx   = zd_idx
        self.H_idx    = H_idx

        self.register_parameter('kap_inv', torch.nn.Parameter(torch.tensor(2.44)))
        self.register_parameter('b',       torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter('c',       torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter('d',       torch.nn.Parameter(torch.ones(1)))
        self.register_parameter('e',       torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter('f',       torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter('alpha',   torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter('beta',    torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter('gamma',   torch.nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        zust = x[:, self.zust_idx]
        z    = x[:, self.z_idx]
        z0   = x[:, self.z0_idx].clamp(min=1e-4)
        zd   = x[:, self.zd_idx].clamp(min=1e-2)
        H    = x[:, self.H_idx].clamp(min=1e-2)

        log_arg    = ((z - (zd + self.c)) / (z0 + self.d).clamp(min=1e-4)).clamp(min=1e-3)
        mean_above = self.kap_inv * (zust + self.b) * (torch.log(log_arg) + self.e) + self.f

        exp_arg    = (self.beta * (z - H) / H).clamp(min=-20.0, max=20.0)
        mean_below = self.alpha * torch.exp(exp_arg) + self.gamma

        return torch.where(z > H, mean_above, mean_below)


class WindGP(gpytorch.models.ExactGP):
    """
    Additive kernel: k_total = k_morph + k_forcing
    Kernels use active_dims into the standardised KERNEL_COLS block only.
    Raw MEAN_COLS appended after KERNEL_COLS are invisible to both kernels.
    """
    def __init__(self, train_x, train_y, likelihood,
                 morph_dims, forcing_dims,
                 zust_idx, z_idx, z0_idx, zd_idx, H_idx):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZustMean(zust_idx, z_idx, z0_idx, zd_idx, H_idx)
        lsc = gpytorch.constraints.Interval(0.05, 3.0)
        self.k_morph = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=MATERN_NU, ard_num_dims=len(morph_dims),
                active_dims=torch.tensor(morph_dims),
                lengthscale_constraint=lsc,
            )
        )
        self.k_forcing = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=MATERN_NU, ard_num_dims=len(forcing_dims),
                active_dims=torch.tensor(forcing_dims),
                lengthscale_constraint=lsc,
            )
        )
        self.covar_module = self.k_morph + self.k_forcing

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# =============================================================================
# PART 4 — Fit on ALL data
# =============================================================================

def fit_model(df_tr):
    """Train on df_tr. Returns model, likelihood, scaler."""
    required_cols = KERNEL_COLS + MEAN_COLS
    df_tr = df_tr.dropna(subset=required_cols).copy()

    y_tr       = df_tr[TARGET].values.astype(np.float32)
    Xk_tr      = df_tr[KERNEL_COLS].values.astype(np.float32)
    Xm_tr      = df_tr[MEAN_COLS].values.astype(np.float32)
    scaler      = StandardScaler()
    Xk_tr_s    = scaler.fit_transform(Xk_tr)
    X_tr        = np.concatenate([Xk_tr_s, Xm_tr], axis=1)

    train_x    = torch.tensor(X_tr, dtype=torch.float32)
    train_y    = torch.tensor(y_tr, dtype=torch.float32)
    noise_tr   = torch.clamp(
        torch.tensor(df_tr['noise_var'].values, dtype=torch.float32), min=0.09
    )
    source_idx = torch.tensor(
        df_tr['source'].map(SOURCE_MAP).fillna(0).values.astype(np.int64),
        dtype=torch.long,
    )

    likelihood = PerSourceNoiseLikelihood(
        noise=noise_tr, source_idx=source_idx, n_sources=N_SOURCES
    )
    model = WindGP(
        train_x, train_y, likelihood,
        morph_dims, forcing_dims,
        zust_idx, z_idx, z0_idx, zd_idx, H_idx,
    )

    model.train(); likelihood.train()
    optimizer = Adam(model.parameters(), lr=LR)
    mll       = ExactMarginalLogLikelihood(likelihood, model)

    print(f"\nTraining on {len(df_tr)} points  "
          f"({(df_tr['source']=='station').sum()} obs  "
          f"+ {(df_tr['source']=='urban_tales').sum()} sim  "
          f"+ {(df_tr['source']=='street_network').sum()} mnw) …")

    best_loss  = float('inf')
    no_improve = 0

    for i in range(1, N_ITER + 1):
        optimizer.zero_grad()
        with gpytorch.settings.cholesky_jitter(1e-2):
            loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if best_loss - loss_val > MIN_DELTA:
            best_loss  = loss_val
            no_improve = 0
        else:
            no_improve += 1

        if i % PRINT_EVERY == 0:
            floors_str = '  '.join(
                f"{src}={likelihood.noise_floors[idx].detach().sqrt().item():.4f}"
                for src, idx in SOURCE_MAP.items()
            )
            print(f"  iter {i:4d}  MLL={-loss_val:8.4f}  "
                  f"noise_floors(m/s): {floors_str}  no_improve={no_improve}")

        if no_improve >= PATIENCE:
            print(f"  Early stop at iter {i}")
            break

    model.eval(); likelihood.eval()
    return model, likelihood, scaler


# =============================================================================
# PART 5 — Hyperparameters
# =============================================================================

def print_hyperparameters(model, likelihood):
    morph_ls      = model.k_morph.base_kernel.lengthscale.detach().squeeze().numpy()
    forcing_ls    = model.k_forcing.base_kernel.lengthscale.detach().squeeze().numpy()
    morph_names   = [KERNEL_COLS[i] for i in morph_dims]
    forcing_names = [KERNEL_COLS[i] for i in forcing_dims]

    def _bar(val, max_val):
        return '█' * max(1, round(val / max_val * 20))

    print(f"\n  ┌─ Hyperparameters {'─'*45}")
    print(f"  │  GP mean 1/κ          : {model.mean_module.kap_inv.detach().item():.4f}")
    print(f"  │  GP mean b/c/d/e/f    : "
          f"{model.mean_module.b.detach().item():.3f} / "
          f"{model.mean_module.c.detach().item():.3f} / "
          f"{model.mean_module.d.detach().item():.3f} / "
          f"{model.mean_module.e.detach().item():.3f} / "
          f"{model.mean_module.f.detach().item():.3f}")
    print(f"  │  GP mean α/β/γ        : "
          f"{model.mean_module.alpha.detach().item():.3f} / "
          f"{model.mean_module.beta.detach().item():.3f} / "
          f"{model.mean_module.gamma.detach().item():.3f}")
    print(f"  │  Learned noise floors (m/s):")
    for src, idx in SOURCE_MAP.items():
        print(f"  │    {src:<20}  σ = {likelihood.noise_floors[idx].detach().sqrt().item():.4f}")
    print(f"  │")
    print(f"  │  k_morph  σ_f = {model.k_morph.outputscale.detach().sqrt().item():.4f} m/s")
    max_m = morph_ls.max() if morph_ls.ndim > 0 else float(morph_ls)
    for n, v in zip(morph_names, np.atleast_1d(morph_ls)):
        print(f"  │    {n:<18}  ℓ = {v:6.4f}  {_bar(v, max_m)}")
    print(f"  │")
    print(f"  │  k_forcing  σ_f = {model.k_forcing.outputscale.detach().sqrt().item():.4f} m/s")
    max_f = forcing_ls.max() if forcing_ls.ndim > 0 else float(forcing_ls)
    for n, v in zip(forcing_names, np.atleast_1d(forcing_ls)):
        print(f"  │    {n:<18}  ℓ = {v:6.4f}  {_bar(v, max_f)}")
    print(f"  └{'─'*65}")


# =============================================================================
# PART 6 — Velocity profile plot
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def plot_velocity_profile(model, scaler):
    """
    For each station, pick the obs row nearest to PROFILE_HOUR on the first
    available day. Sweep height_ag from PROFILE_HEIGHT_MIN to MAX while holding
    all other features constant. Plot wind speed (x) vs height (y).
    Dots mark the actual observed wind speed at the station's measurement height.
    Dashed horizontal lines mark the canopy top (mean_height).
    """
    heights = np.arange(
        PROFILE_HEIGHT_MIN,
        PROFILE_HEIGHT_MAX + PROFILE_HEIGHT_STEP / 2,
        PROFILE_HEIGHT_STEP,
        dtype=np.float32,
    )

    height_ag_kidx = KERNEL_COLS.index('height_ag')
    height_ag_midx = MEAN_COLS.index('height_ag')

    first_day = df_obs['Datum'].dt.normalize().min()
    ref_ts    = first_day + pd.Timedelta(hours=PROFILE_HOUR)

    n_st     = len(all_stations)
    colours  = [cmc.batlow(i / max(n_st - 1, 1)) for i in range(n_st)]

    fig, ax = plt.subplots(figsize=(6, 8))

    for sid, colour in zip(all_stations, colours):
        df_st = df_obs[df_obs['Standort'] == sid].dropna(
            subset=KERNEL_COLS + MEAN_COLS
        ).copy()
        if df_st.empty:
            print(f"  {sid}: no valid obs rows — skipping")
            continue

        idx_nearest = (df_st['Datum'] - ref_ts).abs().idxmin()
        ref_row     = df_st.loc[idx_nearest]
        print(f"  {sid}: profile at {ref_row['Datum']} "
              f"(requested {ref_ts.strftime('%Y-%m-%d %H:%M UTC')})")

        # Kernel block: repeat ref_row, then vary height_ag column
        Xk_base = np.tile(
            ref_row[KERNEL_COLS].values.astype(np.float32), (len(heights), 1)
        )
        Xk_base[:, height_ag_kidx] = heights
        Xk_s = scaler.transform(Xk_base)

        # Mean block: repeat ref_row, then vary height_ag column
        Xm_base = np.tile(
            ref_row[MEAN_COLS].values.astype(np.float32), (len(heights), 1)
        )
        Xm_base[:, height_ag_midx] = heights

        X   = np.concatenate([Xk_s, Xm_base], axis=1)
        x_t = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            f = model(x_t)

        mean   = np.maximum(f.mean.numpy(), 0.0)
        stddev = f.stddev.numpy()
        lower  = np.maximum(mean - 1.96 * stddev, 0.0)
        upper  = mean + 1.96 * stddev

        ax.fill_betweenx(heights, lower, upper, color=colour, alpha=0.15, zorder=2)
        ax.plot(mean, heights, color=colour, lw=1.8, zorder=3, label=sid)

        # Observed wind speed at measurement height
        h_obs  = float(ref_row.get('height_ag', np.nan))
        ws_obs = float(ref_row[TARGET])
        if np.isfinite(h_obs) and np.isfinite(ws_obs):
            ax.scatter(ws_obs, h_obs, color=colour, marker='o', s=60,
                       zorder=5, edgecolors='white', linewidths=0.8)

        # Canopy top
        H_val = float(ref_row.get('mean_height', np.nan))
        if np.isfinite(H_val) and 0 < H_val < PROFILE_HEIGHT_MAX:
            ax.axhline(H_val, color=colour, lw=0.6, ls='--', alpha=0.5, zorder=1)

    ax.set_xlabel('Wind speed (m/s)')
    ax.set_ylabel('Height above ground (m)')
    ax.set_ylim(0, PROFILE_HEIGHT_MAX + 1)
    ax.set_xlim(left=0)
    ax.set_title(
        f"GPR velocity profiles  —  {ref_ts.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Solid: GP mean  |  shading: 95% CI  |  "
        f"dots: obs at station height  |  dashed: canopy top",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc='upper left', framealpha=0.7)
    plt.tight_layout()

    fname = os.path.join(PLOT_DIR, 'velocity_profile.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Velocity profile saved → {fname}")


# =============================================================================
# PART 7 — Run
# =============================================================================

# Train on everything: real stations + UrbanTales sims + MNW street network
model, likelihood, scaler = fit_model(df_train.copy())

print_hyperparameters(model, likelihood)

plot_velocity_profile(model, scaler)