import xarray as xr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import rasterio
from cmcrameri import cm

#change paths for nc and png as well as max height as seen on png
path = 'zurich_4'
os.makedirs('../winds/urban_tales/'+ path, exist_ok=True)
max_height = 24.0


ds_path = '../winds/urban_tales/'+ path +'/CH-Zur-V2_d90_ped.nc'

# ── 0. Load uped (needed for domain size) ────────────────────────────────────
ds   = xr.open_dataset(ds_path)
uped = ds.Uped
uped_2d = uped.mean(dim='time').values if 'time' in uped.dims else uped.values
uped_2d = uped_2d[::-1, :]   # flip y-axis to match raster convention (top=north)
print(f"uped 2D shape: {uped_2d.shape}")

fig, ax = plt.subplots(figsize=(6, 8))
im = ax.imshow(uped_2d, cmap='viridis', origin='upper')
plt.colorbar(im, ax=ax, label='m/s')
ax.set_title(f'Raw Uped ({uped_2d.shape[0]}×{uped_2d.shape[1]})')
plt.savefig('../winds/urban_tales/' + path + '/uped_raw.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved uped_raw.png")

# ── 1. PNG → building heights, resized to PALM domain ────────────────────────
img = Image.open('../winds/urban_tales/' + path + '/CH-Zur-V2_d90.png').convert('L')
arr = np.array(img)


non_white   = arr < 250
h, w        = arr.shape
row_density = non_white.sum(axis=1) / w
col_density = non_white.sum(axis=0) / h

dense_rows = np.where(row_density > 0.3)[0]
dense_cols = np.where(col_density > 0.3)[0]

padding = 3
rmin = dense_rows[0]  + 1 + padding
rmax = dense_rows[-1]     - padding
cmin = dense_cols[0]  + 1 + padding
cmax = dense_cols[-1]     - padding

print(f"Detected map bounds: rows {rmin}:{rmax}, cols {cmin}:{cmax}")
map_arr = arr[rmin:rmax, cmin:cmax]

# Downsample to exact domain size (288 wide, 384 tall)
# Resize PNG to match PALM domain exactly
palm_h, palm_w = uped_2d.shape
print(palm_h, palm_w)
map_img_small  = Image.fromarray(map_arr).resize((palm_w, palm_h), Image.NEAREST)
map_arr_small  = np.array(map_img_small) 
# map_img_small = Image.fromarray(map_arr).resize((288, 384), Image.NEAREST)
# map_arr_small = np.array(map_img_small)

# Rescale grayscale to building height: white=0m, black=16m
height_m = (255 - map_arr_small) / 255.0 * max_height
height_m[map_arr_small > 250] = np.nan  # background → NaN

# Save check PNG and TIFF
fig, ax = plt.subplots(figsize=(6, 8))
ax.imshow(map_arr_small, cmap='gray', vmin=0, vmax=255)
ax.set_title(f'Cropped + downsampled: {map_arr_small.shape}')
plt.savefig('../winds/urban_tales/'+ path +'/crop_check.png', dpi=150, bbox_inches='tight')
plt.close()

Image.fromarray(height_m.astype(np.float32)).save('../winds/urban_tales/'+ path +'/building_heights.tiff')
print(f"Saved building_heights.tiff — height range: "
      f"{np.nanmin(height_m):.2f} – {np.nanmax(height_m):.2f} m")

# ── 1. Load + coarsen uped ────────────────────────────────────────────────────


nr, nc   = uped_2d.shape[0] // 30, uped_2d.shape[1] // 30
uped_30m = np.nanmean(uped_2d[:nr*30, :nc*30].reshape(nr, 30, nc, 30), axis=(1, 3))


# uped_trim   = uped_2d[24:, :270]          # trim to 360×270
# uped_blocks = uped_trim.reshape(12, 30, 9, 30)
# uped_30m    = np.nanmean(uped_blocks, axis=(1, 3))   # (12, 9)

ds_out = xr.Dataset({
    'Uped_30m': xr.DataArray(
        uped_30m, dims=['y', 'x'],
        attrs={'long_name': 'Mean pedestrian wind speed', 'units': 'm/s', 'cell_size': '30m'}
    )
})
ds_out.to_netcdf('../winds/urban_tales/'+ path +'/uped_30m.nc')
print("Saved to ", '../winds/urban_tales/'+ path +'/uped_30m.nc')

fig, ax = plt.subplots(figsize=(5, 7))
im = ax.imshow(uped_30m, cmap='viridis', origin='upper')
plt.colorbar(im, ax=ax, label='m/s')
ax.set_title('Uped 30m mean')
plt.savefig('../winds/urban_tales/'+ path +'/uped_30m.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 2. Coarsen building height → density + mean height at 30m ────────────────
# height_trim   = height_m[24:, :270]       # same trim as uped
# height_blocks = height_trim.reshape(12, 30, 9, 30)
# valid_mask    = ~np.isnan(height_blocks)

height_blocks = height_m[:nr*30, :nc*30].reshape(nr, 30, nc, 30)
valid_mask    = ~np.isnan(height_blocks)

density_30m = valid_mask.sum(axis=(1, 3)) / (30 * 30)

n_valid    = valid_mask.sum(axis=(1, 3))
height_30m = np.where(
    n_valid > 0,
    np.nansum(height_blocks, axis=(1, 3)) / np.maximum(n_valid, 1),
    0.0
)

print(f"Density range: {density_30m.min():.2f} – {density_30m.max():.2f}")
print(f"Height range:  {height_30m.min():.2f} – {height_30m.max():.2f} m")

Image.fromarray(density_30m.astype(np.float32)).save('../winds/urban_tales/'+ path +'/density_30m.tiff')
Image.fromarray(height_30m.astype(np.float32)).save('../winds/urban_tales/'+ path +'/height_30m.tiff')
print("Saved density_30m.tiff, height_30m.tiff")

for data, title, cmap, fname, label in [
    (density_30m, 'Building density (30m)',   'YlOrRd', 'density_30m.png', '0–1'),
    (height_30m,  'Mean building height (30m)', 'plasma', 'height_30m.png', 'm'),
]:
    fig, ax = plt.subplots(figsize=(5, 7))
    im = ax.imshow(data, cmap=cmap, origin='upper')
    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(title)
    plt.savefig('../winds/urban_tales/'+ path +f'/{fname}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")




def quick_plot(data, label, path):
        h, w = data.shape
        dpi = 150
        fig, ax = plt.subplots(figsize=(15, 15), dpi=dpi)
        valid = data[data > 0]
        vmax = float(np.percentile(valid, 98)) if valid.size else float(data.max())
        im = ax.imshow(data, cmap=cm.batlow, vmin=0, vmax=vmax, interpolation='nearest')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
        ax.set_axis_off()
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {os.path.abspath(path)}")



def calculate_roughness_simple_30m(output_resolution=30, input_resolution=10,
                                       min_building_height=3.0):
    """Calculate roughness parameters at 30m from 10m inputs"""


    with rasterio.open('../winds/urban_tales/'+ path +'/density_30m.tiff') as lp:
        lambda_p_combined = lp.read(1)
    
    with rasterio.open('../winds/urban_tales/'+ path +'/height_30m.tiff') as mh:
        mh_30 = mh.read(1)
    


    # quick_plot(lambda_p_combined, 'Plan area fraction λp [-]', 'lambda_p_combined.png')

    # ── 8. Lambda f ───────────────────────────────────────────────────────
    grid_cell_area    = output_resolution ** 2
    kanda_mask        = (lambda_p_combined > 0.05) & (lambda_p_combined <= 0.45)
    lambda_f_combined = np.zeros_like(mh_30)

    lambda_f_combined[kanda_mask]  = (1.42 * lambda_p_combined[kanda_mask]**2 +
                                      0.4  * lambda_p_combined[kanda_mask])
    lambda_f_combined[~kanda_mask] = (mh_30[~kanda_mask] *
                                      np.sqrt(lambda_p_combined[~kanda_mask] / grid_cell_area))

    print(f"  ✓ lambda_p_combined range: {lambda_p_combined.min():.4f} - {lambda_p_combined.max():.4f}")
    print(f"  ✓ Mean height range:       {mh_30.min():.2f} - {mh_30.max():.2f} m")

    quick_plot(lambda_p_combined,  'Plan area fraction λp [-]',    '../winds/urban_tales/'+ path + '/lambda_p_combined.png')
    quick_plot(lambda_f_combined,  'Frontal area fraction λf [-]', '../winds/urban_tales/'+ path + '/lambda_f_combined.png')
    quick_plot(mh_30,              'Mean height [m]',              '../winds/urban_tales/'+ path + '/mh_30.png')



    alpha = 4.43
    beta = 1.0
    Cd = 1.2
    kappa = 0.4
    
    has_roughness = mh_30 > 1
    z0 = np.full_like(lambda_p_combined, 0.000001, dtype=np.float32)
    zd = np.full_like(lambda_p_combined, 0.02, dtype=np.float32)
    
    if np.any(has_roughness):
        lambda_p_rough = lambda_p_combined[has_roughness]
        lambda_f_rough = lambda_f_combined[has_roughness]
        mean_height_rough = mh_30[has_roughness]
        
        zd_over_H = 1 + (lambda_p_rough - 1) * alpha ** (-lambda_p_rough)
        zd_rough = zd_over_H * mean_height_rough
        # zd_rough = np.maximum(zd_rough, 0.02)
        zd_rough =np.clip(zd_rough, 0.05, mean_height_rough*0.65)

        
        frac_term = 1 - zd_rough / mean_height_rough
        exponent_term = 1/(np.sqrt(0.5 * beta * Cd * frac_term * lambda_f_rough / kappa**2))
        z0_over_H = frac_term * np.exp(-exponent_term)
        z0_rough = z0_over_H * mean_height_rough
        z0_rough = np.maximum(z0_rough, 0.001)
        
        z0[has_roughness] = z0_rough
        zd[has_roughness] = zd_rough

    quick_plot(z0,              'Roughness length [m]',              '../winds/urban_tales/'+ path + '/z0.png')
    quick_plot(zd,              'Displacement height[m]',            '../winds/urban_tales/'+ path + '/zd.png')



    with rasterio.open(
        '../winds/urban_tales/'+ path +'/z0.tiff',
        'w',
        driver='GTiff',
        height=z0.shape[0],
        width=z0.shape[1],
        count=1,
        dtype=z0.dtype,
        # crs='+proj=latlong',
        # transform=transform,
    ) as dst:
        dst.write(z0, 1)

    with rasterio.open(
        '../winds/urban_tales/'+ path +'/zd.tiff',
        'w',
        driver='GTiff',
        height=zd.shape[0],
        width=zd.shape[1],
        count=1,
        dtype=z0.dtype,
        # crs='+proj=latlong',
        # transform=transform,
    ) as dst:
        dst.write(zd, 1)

    
    return z0, zd


z0, zd = calculate_roughness_simple_30m()



# ── 3. Direction bands at native resolution → coarsened to 30m ───────────────
bool_mask = np.where(np.isnan(height_m), 0, (height_m > 10).astype(np.uint8))

# Coarsen 1m boolean mask to 10m (any building in 10x10 block → 1)
nr_10 = bool_mask.shape[0] // 10
nc_10 = bool_mask.shape[1] // 10
bool_mask_10m = (bool_mask[:nr_10*10, :nc_10*10]
                 .reshape(nr_10, 10, nc_10, 10)
                 .max(axis=(1, 3))
                 .astype(np.uint8))

# Neighbor check at 10m resolution
data      = bool_mask_10m
c         = data[1:-1, 1:-1]
top       = data[0:-2, 1:-1];  bottom    = data[2:,   1:-1]
left      = data[1:-1, 0:-2];  right     = data[1:-1, 2:  ]
top_left  = data[0:-2, 0:-2];  top_right = data[0:-2, 2:  ]
bot_left  = data[2:,   0:-2];  bot_right = data[2:,   2:  ]

mask_0    = (c == 0)
dir_bands = np.zeros((4, *c.shape), dtype=np.uint8)
dir_bands[0] = np.where(mask_0 & (top == 0)      & (bottom == 0),    1, 0)
dir_bands[1] = np.where(mask_0 & (bot_left == 0) & (top_right == 0), 1, 0)
dir_bands[2] = np.where(mask_0 & (left == 0)     & (right == 0),     1, 0)
dir_bands[3] = np.where(mask_0 & (top_left == 0) & (bot_right == 0), 1, 0)

# Coarsen to 30m by summing 3x3 blocks → max value 9
nr_d = dir_bands.shape[1] // 3
nc_d = dir_bands.shape[2] // 3

for i, name in enumerate(['vertical', 'diag_ne', 'horizontal', 'diag_se']):
    coarsened = (dir_bands[i, :nr_d*3, :nc_d*3]
                 .reshape(nr_d, 3, nc_d, 3)
                 .sum(axis=(1, 3))
                 .astype(np.uint8))
    out_path = f'../winds/urban_tales/{path}/{name}_30m.tiff'
    with rasterio.open(out_path, 'w', driver='GTiff',
                       height=coarsened.shape[0], width=coarsened.shape[1],
                       count=1, dtype=coarsened.dtype) as dst:
        dst.write(coarsened, 1)
    print(f"Saved {out_path}  shape={coarsened.shape}  max={coarsened.max()}")