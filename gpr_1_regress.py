"""
gpr_wind.py
===========
Leave-one-station-out GPR for urban wind speed downscaling.
Reads obs/gpr_obs.pkl and obs/gpr_train_gpr.pkl produced by gpr_0_prep.py.

LOO logic
---------
  For each test station:
    train  = df_train rows where Standort != test_station  (other obs + all sims)
    test   = df_train rows where Standort == test_station  (held-out training hours)
    plot   = df_obs   rows where Standort == test_station  (all hours, June 1–7)

Noise model
-----------
  PerSourceNoiseLikelihood learns a separate additive noise floor (σ²) per
  observation source type ('station', 'urban_tales', 'street_network', …).
  The per-point fixed noise from gpr_0_prep.py (station QC, instrument
  uncertainty) is kept as the base; the learned floors add on top, allowing
  the GP to discover that e.g. LES cells are noisier/quieter than station obs
  as a class, independently of their individual assigned variances.

Outputs
-------
  plots/gpr_wind/ts_<station>.png
  Hyperparameter table per fold + summary across folds
  LOO metrics summary
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
from gpr_0_model import (
    SOURCE_MAP, N_SOURCES, KERNEL_COLS, MEAN_COLS,
    morph_dims, forcing_dims, zust_idx, z_idx, z0_idx, zd_idx, H_idx,
    PerSourceNoiseLikelihood, WindGP, build_model, load_model,
)

# =============================================================================
# CONFIG
# =============================================================================

OBS_PATH   = '../winds/obs/gpr_obs.pkl'
TRAIN_PATH = '../winds/obs/gpr_train_gpr.pkl'
PLOT_DIR   = '../winds/plots/gpr_wind'
os.makedirs(PLOT_DIR, exist_ok=True)

N_ITER      = 2500
LR          = 0.1
PRINT_EVERY = 100
PATIENCE    = 200   # stop if no improvement over this many iters
MIN_DELTA   = 1e-4  # minimum improvement to count



N_RESTARTS  = 1     # number of random restarts per LOO fold
MATERN_NU   = 2.5   # try 1.5, 2.5, or 3.5

# softplus⁻¹(0.09) ≈ -2.35  →  all floors initialise at ~(0.3 m/s)²
_INIT_RAW_FLOOR = -3.7


# =============================================================================
# PART 1 — Load
# =============================================================================

obs_obj   = pd.read_pickle(OBS_PATH)
train_obj = pd.read_pickle(TRAIN_PATH)

df_obs   = obs_obj['df'].copy()
df_train = train_obj['df'].copy()
meta     = train_obj['meta']

USE_SOURCES = ['station', 'urban_tales', 'street_network']
FEAT_COLS = meta['feat_cols']
TARGET    = meta['target_col']
N_FEAT    = len(FEAT_COLS)


print(f"k_morph   dims : {[FEAT_COLS[i] for i in morph_dims]}")
print(f"k_forcing dims : {[FEAT_COLS[i] for i in forcing_dims]}")

df_obs['Datum']   = pd.to_datetime(df_obs['Datum'])
df_train['Datum'] = pd.to_datetime(df_train['Datum'])
df_train = df_train[df_train['source'].isin(USE_SOURCES)].copy()

all_stations = sorted(
    df_train[df_train['source'] == 'station']['Standort']
    .dropna().unique().tolist()
)
df_sim = df_train[df_train['source'] == 'urban_tales'].copy()
df_mnw = df_train[df_train['source'] == 'street_network'].copy()
print(f"UT rows : {len(df_sim)}")
print(f"MNW rows : {len(df_mnw)}")
print(f"Stations  : {all_stations}")
print(f"Features  : {FEAT_COLS}")
print(f"df_obs    : {len(df_obs)} rows  (all June hours)")
print(f"df_train  : {len(df_train)} rows  "
      f"({(df_train['source']=='station').sum()} obs_sub  "
      f"+ {len(df_sim)} sim)")



def fit_and_predict(df_tr, df_te, df_obs_station):
    # Drop any rows with NaNs in required columns before building tensors
    required_cols = KERNEL_COLS + MEAN_COLS
    df_tr          = df_tr.dropna(subset=required_cols).copy()
    df_te          = df_te.dropna(subset=required_cols).copy()
    df_obs_station = df_obs_station.dropna(subset=required_cols).copy()
    
    y_tr = df_tr[TARGET].values.astype(np.float32)

    Xk_tr  = df_tr[KERNEL_COLS].values.astype(np.float32)
    Xk_te  = df_te[KERNEL_COLS].values.astype(np.float32)
    Xk_obs = df_obs_station[KERNEL_COLS].values.astype(np.float32)

    Xm_tr  = df_tr[MEAN_COLS].values.astype(np.float32)
    Xm_te  = df_te[MEAN_COLS].values.astype(np.float32)
    Xm_obs = df_obs_station[MEAN_COLS].values.astype(np.float32)

    scaler   = StandardScaler()
    Xk_tr_s  = scaler.fit_transform(Xk_tr)
    Xk_te_s  = scaler.transform(Xk_te)
    Xk_obs_s = scaler.transform(Xk_obs)

    X_tr      = np.concatenate([Xk_tr_s, Xm_tr],  axis=1)
    X_te      = np.concatenate([Xk_te_s, Xm_te],  axis=1)
    X_obs_all = np.concatenate([Xk_obs_s, Xm_obs], axis=1)

    train_x   = torch.tensor(X_tr,      dtype=torch.float32)
    train_y   = torch.tensor(y_tr,      dtype=torch.float32)
    test_x    = torch.tensor(X_te,      dtype=torch.float32)
    obs_all_x = torch.tensor(X_obs_all, dtype=torch.float32)

    noise_tr = torch.clamp(
        torch.tensor(df_tr['noise_var'].values, dtype=torch.float32),
        min=0.09,
    )
    source_idx = torch.tensor(
        df_tr['source'].map(SOURCE_MAP).fillna(0).values.astype(np.int64),
        dtype=torch.long,
    )

    print(f"  Training on {len(df_tr)} points  "
          f"({(df_tr['source']=='station').sum()} obs  "
          f"+ {(df_tr['source']=='urban_tales').sum()} sim) …")

    best_mll         = float('inf')
    best_model_state = None
    best_lik_state   = None

    for restart in range(N_RESTARTS):
        print(f"  Restart {restart + 1}/{N_RESTARTS} …")

        model, likelihood = build_model(train_x, train_y, noise_tr, source_idx)
        model.train(); likelihood.train()
        optimizer = Adam(model.parameters(), lr=LR)
        mll       = ExactMarginalLogLikelihood(likelihood, model)

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
                print(f"    iter {i:4d}  MLL={-loss_val:8.4f}  "
                      f"noise_floors(m/s): {floors_str}  no_improve={no_improve}")

            if no_improve >= PATIENCE:
                print(f"    Early stop at iter {i}  "
                      f"(no improvement for {PATIENCE} iters)")
                break

        print(f"  Restart {restart + 1} final MLL={-best_loss:.4f}")
        if best_loss < best_mll:
            best_mll         = best_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_lik_state   = {k: v.clone() for k, v in likelihood.state_dict().items()}

    model.load_state_dict(best_model_state)
    likelihood.load_state_dict(best_lik_state)
    model.eval()
    likelihood.eval()
    print(f"  Best restart MLL={-best_mll:.4f}")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        f_te      = model(test_x)
        f_obs_all = model(obs_all_x)

    def _attach(df, f):
        mean   = f.mean.numpy()
        stddev = f.stddev.numpy()
        out = df.copy()
        out['pred']       = np.maximum(mean, 0.0)
        out['pred_lower'] = np.maximum(mean - 1.96 * stddev, 0.0)
        out['pred_upper'] = mean + 1.96 * stddev
        return out

    return _attach(df_te, f_te), _attach(df_obs_station, f_obs_all), \
           model, likelihood, scaler, best_model_state, best_lik_state, \
           noise_tr, source_idx
# =============================================================================
# PART 5 — Hyperparameters
# =============================================================================

def extract_hyperparameters(model, likelihood):
    morph_ls   = model.k_morph.base_kernel.lengthscale.detach().squeeze().numpy()
    forcing_ls = model.k_forcing.base_kernel.lengthscale.detach().squeeze().numpy()
    morph_names   = [FEAT_COLS[i] for i in morph_dims]
    forcing_names = [FEAT_COLS[i] for i in forcing_dims]

    hp = {
        'mean_kap_inv': model.mean_module.kap_inv.detach().item(),
        'mean_b':       model.mean_module.b.detach().item(),
        'mean_c':       model.mean_module.c.detach().item(),
        'mean_d':       model.mean_module.d.detach().item(),
        'mean_e':       model.mean_module.e.detach().item(),
        'mean_f':       model.mean_module.f.detach().item(),
        'mean_alpha':   model.mean_module.alpha.detach().item(),
        'mean_beta':    model.mean_module.beta.detach().item(),
        'mean_gamma':   model.mean_module.gamma.detach().item(),
        'sigma_f_morph':   model.k_morph.outputscale.detach().sqrt().item(),
        'sigma_f_forcing': model.k_forcing.outputscale.detach().sqrt().item(),
        **{f'ls_m_{n}': float(v)
           for n, v in zip(morph_names, np.atleast_1d(morph_ls))},
        **{f'ls_f_{n}': float(v)
           for n, v in zip(forcing_names, np.atleast_1d(forcing_ls))},
    }
    # Per-source noise floors (converted to m/s, i.e. std-dev units)
    for src, idx in SOURCE_MAP.items():
        hp[f'noise_floor_{src}'] = (
            likelihood.noise_floors[idx].detach().sqrt().item()
        )
    return hp


def print_hyperparameters(hp, station_id):
    morph_names   = [FEAT_COLS[i] for i in morph_dims]
    forcing_names = [FEAT_COLS[i] for i in forcing_dims]

    def _bar(val, max_val):
        return '█' * max(1, round(val / max_val * 20))

    print(f"\n  ┌─ Hyperparameters [{station_id}] {'─'*35}")
    print(f"  │  GP mean 1/κ          : {hp['mean_kap_inv']:.4f}")
    print(f"  │  GP mean b/c/d/e/f    : "
          f"{hp['mean_b']:.3f} / {hp['mean_c']:.3f} / "
          f"{hp['mean_d']:.3f} / {hp['mean_e']:.3f} / {hp['mean_f']:.3f}")
    print(f"  │  GP mean α/β/γ        : "
          f"{hp['mean_alpha']:.3f} / {hp['mean_beta']:.3f} / {hp['mean_gamma']:.3f}")
    print(f"  │  Learned noise floors (m/s):")
    for src in SOURCE_MAP:
        key = f'noise_floor_{src}'
        if key in hp:
            print(f"  │    {src:<20}  σ = {hp[key]:.4f}")
    print(f"  │")
    print(f"  │  k_morph  σ_f = {hp['sigma_f_morph']:.4f} m/s")
    m_vals = [hp[f'ls_m_{n}'] for n in morph_names]
    max_m  = max(m_vals) if m_vals else 1.0
    for n, v in zip(morph_names, m_vals):
        print(f"  │    {n:<18}  ℓ = {v:6.4f}  {_bar(v, max_m)}")
    print(f"  │")
    print(f"  │  k_forcing  σ_f = {hp['sigma_f_forcing']:.4f} m/s")
    f_vals = [hp[f'ls_f_{n}'] for n in forcing_names]
    max_f  = max(f_vals) if f_vals else 1.0
    for n, v in zip(forcing_names, f_vals):
        print(f"  │    {n:<18}  ℓ = {v:6.4f}  {_bar(v, max_f)}")
    print(f"  └{'─'*55}")


def print_hyperparameter_summary(all_hp):
    if not all_hp:
        return
    df_hp = pd.DataFrame(all_hp).set_index('city')
    print(f"\n{'='*70}")
    print("Hyperparameter summary across LOO folds")
    print(f"{'='*70}")
    print(df_hp.round(4).to_string())
    print(f"{'─'*70}")
    print("Mean across folds:")
    print(df_hp.mean().round(4).to_string())
    print(f"{'='*70}\n")


# =============================================================================
# PART 6 — Time series plot (June 1–7)
# =============================================================================

C_OBS = cmc.batlow(0.05)
C_GPR = cmc.batlow(0.75)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


def plot_timeseries(df_te_pred, df_obs_all_pred, label, metrics):
    # Use first 7 days of whatever date range is actually present
    t0 = df_obs_all_pred['Datum'].min().normalize()
    t1 = t0 + pd.Timedelta(days=7)

    df_cont  = df_obs_all_pred[
        (df_obs_all_pred['Datum'] >= t0) & (df_obs_all_pred['Datum'] < t1)
    ].sort_values('Datum')
    df_cross = df_te_pred[
        (df_te_pred['Datum'] >= t0) & (df_te_pred['Datum'] < t1)
    ].sort_values('Datum')

    if df_cont.empty:
        print(f"  No June 1–7 data for {station_id} — skipping plot")
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(df_cont['Datum'], df_cont[TARGET],
            color='#aaaaaa', lw=0.8, alpha=0.6, zorder=1)
    ax.fill_between(df_cont['Datum'],
                    df_cont['pred_lower'], df_cont['pred_upper'],
                    color=C_GPR, alpha=0.18, zorder=2)
    ax.plot(df_cont['Datum'], df_cont['pred'],
            color=C_GPR, lw=1.6, zorder=3)
    if not df_cross.empty:
        ax.scatter(df_cross['Datum'], df_cross[TARGET],
                   marker='x', s=70, linewidths=2.0,
                   color=C_OBS, zorder=4)

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    for tick, ts in zip(ax.get_xticklabels(),
                        pd.date_range(t0, t1, freq='12h',
                                      inclusive='left', tz='UTC')):
        tick.set_color('#999999' if ts.hour == 12 else 'black')
        if ts.hour == 0:
            tick.set_fontweight('bold')
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlim(t0, t1)
    ax.set_ylabel('Wind speed (m/s)')

    r2, r, rmse, bias = metrics
    ax.set_title(
        f"R²={r2:.3f}  R={r:.3f}  RMSE={rmse:.3f}  Bias={bias:+.3f} m/s",
        fontsize=10,
    )
    ax.legend(handles=[
        Line2D([0], [0], color='#aaaaaa', lw=0.8, alpha=0.8,
               label='Obs (all hours)'),
        Line2D([0], [0], color=C_OBS, marker='x', lw=0,
               markersize=8, markeredgewidth=2, label='Obs (training hours)'),
        Line2D([0], [0], color=C_GPR, lw=1.6, label='GPR mean'),
        plt.fill([], [], color=C_GPR, alpha=0.35, label='95% CI')[0],
    ], fontsize=8, loc='upper right')

    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, f"ts_{label}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {fname}")


# =============================================================================
# PART 7 — Leave-one-city-out loop
# =============================================================================

def compute_metrics(obs, pred):
    r2   = r2_score(obs, pred)
    r    = pearsonr(obs, pred)[0]
    rmse = np.sqrt(np.mean((pred - obs) ** 2))
    bias = np.mean(pred - obs)
    return r2, r, rmse, bias


# Identify from data
geo_cities = sorted(
    df_train[df_train['source'].isin(['station', 'street_network'])]
    ['geo_city'].dropna().unique().tolist()
)
print(f"City folds : {geo_cities}")

# Hardcoded
# geo_cities = {'zurich'}

all_results = []
all_hp      = []

for test_city in geo_cities:
    print(f"\n{'='*60}")
    print(f"Test city : {test_city.upper()}")
    print(f"{'='*60}")

    # Test: all obs from the held-out city (stations + MNW), all June hours
    # Train: everything else — other cities' stations + MNW + all sims
    mask_test = df_train['geo_city'] == test_city
    mask_sim  = df_train['source'] == 'urban_tales'

    df_te_raw = df_train[mask_test].copy()
    df_tr     = df_train[~mask_test | mask_sim].copy()   # other cities + all sims

    # df_obs for plotting: all June hours from test city stations
    test_stations = df_te_raw['Standort'].unique().tolist()
    df_obs_city   = df_obs[df_obs['Standort'].isin(test_stations)].copy()

    print(f"  Train : {len(df_tr)} rows  "
          f"({(df_tr['source']=='station').sum()} obs  "
          f"+ {(df_tr['source']=='urban_tales').sum()} sim  "
          f"+ {(df_tr['source']=='street_network').sum()} mnw)")
    print(f"  Test  : {len(df_te_raw)} rows  "
          f"({df_te_raw['Standort'].nunique()} stations)")

    if df_te_raw.empty:
        print("  No test data — skipping")
        continue

    df_te, df_obs_all_pred, model, likelihood, scaler, \
        best_model_state, best_lik_state, noise_tr, source_idx = fit_and_predict(
            df_tr, df_te_raw, df_obs_city
        )

    hp = extract_hyperparameters(model, likelihood)
    hp['city'] = test_city
    all_hp.append(hp)
    print_hyperparameters(hp, test_city)

    # Metrics per station within the held-out city
    print(f"\n  Per-station metrics ({test_city}):")
    city_results = []
    for sid in sorted(test_stations):
        df_sid = df_obs_all_pred[df_obs_all_pred['Standort'] == sid][[TARGET, 'pred']].dropna()
        if len(df_sid) < 2:
            continue
        m = compute_metrics(df_sid[TARGET].values, df_sid['pred'].values)
        r2, r, rmse, bias = m
        src = df_te_raw[df_te_raw['Standort'] == sid]['source'].iloc[0] \
              if sid in df_te_raw['Standort'].values else '?'
        print(f"    {sid:<30}  [{src:<14}]  "
              f"R²={r2:.3f}  R={r:.3f}  RMSE={rmse:.3f}  Bias={bias:+.3f}")
        city_results.append({'station': sid, 'city': test_city, 'source': src,
                             'R²': r2, 'R': r, 'RMSE': rmse, 'Bias': bias,
                             'N': len(df_sid)})

    if city_results:
        df_city = pd.DataFrame(city_results)
        print(f"\n  City mean  R²={df_city['R²'].mean():.3f}  "
              f"R={df_city['R'].mean():.3f}  "
              f"RMSE={df_city['RMSE'].mean():.3f}  "
              f"Bias={df_city['Bias'].mean():+.3f}")
        all_results.extend(city_results)

    # Save model checkpoint for grid prediction
    os.makedirs('obs/models', exist_ok=True)
    save_path = f'obs/models/gpr_{test_city}.pt'
    torch.save({
        'model_state':      best_model_state,
        'likelihood_state': best_lik_state,
        'scaler_mean':      scaler.mean_,
        'scaler_scale':     scaler.scale_,
        'noise_tr':         noise_tr,
        'source_idx':       source_idx,
        'hp':               hp,
        'test_city':        test_city,
    }, save_path)
    print(f"  Model saved → {save_path}")

    for sid in sorted(test_stations):
            df_sid_obs = df_obs_all_pred[df_obs_all_pred['Standort'] == sid].copy()
            df_sid_te  = df_te[df_te['Standort'] == sid].copy()
            if df_sid_obs.empty:
                continue
            sid_valid = df_sid_obs[[TARGET, 'pred']].dropna()
            if len(sid_valid) < 2:
                continue
            m_sid = compute_metrics(sid_valid[TARGET].values, sid_valid['pred'].values)
            plot_timeseries(df_sid_te, df_sid_obs,
                            f"{test_city}_{sid}", m_sid)


# =============================================================================
# PART 8 — Summaries
# =============================================================================

print_hyperparameter_summary(all_hp)

if all_results:
    df_res = pd.DataFrame(all_results)
    print(f"\n{'='*60}")
    print("City-level LOO metrics:")
    print(df_res.groupby('city')[['R²','R','RMSE','Bias']].mean().round(3).to_string())
    print(f"\nOverall mean:")
    print(f"  R²   = {df_res['R²'].mean():.3f}")
    print(f"  R    = {df_res['R'].mean():.3f}")
    print(f"  RMSE = {df_res['RMSE'].mean():.3f} m/s")
    print(f"  Bias = {df_res['Bias'].mean():+.3f} m/s")

    # Also break down by source type within cities
    print(f"\nBy source type:")
    print(df_res.groupby('source')[['R²','R','RMSE','Bias']].mean().round(3).to_string())