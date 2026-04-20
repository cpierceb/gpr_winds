"""
gpr_0_prep.py
=============
Builds two pickles for gpr_wind.py from raw data sources.

  gpr_obs.pkl       — all June station observations with full feature set,
                      pooled across all cities in CITIES.
  gpr_train_gpr.pkl — subsampled obs (configurable hours) + UT sim snapshots.

Urban Tales snapshots are city-agnostic (Zurich PALM runs) and are always
appended to the training set regardless of which cities are active.

To add a new city:
  1. Write a load_observations_<city>() function returning (df_wide, stations_meta).
  2. Add a CITY_CONFIGS entry with raster paths, noise map, exclude list.
  3. Add the city string to CITIES.
"""

import numpy as np
import pandas as pd
import rasterio
import json
import glob
import os
import xarray as xr
from pyproj import Transformer
from collections import defaultdict

# =============================================================================
# TOP-LEVEL CONFIG
# =============================================================================

# CITIES = ['zurich']
CITIES = ['zurich', 'milan', 'mnw']

TRAIN_HOURS = [0, 12]

# Default date range (2018). Cities with different years override via CITY_CONFIGS.
JUNE_START = pd.Timestamp('2018-06-01', tz='UTC')
JUNE_END   = pd.Timestamp('2018-06-30 23:00:00', tz='UTC')

TARGET_COL = 'WVv'
FEAT_COLS  = ['lambda_p', 'mean_height', 'elev_diff', 'height_ag',
              'zust', 'wd_vert', 'wd_diag_ne', 'wd_horiz', 'wd_diag_se']

SOURCE_MAP = {
    'station':        0,
    'urban_tales':    1,
    'street_network': 2,
}
N_SOURCES = len(SOURCE_MAP)

NOISE_UT = 0.1

OUT_OBS   = '../winds/obs/gpr_obs.pkl'
OUT_TRAIN = '../winds/obs/gpr_train_gpr.pkl'
os.makedirs('obs', exist_ok=True)

# Default ERA5 glob (2018). Cities with different years override via CITY_CONFIGS.
ERA5_GLOB = '../raw_data/era5/*2018*.nc'

# ── Urban Tales snapshots (Zurich PALM — independent of CITIES) ───────────────
UT_SNAPSHOTS = [
    {
        'uped_nc':      'urban_tales/zurich/uped_30m.nc',
        'density_tiff': 'urban_tales/zurich/density_30m.tiff',
        'height_tiff':  'urban_tales/zurich/height_30m.tiff',
        'z0_tif':       'urban_tales/zurich/z0.tiff',
        'zd_tif':       'urban_tales/zurich/zd.tiff',
        'wd_vert_tif':    'urban_tales/zurich/vertical_30m.tiff',
        'wd_diag_ne_tif': 'urban_tales/zurich/diag_ne_30m.tiff',
        'wd_horiz_tif':   'urban_tales/zurich/horizontal_30m.tiff',
        'wd_diag_se_tif': 'urban_tales/zurich/diag_se_30m.tiff',
        'zust': 0.208309066, 'wd': 0.0,  'height_ag': 1.75,
        'timestamp': pd.Timestamp('2018-06-01 12:00:00', tz='UTC'),
    },
    {
        'uped_nc':      'urban_tales/zurich_2/uped_30m.nc',
        'density_tiff': 'urban_tales/zurich_2/density_30m.tiff',
        'height_tiff':  'urban_tales/zurich_2/height_30m.tiff',
        'z0_tif':       'urban_tales/zurich_2/z0.tiff',
        'zd_tif':       'urban_tales/zurich_2/zd.tiff',
        'wd_vert_tif':    'urban_tales/zurich_2/vertical_30m.tiff',
        'wd_diag_ne_tif': 'urban_tales/zurich_2/diag_ne_30m.tiff',
        'wd_horiz_tif':   'urban_tales/zurich_2/horizontal_30m.tiff',
        'wd_diag_se_tif': 'urban_tales/zurich_2/diag_se_30m.tiff',
        'zust': 0.208309066, 'wd': 90.0, 'height_ag': 1.75,
        'timestamp': pd.Timestamp('2018-06-02 12:00:00', tz='UTC'),
    },
    {
        'uped_nc':      'urban_tales/zurich_3/uped_30m.nc',
        'density_tiff': 'urban_tales/zurich_3/density_30m.tiff',
        'height_tiff':  'urban_tales/zurich_3/height_30m.tiff',
        'z0_tif':       'urban_tales/zurich_3/z0.tiff',
        'zd_tif':       'urban_tales/zurich_3/zd.tiff',
        'wd_vert_tif':    'urban_tales/zurich_3/vertical_30m.tiff',
        'wd_diag_ne_tif': 'urban_tales/zurich_3/diag_ne_30m.tiff',
        'wd_horiz_tif':   'urban_tales/zurich_3/horizontal_30m.tiff',
        'wd_diag_se_tif': 'urban_tales/zurich_3/diag_se_30m.tiff',
        'zust': 0.207822345, 'wd': 0.0,  'height_ag': 1.75,
        'timestamp': pd.Timestamp('2018-06-03 12:00:00', tz='UTC'),
    },
    {
        'uped_nc':      'urban_tales/zurich_4/uped_30m.nc',
        'density_tiff': 'urban_tales/zurich_4/density_30m.tiff',
        'height_tiff':  'urban_tales/zurich_4/height_30m.tiff',
        'z0_tif':       'urban_tales/zurich_4/z0.tiff',
        'zd_tif':       'urban_tales/zurich_4/zd.tiff',
        'wd_vert_tif':    'urban_tales/zurich_4/vertical_30m.tiff',
        'wd_diag_ne_tif': 'urban_tales/zurich_4/diag_ne_30m.tiff',
        'wd_horiz_tif':   'urban_tales/zurich_4/horizontal_30m.tiff',
        'wd_diag_se_tif': 'urban_tales/zurich_4/diag_se_30m.tiff',
        'zust': 0.207822345, 'wd': 90.0, 'height_ag': 1.75,
        'timestamp': pd.Timestamp('2018-06-04 12:00:00', tz='UTC'),
    },
    {
        'uped_nc':      'urban_tales/basel/uped_30m.nc',
        'density_tiff': 'urban_tales/basel/density_30m.tiff',
        'height_tiff':  'urban_tales/basel/height_30m.tiff',
        'z0_tif':       'urban_tales/basel/z0.tiff',
        'zd_tif':       'urban_tales/basel/zd.tiff',
        'wd_vert_tif':    'urban_tales/basel/vertical_30m.tiff',
        'wd_diag_ne_tif': 'urban_tales/basel/diag_ne_30m.tiff',
        'wd_horiz_tif':   'urban_tales/basel/horizontal_30m.tiff',
        'wd_diag_se_tif': 'urban_tales/basel/diag_se_30m.tiff',
        'zust': 0.204920596, 'wd': 0.0, 'height_ag': 1.75,
        'timestamp': pd.Timestamp('2018-06-04 12:00:00', tz='UTC'),
    },
]


# =============================================================================
# PER-CITY CONFIG
# =============================================================================

def city_paths(city):
    return {
        'dtm_path':        f'../tiffs/dtm/{city}_30.tif',
        'lambda_p_tif':    f'../tiffs/z0/{city}_lambda_p_30.tif',
        'mean_height_tif': f'../tiffs/z0/{city}_mean_height_30.tif',
        'z0_tif':          f'../tiffs/z0/{city}_30.tif',
        'zd_tif':          f'../tiffs/zd/{city}_30.tif',
        'wd_vert_tif':     f'../tiffs/dir/{city}/{city}_vertical_30m.tif',
        'wd_diag_ne_tif':  f'../tiffs/dir/{city}/{city}_diag_ne_30m.tif',
        'wd_horiz_tif':    f'../tiffs/dir/{city}/{city}_horizontal_30m.tif',
        'wd_diag_se_tif':  f'../tiffs/dir/{city}/{city}_diag_se_30m.tif',
    }

CITY_CONFIGS = {
    'zurich': {
        **city_paths('zurich'),
        'station_height_ag': 10.0,
        'exclude_stations':  ['UEB'],
        'noise_map': {
            'SMA':                     0.1,
            'REH':                     0.1,
            'Zch_Rosengartenstrasse':  0.2,
            'Zch_Schimmelstrasse':     0.2,
            'Zch_Stampfenbachstrasse': 0.2,
        },
        'height_ag_map': {
            'Zch_Rosengartenstrasse':  2.0,
            'Zch_Schimmelstrasse':     2.0,
            'Zch_Stampfenbachstrasse': 2.0,
        },
        'noise_default': 0.2,
    },
    'milan': {
        **city_paths('milan'),
        'station_height_ag': 10.0,
        'exclude_stations':  [],
        'noise_map': {
            'ZAVATTARI': 0.2,
            'BRERA':     0.2,
            'MARCHE':    0.2,
            'LAMBRATE':  0.2,
            'JUVARA':    0.2,
        },
        'noise_default': 0.3,
    },
    'rome': {
        **city_paths('rome'),
        'station_height_ag': 10.0,
        'exclude_stations':  [],
        'noise_map':         {},
        'noise_default':     0.2,
    },
    'mnw': {
        'filter_to_cities':  ['milan', 'rome'],
        'station_height_ag': 2.0,
        'exclude_stations':  [],
        'noise_map':         {},
        'noise_default':     0.4,
        'era5_glob':         '../raw_data/era5/*2022*.nc',
        'june_start':        pd.Timestamp('2022-05-01', tz='UTC'),
        'june_end':          pd.Timestamp('2022-05-31 23:00:00', tz='UTC'),
        'json_paths':        ['../winds/obs/italy/mnw_may_22.json'],
    },
}

# =============================================================================
# RASTER HELPERS
# =============================================================================

_cache = {}

def get_raster(path, nodata=None, fill=0.0):
    if path not in _cache:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            if nodata is not None:
                data[data == nodata] = fill
            _cache[path] = {'data': data, 'transform': src.transform}
    return _cache[path]

def sample_nearest(info, x, y):
    row, col = rasterio.transform.rowcol(info['transform'], x, y)
    d = info['data']
    if 0 <= row < d.shape[0] and 0 <= col < d.shape[1]:
        return float(d[row, col])
    return np.nan


# =============================================================================
# CITY OBSERVATION LOADERS
# =============================================================================

def load_observations_mnw(cfg):
    """
    Loads wind observations from BUFR-JSON files.

    Each line is one JSON object. Wind direction (B11001) and wind speed
    (B11002) appear on separate lines for the same station + timestamp —
    they are merged by grouping on (Datum, Standort).

    Only stations whose lat/lon fall within the Milan raster extent are kept.
    elev_diff is computed in the city loop (local DTM minus ERA5 geopotential),
    exactly as for all other station types.
    """

    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)

    # Derive WGS84 bounding box from the Milan lambda_p raster
    tr_inv = Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
    city_boxes = []
    for fc in cfg.get('filter_to_cities', []):
        with rasterio.open(CITY_CONFIGS[fc]['lambda_p_tif']) as src:
            b = src.bounds
        lon0, lat0 = tr_inv.transform(b.left,  b.bottom)
        lon1, lat1 = tr_inv.transform(b.right, b.top)
        city_boxes.append((lat0, lat1, lon0, lon1))

    # ── Parse JSON files ──────────────────────────────────────────────────────
    # Each file is a sequence of JSON objects, one per line.
    # Each object looks like:
    #   { "data": [
    #       { "vars": { "B04001":..., "B04002":..., ..., "B05001":..., "B06001":... } },
    #       { "vars": { "B11001": {"v": 205} } }   ← direction line
    #   ] }
    # or:
    #   { "data": [
    #       { "vars": { ... same meta ... } },
    #       { "vars": { "B11002": {"v": 3.2} } }   ← speed line
    #   ] }

    records = []
    for fpath in cfg['json_paths']:
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found — skipping")
            continue
        print(f"  Parsing {fpath} …")
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if 'data' not in obj or len(obj['data']) < 2:
                    continue

                meta_block = obj['data'][0].get('vars', {})
                meas_block = obj['data'][1].get('vars', {})

                lat = meta_block.get('B05001', {}).get('v')
                lon = meta_block.get('B06001', {}).get('v')
                if lat is None or lon is None:
                    continue

                year  = meta_block.get('B04001', {}).get('v')
                month = meta_block.get('B04002', {}).get('v')
                day   = meta_block.get('B04003', {}).get('v')
                hour  = meta_block.get('B04004', {}).get('v', 0)

                # One of these will be None depending on the line type
                wd = meas_block.get('B11001', {}).get('v')
                ws = meas_block.get('B11002', {}).get('v')

                records.append({
                    'lat': float(lat), 'lon': float(lon),
                    'year': year, 'month': month, 'day': day, 'hour': hour,
                    'WD': wd, 'WVv': ws,
                })

    if not records:
        print("  Warning: no MNW records parsed")
        return pd.DataFrame(columns=['Datum', 'Standort', 'WVv', 'WD']), {}

    df = pd.DataFrame(records)

    # ── Filter to June 2022 ───────────────────────────────────────────────────
    df = df[(df['year'] == 2022) & (df['month'] == 5)].copy()
    print(f"  After May 2022 filter: {len(df)} raw records")

    # ── Spatial filter: keep only stations within Milan raster bounds ─────────
    # ── Spatial filter: keep only stations within any filter_to_cities extent ─
    unique_pts = df[['lat', 'lon']].drop_duplicates().copy()
    unique_pts['matched_city'] = None

    for city_name, (lat0, lat1, lon0, lon1) in zip(
            cfg.get('filter_to_cities', []), city_boxes):
        mask = (
            (unique_pts['lat'] > lat0) & (unique_pts['lat'] < lat1) &
            (unique_pts['lon'] > lon0) & (unique_pts['lon'] < lon1)
        )
        unique_pts.loc[mask, 'matched_city'] = city_name

    valid_pts = unique_pts.dropna(subset=['matched_city']).reset_index(drop=True)
    print(f"  {len(valid_pts)} stations within bounds of "
          f"{cfg.get('filter_to_cities', [])} "
          f"(of {len(unique_pts)} unique)")
    print(valid_pts.groupby('matched_city').size().to_string())

    df = df.merge(valid_pts, on=['lat', 'lon'], how='inner')

    # ── Build timestamps ──────────────────────────────────────────────────────
    df['Datum'] = pd.to_datetime(
        dict(year=df['year'], month=df['month'], day=df['day'], hour=df['hour']),
        utc=True,
    )

    # ── Station ID from rounded lat/lon ───────────────────────────────────────
    df['Standort'] = (
        'MNW_' + df['lat'].round(4).astype(str)
        + '_'  + df['lon'].round(4).astype(str)
    )

    # ── Merge direction and speed lines ──────────────────────────────────────
    # Direction and speed are on separate lines → group by (Datum, Standort)
    # and take the first non-null value for each variable.
    df = (df.groupby(['Datum', 'Standort', 'lat', 'lon', 'matched_city'], as_index=False)
            .agg({'WD': 'first', 'WVv': 'first'}))

    df = df.dropna(subset=['WVv']).reset_index(drop=True)

    # ── Build stations_meta ───────────────────────────────────────────────────
    stations_meta = {}
    for _, row in df[['Standort', 'lat', 'lon', 'matched_city']].drop_duplicates().iterrows():
            x3035, y3035 = transformer.transform(row['lon'], row['lat'])
            stations_meta[row['Standort']] = {
                'Koordinaten_WGS84_lat': row['lat'],
                'Koordinaten_WGS84_lng': row['lon'],
                '_x3035':       x3035,
                '_y3035':       y3035,
                '_height_ag':   cfg['station_height_ag'],
                '_matched_city': row['matched_city'],
            }
    print(f"  MNW final: {len(df)} rows, {df['Standort'].nunique()} stations")
    return df[['Datum', 'Standort', 'WVv', 'WD']], stations_meta


def load_observations_zurich(cfg):
    def load_meteoswiss(csv_path, station_code, start_date, end_date):
        """Load MeteoSwiss wind observations (speed + direction)"""
        df = pd.read_csv(csv_path, sep=";", decimal=".", na_values=["", " "])
        df['timestamp'] = pd.to_datetime(df['reference_timestamp'], format="%d.%m.%Y %H:%M")
        df = df[df['station_abbr'] == station_code]
        
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        mask = (df['timestamp'].dt.date >= start_dt) & (df['timestamp'].dt.date <= end_dt)
        
        df = df[mask][['timestamp', 'fkl010h0', 'dkl010h0']].copy()
        df = df.rename(columns={'fkl010h0': 'observed_speed', 'dkl010h0': 'observed_direction'})
        df = df.dropna()
        
        print(f"  Loaded {len(df)} observations (speed + direction)")
        return df

    def load_meteoswiss_tower(csv_path, station_code, start_date, end_date):
        """Load MeteoSwiss tower wind observations (speed + direction from tower height)"""
        df = pd.read_csv(csv_path, sep=";", decimal=".", na_values=["", " "])
        df['timestamp'] = pd.to_datetime(df['reference_timestamp'], format="%d.%m.%Y %H:%M")
        df = df[df['station_abbr'] == station_code]
        
        start_dt = pd.to_datetime(start_date).date()
        end_dt = pd.to_datetime(end_date).date()
        mask = (df['timestamp'].dt.date >= start_dt) & (df['timestamp'].dt.date <= end_dt)
        
        df = df[mask][['timestamp', 'fk1towh0', 'dk1towh0']].copy()
        df = df.rename(columns={'fk1towh0': 'observed_speed', 'dk1towh0': 'observed_direction'})
        df = df.dropna()
        
        print(f"  Loaded {len(df)} tower observations (speed + direction)")
        return df

    METEOSWISS_STATIONS = {
        'SMA': {
            'paths': ['../winds/obs/zurich/ogd-smn_sma_h_historical_2010-2019.csv',
                      '../winds/obs/zurich/ogd-smn_sma_h_historical_2020-2029.csv'],
            'type': 'meteoswiss',
            'lat': 47.37689, 'lon': 8.56835,
            'x': 4212616.767770121, 'y': 2697085.317715836,
            'height_ag': 10.0,
        },
        'REH': {
            'paths': ['../winds/obs/zurich/ogd-smn_reh_h_historical_2010-2019.csv',
                      '../winds/obs/zurich/ogd-smn_reh_h_historical_2020-2029.csv'],
            'type': 'meteoswiss',
            'lat': 47.42529, 'lon': 8.49972,
            'x': 4209113.276438552, 'y': 2702684.8177752094,
            'height_ag': 10.0,
        },
        'UEB': {
            'paths': ['../winds/obs/zurich/ogd-smn-tower_ueb_h_historical_2010-2019.csv',
                      '../winds/obs/zurich/ogd-smn-tower_ueb_h_historical_2020-2029.csv'],
            'type': 'meteoswiss_tower',
            'lat': 47.35312, 'lon': 8.49917,
            'x': 4206852.71, 'y': 2694248.38,
            'height_ag': 40.0,
        },
    }

    df_ugz_raw = pd.read_csv('../winds/obs/zurich/ugz_ogd_meteo_h1_2018.csv')
    df_ugz = (df_ugz_raw[df_ugz_raw['Parameter'].isin(['WVv', 'WD'])]
              [['Datum', 'Standort', 'Parameter', 'Wert']]
              .assign(Wert=lambda d: pd.to_numeric(d['Wert'], errors='coerce'))
              .pivot_table(index=['Datum', 'Standort'],
                           columns='Parameter', values='Wert')
              .reset_index()
              .dropna(subset=['WVv']))
    df_ugz.columns.name = None
    df_ugz['Datum'] = pd.to_datetime(df_ugz['Datum'], utc=True)

    ms_frames = []
    for sid, scfg in METEOSWISS_STATIONS.items():
        if sid in cfg['exclude_stations']:
            continue
        dfs = []
        for p in scfg['paths']:
            if not os.path.exists(p):
                continue
            loader = load_meteoswiss if scfg['type'] == 'meteoswiss' else load_meteoswiss_tower
            d = loader(p, sid,
                       JUNE_START.strftime('%Y-%m-%d'),
                       JUNE_END.strftime('%Y-%m-%d'))
            if len(d):
                dfs.append(d)
        if not dfs:
            continue
        df_ms = (pd.concat(dfs)
                   .drop_duplicates('timestamp').sort_values('timestamp')
                   .rename(columns={'timestamp': 'Datum',
                                    'observed_speed': 'WVv',
                                    'observed_direction': 'WD'}))
        df_ms['Datum']    = pd.to_datetime(df_ms['Datum'], utc=True)
        df_ms['Standort'] = sid
        ms_frames.append(df_ms[['Datum', 'Standort', 'WVv', 'WD']])

    df_wide = pd.concat([df_ugz, *ms_frames], ignore_index=True)

    with open('../winds/obs/zurich/uzg_ogd_metadaten.json') as f:
        meta_json = json.load(f)
    stations_meta = {s['ID']: s for s in meta_json['Standorte']}
    for sid, scfg in METEOSWISS_STATIONS.items():
        stations_meta[sid] = {
            'Koordinaten_WGS84_lat': scfg['lat'],
            'Koordinaten_WGS84_lng': scfg['lon'],
            '_x3035': scfg['x'], '_y3035': scfg['y'],
            '_height_ag': scfg['height_ag'],
        }

    return df_wide, stations_meta


def load_observations_milan(cfg):
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)

    MILAN_STATIONS = {
        'ZAVATTARI': {'speed_id': 19005, 'dir_id': None,  'lat': 45.47606341, 'lon': 9.141786267},
        'BRERA':     {'speed_id': 19008, 'dir_id': None,  'lat': 45.47165629, 'lon': 9.189110384},
        'MARCHE':    {'speed_id': 19020, 'dir_id': 19021, 'lat': 45.49631644, 'lon': 9.190933555},
        'LAMBRATE':  {'speed_id': 14391, 'dir_id': 14390, 'lat': 45.49677982, 'lon': 9.2575154 },
        'JUVARA':    {'speed_id': 19243, 'dir_id': 19244, 'lat': 45.47322573, 'lon': 9.222302345},
    }

    start_dt = pd.to_datetime('2018-01-01')
    end_dt   = pd.to_datetime('2018-12-31') + pd.Timedelta(days=1)

    df_speed_raw = pd.read_csv('../winds/obs/milan/velocita_vento_2018.csv',
                                sep=';', decimal='.', encoding='latin1')
    df_dir_raw   = pd.read_csv('../winds/obs/milan/direzione_vento_2018.csv',
                                sep=';', decimal='.', encoding='latin1')

    obs_frames = []
    for code, scfg in MILAN_STATIONS.items():
        if code in cfg['exclude_stations']:
            continue
        df_s = (df_speed_raw[df_speed_raw['IdSensore'] == scfg['speed_id']]
                .query("idOperatore == 'Valore medio'").copy())
        df_s['ts']    = pd.to_datetime(df_s['Data'], format='%d/%m/%Y %H:%M:%S')
        df_s = df_s[df_s['ts'].dt.minute == 0].copy()
        df_s = df_s[(df_s['ts'] >= start_dt) & (df_s['ts'] < end_dt)].copy()
        df_s['Datum'] = df_s['ts'] - pd.Timedelta(hours=1)
        df_s['WVv']   = pd.to_numeric(df_s['Valore'], errors='coerce')
        df_s = df_s[['Datum', 'WVv']].dropna()

        if scfg['dir_id'] is not None:
            df_d = df_dir_raw[df_dir_raw['IdSensore'] == scfg['dir_id']].copy()
            df_d['ts']        = pd.to_datetime(df_d['Data'], format='%d/%m/%Y %H:%M:%S')
            df_d['ts_utc']    = df_d['ts'] - pd.Timedelta(hours=1)
            df_d['direction'] = pd.to_numeric(df_d['Valore'], errors='coerce')
            df_d = df_d.dropna(subset=['direction']).sort_values('ts_utc')
            df_d['target_hour'] = df_d['ts_utc'].dt.ceil('h')
            hourly_dir = [
                {'Datum': tgt, 'WD': grp.loc[
                    (grp['ts_utc'] > tgt - pd.Timedelta(hours=1)) &
                    (grp['ts_utc'] <= tgt), 'direction'].mean()}
                for tgt, grp in df_d.groupby('target_hour')
                if len(grp)
            ]
            df_obs = df_s.merge(pd.DataFrame(hourly_dir), on='Datum', how='left')
        else:
            df_obs       = df_s.copy()
            df_obs['WD'] = np.nan

        df_obs['Standort'] = code
        obs_frames.append(df_obs)

    df_wide = pd.concat(obs_frames, ignore_index=True)
    df_wide['Datum'] = pd.to_datetime(df_wide['Datum'], utc=True)

    stations_meta = {}
    for code, scfg in MILAN_STATIONS.items():
        x3035, y3035 = transformer.transform(scfg['lon'], scfg['lat'])
        stations_meta[code] = {
            'Koordinaten_WGS84_lat': scfg['lat'],
            'Koordinaten_WGS84_lng': scfg['lon'],
            '_x3035': x3035, '_y3035': y3035,
            '_height_ag': cfg['station_height_ag'],
        }

    return df_wide, stations_meta


LOADERS = {
    'zurich': load_observations_zurich,
    'milan':  load_observations_milan,
    'mnw':    load_observations_mnw,
}


# =============================================================================
# STEP 1 — Load obs + sample rasters for all cities
# =============================================================================
print("\n[1] Loading observations and sampling rasters …")

all_obs_frames    = []
all_station_feats = {}
transformer       = Transformer.from_crs('EPSG:4326', 'EPSG:3035', always_xy=True)

# We need geopotential for elev_diff for all cities. ERA5 geopotential is
# time-invariant so any year's file works — open one 2018 file for all.
ds_era5_static = xr.open_dataset(sorted(glob.glob(ERA5_GLOB))[0])

for city in CITIES:
    if city not in CITY_CONFIGS:
        raise ValueError(f"No config for '{city}'. Add it to CITY_CONFIGS.")
    if city not in LOADERS:
        raise ValueError(f"No loader for '{city}'. Add load_observations_{city}().")

    cfg             = CITY_CONFIGS[city]
    city_june_start = cfg.get('june_start', JUNE_START)
    city_june_end   = cfg.get('june_end',   JUNE_END)
    city_era5_glob  = cfg.get('era5_glob',  ERA5_GLOB)

    print(f"\n  ── {city.upper()} ──")

    df_wide, stations_meta = LOADERS[city](cfg)
    df_wide['city'] = city
    df_wide['Datum'] = pd.to_datetime(df_wide['Datum'], utc=True)
    df_wide = (df_wide
               .query('@city_june_start <= Datum <= @city_june_end')
               .pipe(lambda d: d[~d['Standort'].isin(cfg['exclude_stations'])])
               .reset_index(drop=True))

    print(f"  {len(df_wide)} rows, {df_wide['Standort'].nunique()} stations")

    # Raster sampling
    # Inherit raster paths from filter_to_cities if not explicitly set
    if 'lambda_p_tif' not in cfg:
        parent = cfg['filter_to_cities'][0]
        for key in ('dtm_path', 'lambda_p_tif', 'mean_height_tif',
                    'z0_tif', 'zd_tif'):
            if key in CITY_CONFIGS[parent]:
                cfg[key] = CITY_CONFIGS[parent][key]

    lp_info  = get_raster(cfg['lambda_p_tif'])
    mh_info  = get_raster(cfg['mean_height_tif'])
    dtm_info = get_raster(cfg['dtm_path'])

    city_feats = {}
    for sid, smeta in stations_meta.items():
        if sid not in df_wide['Standort'].values:
            continue
        if '_x3035' in smeta:
            x, y     = smeta['_x3035'], smeta['_y3035']
            lat, lon = smeta['Koordinaten_WGS84_lat'], smeta['Koordinaten_WGS84_lng']
            h_ag     = smeta.get('_height_ag', cfg['station_height_ag'])
        else:
            x, y     = transformer.transform(smeta['Koordinaten_WGS84_lng'],
                                              smeta['Koordinaten_WGS84_lat'])
            lat, lon = smeta['Koordinaten_WGS84_lat'], smeta['Koordinaten_WGS84_lng']
            h_ag     = cfg['station_height_ag']
        h_ag = cfg.get('height_ag_map', {}).get(sid, h_ag)


        local_elev = sample_nearest(dtm_info, x, y)
        era5_elev  = float(
            ds_era5_static['z']
            .sel(latitude=lat, longitude=lon, method='nearest')
            .isel(valid_time=0).values.item()
        ) / 9.80665

        # For cities with filter_to_cities, use the matched city's rasters directly
        matched_city = smeta.get('_matched_city')
        if matched_city:
            fc_cfg   = CITY_CONFIGS[matched_city]
            lp_info  = get_raster(fc_cfg['lambda_p_tif'])
            mh_info  = get_raster(fc_cfg['mean_height_tif'])
            dtm_info = get_raster(fc_cfg['dtm_path'])

        # sample z0 and zd if rasters are available for this city
        fc_cfg_for_z = CITY_CONFIGS[matched_city] if matched_city else cfg
        z0_val = np.nan
        zd_val = np.nan
        if 'z0_tif' in fc_cfg_for_z:
            z0_val = sample_nearest(get_raster(fc_cfg_for_z['z0_tif']), x, y)
        if 'zd_tif' in fc_cfg_for_z:
            zd_val = sample_nearest(get_raster(fc_cfg_for_z['zd_tif']), x, y)

        city_feats[sid] = {
            'lambda_p':    sample_nearest(lp_info,  x, y),
            'mean_height': sample_nearest(mh_info,  x, y),
            'elev_diff':   sample_nearest(dtm_info, x, y) - era5_elev,
            'height_ag':   h_ag,
            'z0':          z0_val,
            'zd':          zd_val,
            '_lat': lat, '_lon': lon,
            '_era5_glob':   city_era5_glob,
            '_june_start':  city_june_start,
            '_june_end':    city_june_end,
        }
        print(f"    {sid}: { {k: round(v,3) for k,v in city_feats[sid].items() if not k.startswith('_')} }")

    feats_df = pd.DataFrame(city_feats).T.rename_axis('Standort').reset_index()
    df_wide  = df_wide.merge(feats_df, on='Standort', how='left')

    if city == 'mnw':
        sid_to_geo = {sid: smeta['_matched_city']
                      for sid, smeta in stations_meta.items()
                      if '_matched_city' in smeta}
        df_wide['geo_city'] = df_wide['Standort'].map(sid_to_geo)
    else:
        df_wide['geo_city'] = city

    df_wide['noise_var'] = (
        df_wide['Standort']
        .map(cfg['noise_map'])
        .fillna(cfg['noise_default'])
        .astype(float) ** 2
    )

    all_obs_frames.append(df_wide)
    all_station_feats.update(city_feats)

ds_era5_static.close()

df_all_obs = pd.concat(all_obs_frames, ignore_index=True)
print(f"\n  Total obs (pre-ERA5): {len(df_all_obs)} rows, "
      f"{df_all_obs['Standort'].nunique()} stations across {CITIES}")

def wd_to_components(wd_deg):
    
    theta =  np.asarray(wd_deg.astype(np.float32)) % 180.0
    # scale by 45 
    seg = theta / 45.0
    # i0 and i1: the quadrants
    i0 = np.floor(seg).astype(int) % 4
    i1 = (i0 + 1) % 4
    # w0 and w1: their respective weights
    w1 = (seg - np.floor(seg)).astype(float32) 
    w0 = 1.0 - w1
    # match the quadrants to the bands we created
    # quadrant order is 0: horizontal, 1: ne-sw, 2: vertical, 3: se-nw
    # band order is: 0: vertical, 1: ne-sw, 2: horizontal, 3: se-nw
    angle_to_band = np.array([2, 1, 0, 3])
    band0 = angle_to_band[i0]
    band1 = angle_to_band[i1]
    out = np.zeros((theta.size, 4), dtype = np.float32)
    np.add.at(out, (np.arange(theta.size), band0), w0)
    np.add.at(out, (np.arange(theta.size), band1), w1)

    return out

# =============================================================================
# STEP 2 — ERA5 dynamic features: one pass per ERA5 archive (year)
#
# Cities may use different ERA5 year archives (2018 vs 2022). We group
# stations by their ERA5 glob, then do one xarray pass per group so we
# never open a 2018 file looking for 2022 timestamps (or vice versa).
# =============================================================================
print("\n[2] Loading ERA5 dynamic features …")

all_coords = {sid: (f['_lat'], f['_lon']) for sid, f in all_station_feats.items()}
all_sids   = set(df_all_obs['Standort'].unique()) & set(all_coords)

# Group station IDs by which ERA5 archive they need
sids_by_era5 = defaultdict(list)
for sid in all_sids:
    sids_by_era5[all_station_feats[sid]['_era5_glob']].append(sid)

era5_rows = []
for era5_glob, sids_group in sids_by_era5.items():
    # All stations in this group share the same date window
    june_start = all_station_feats[sids_group[0]]['_june_start']
    june_end   = all_station_feats[sids_group[0]]['_june_end']
    files      = sorted(glob.glob(era5_glob))
    print(f"  ERA5 glob: {era5_glob}  →  {len(files)} files, "
          f"{len(sids_group)} stations")

    for fpath in files:
        ds = xr.open_dataset(fpath)
        if 'zust' not in ds or 'u10' not in ds:
            ds.close()
            continue
        for sid in sids_group:
            lat, lon = all_coords[sid]
            ds_pt = ds.sel(latitude=lat, longitude=lon, method='nearest')
            era5_rows.append(pd.DataFrame({
                'Datum':    pd.to_datetime(ds_pt['valid_time'].values, utc=True),
                'u10':      ds_pt['u10'].values,
                'v10':      ds_pt['v10'].values,
                'zust':     ds_pt['zust'].values,
                'Standort': sid,
            }))
        ds.close()

era5_all = pd.concat(era5_rows, ignore_index=True)

# Each station's rows are filtered to its own date window via the merge —
# the left join on (Datum, Standort) naturally keeps only matching timestamps.
era5_all['era5_wd']    = (180 + np.degrees(
                           np.arctan2(era5_all['u10'], era5_all['v10']))) % 360

wd_comps = wd_to_components(era5_all['era5_wd'].values)
era5_all[['wd_vert', 'wd_diag_ne', 'wd_horiz', 'wd_diag_se']] = wd_comps
era5_all = era5_all.drop(columns=['u10', 'v10', 'era5_wd'])
print(f"  ERA5 rows total: {len(era5_all)}")

df_all_obs = (df_all_obs
              .drop(columns=['_lat', '_lon',
                              '_era5_glob', '_june_start', '_june_end'],
                    errors='ignore')
              .merge(era5_all, on=['Datum', 'Standort'], how='left')
              .dropna()
              .reset_index(drop=True))
city_to_source = {
    'zurich': 'station',
    'milan':  'station',
    'mnw':    'street_network',
}
df_all_obs['source'] = df_all_obs['city'].map(city_to_source).fillna('station')

print(f"  After merge + dropna: {len(df_all_obs)} rows")
print(df_all_obs.groupby(['city', 'Standort'])[TARGET_COL].count().to_string())


# =============================================================================
# STEP 3 — Urban Tales snapshots
# =============================================================================
print("\n[3] Loading Urban Tales snapshots …")

ut_frames = []
for i, snap in enumerate(UT_SNAPSHOTS):
    ds_uped  = xr.open_dataset(snap['uped_nc'])
    uped_30m = ds_uped['Uped_30m'].values
    ds_uped.close()

    with rasterio.open(snap['density_tiff']) as src:
        lambda_p_ut = src.read(1).astype(np.float32)
        rows, cols  = np.indices(lambda_p_ut.shape)
        xs, ys      = rasterio.transform.xy(src.transform, rows.ravel(), cols.ravel())
    with rasterio.open(snap['height_tiff']) as src:
        height_ut = src.read(1).astype(np.float32)
    with rasterio.open(snap['z0_tif']) as src:
        z0_ut = src.read(1).astype(np.float32)
    with rasterio.open(snap['zd_tif']) as src:
        zd_ut = src.read(1).astype(np.float32)

    n_cells = uped_30m.size
    df_snap = pd.DataFrame({
        'Datum':       snap['timestamp'],
        'Standort':    [f'UT{i}_{j}' for j in range(n_cells)],
        'city':        'zurich',
        TARGET_COL:    uped_30m.ravel(),
        'lambda_p':    lambda_p_ut.ravel(),
        'mean_height': height_ut.ravel(),
        'z0':          z0_ut.ravel(),
        'zd':          zd_ut.ravel(),
        'zust':        snap['zust'],
        **dict(zip(
            ['wd_vert', 'wd_diag_ne', 'wd_horiz', 'wd_diag_se'],
            wd_to_components([snap['wd']])[0]
        )),
        'elev_diff':   0.0,
        'height_ag':   snap['height_ag'],
        'source':      'urban_tales',
        'noise_var':   float(NOISE_UT ** 2),
    }).dropna()

    print(f"  Snapshot {i}: {len(df_snap)} cells  "
          f"zust={snap['zust']:.4f}  wd={snap['wd']}°  "
          f"WVv=[{df_snap[TARGET_COL].min():.3f}, {df_snap[TARGET_COL].max():.3f}]")
    ut_frames.append(df_snap)

df_sim = pd.concat(ut_frames, ignore_index=True)


# =============================================================================
# STEP 4 — Assemble and save
# =============================================================================
print("\n[4] Assembling outputs …")

keep = ['Datum', 'Standort', 'city', 'geo_city', 'source', TARGET_COL] \
     + FEAT_COLS + ['z0', 'zd', 'noise_var']

df_obs     = df_all_obs[[c for c in keep if c in df_all_obs.columns]].copy()
df_obs_sub = df_obs[df_obs['Datum'].dt.hour.isin(TRAIN_HOURS)].copy()
df_sim_out = df_sim[[c for c in keep if c in df_sim.columns]].copy()
df_train   = pd.concat([df_obs_sub, df_sim_out], ignore_index=True)

print(f"  gpr_obs       : {len(df_obs)} rows  (all June hours)")
print(f"  gpr_train_gpr : {len(df_train)} rows")
print(f"    obs sub     : {len(df_obs_sub)}  (hours {TRAIN_HOURS})")
print(f"    sim cells   : {len(df_sim_out)}  ({len(UT_SNAPSHOTS)} snapshots)")
print(f"\n  Breakdown by source and city:")
print(df_train.groupby(['city', 'source'])[TARGET_COL]
      .agg(['count', 'mean', 'std']).round(3).to_string())

meta_out = {
    'feat_cols':     FEAT_COLS,
    'target_col':    TARGET_COL,
    'cities':        CITIES,
    'train_hours':   TRAIN_HOURS,
    'stations_meta': {k: {kk: vv for kk, vv in v.items()
                          if not kk.startswith('_')}
                      for k, v in all_station_feats.items()},
    'june_start':    JUNE_START,
    'june_end':      JUNE_END,
    'noise_ut':      NOISE_UT,
    'city_configs':  {c: {k: v for k, v in CITY_CONFIGS[c].items()
                          if k in ('noise_map', 'noise_default', 'exclude_stations')}
                      for c in CITIES},
}

pd.to_pickle({'df': df_obs,   'meta': meta_out}, OUT_OBS)
pd.to_pickle({'df': df_train, 'meta': meta_out}, OUT_TRAIN)
print(f"\n[5] Saved:")
print(f"  {OUT_OBS}   ({len(df_obs)} rows)")
print(f"  {OUT_TRAIN} ({len(df_train)} rows)")