import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import joblib
import warnings
import cftime

warnings.filterwarnings("ignore", category=FutureWarning)

# Config
BASE_DIR = "assets/data"
SHAPEFILE = "assets/shp/gadm_simplified.geojson"
CACHE_DIR = ".cache_dataproc"
memory = joblib.Memory(CACHE_DIR, verbose=0)

HIST_START = 1991
HIST_END = 2014
DECADES = [
    (2021,2050),
    (2031,2060),
    (2041,2070),
    (2051,2080),
    (2061,2090),
    (2071,2100),
]
SCENARIOS = ["ssp126","ssp245","ssp370","ssp585"]

VAR_NAME_MAP = {
    "pr": ["pr","PR","precip","precipitation","precipitation_amount"],
    "tas": ["tas","TAS","tasmean","t2m","air_temperature"],
    "tasmin": ["tasmin","TASMIN","tmin","tas_min"],
    "tasmax": ["tasmax","TASMAX","tmax","tas_max"],
    "tos": ["tos","sst","sea_surface_temperature"]
}

def get_indonesia_regencies():
    """Return list of NAME_2 from shapefile (or empty list)."""
    try:
        gdf = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
        return sorted(gdf["NAME_2"].unique().tolist())
    except Exception:
        return []

# -------------------
# Helpers
# -------------------
def _normalize_coords_names(obj):
    """Rename dims/coords that look like lat/lon/time to standard 'lat','lon','TIME'."""
    rename_map = {}
    names_to_check = list(obj.coords) + list(getattr(obj, "dims", []))
    for name in names_to_check:
        ln = name.lower()
        if "lon" in ln or name in ("x", "longitude", "long"):
            rename_map[name] = "lon"
        elif "lat" in ln or name in ("y","latitude"):
            rename_map[name] = "lat"
        elif "time" in ln or name == "t":
            rename_map[name] = "TIME"
    if rename_map:
        try:
            obj = obj.rename(rename_map)
        except Exception:
            pass
    return obj

def _find_varname(ds, requested_var):
    """Find actual variable name in dataset matching requested_var using VAR_NAME_MAP."""
    if requested_var in ds.data_vars:
        return requested_var
    lower_map = {v.lower(): v for v in ds.data_vars}
    for candidate in VAR_NAME_MAP.get(requested_var.lower(), [requested_var]):
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    # fallback: first data var
    return list(ds.data_vars)[0] if ds.data_vars else None

def _time_to_datetime(obj):
    """Ensure TIME coordinate is pandas datetime64 if possible."""
    if "TIME" in obj.coords:
        vals = obj["TIME"].values
        if len(vals) == 0:
            return obj
        if np.issubdtype(vals.dtype, np.datetime64):
            return obj
        if isinstance(vals[0], (cftime.DatetimeNoLeap, cftime.DatetimeGregorian, cftime.DatetimeProlepticGregorian)):
            try:
                obj = obj.assign_coords(TIME=pd.to_datetime([t.strftime("%Y-%m-%d") for t in vals]))
            except Exception:
                obj["TIME"] = pd.to_datetime(obj["TIME"].values.astype("O"))
            return obj
        try:
            obj = obj.assign_coords(TIME=pd.to_datetime(vals))
        except Exception:
            pass
    return obj

def _da_to_clean_df(da, var):
    """
    Convert a xarray DataArray (spatial-mean) into a clean pandas DataFrame:
    - try to cast values to float
    - convert to dataframe
    - normalize column name case-insensitively
    - coerce column to numeric and drop NaNs
    """
    try:
        # attempt to convert data to float (if it's object dtype)
        da = da.astype('float64')
    except Exception:
        pass

    try:
        df = da.to_dataframe(name=var).reset_index()
    except Exception:
        # fallback: compute first then to_dataframe
        df = da.compute().to_dataframe(name=var).reset_index()
    df['year'] = pd.to_datetime(df['TIME']).dt.year.astype(int)
    df['month'] = pd.to_datetime(df['TIME']).dt.month.astype(int)

    # normalize column name (case-insensitive)
    if var not in df.columns:
        for c in df.columns:
            if c.lower() == var.lower():
                df = df.rename(columns={c: var})
                break

    # coerce to numeric and drop non-numeric rows
    if var in df.columns:
        df[var] = pd.to_numeric(df[var], errors='coerce')
        df = df.dropna(subset=[var])

    return df

def _list_nc_files_for(scenario, var_name):
    folder = os.path.join(BASE_DIR, scenario)
    if not os.path.isdir(folder):
        return []
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".nc")]
    # prefer files that contain var name in filename
    matching = [p for p in files if var_name.lower() in os.path.basename(p).lower()]
    return sorted(matching) if matching else sorted(files)

def _get_aoi_bounds(aoi):
    """Return (lon_min, lat_min, lon_max, lat_max) for AOI name using shapefile, or None."""
    try:
        gdf = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
        sel = gdf.loc[gdf["NAME_2"] == aoi]
        if sel.empty:
            return None
        lon_min, lat_min, lon_max, lat_max = sel.total_bounds
        # add small pad
        pad_lon = (lon_max - lon_min) * 0.02
        pad_lat = (lat_max - lat_min) * 0.02
        return (lon_min - pad_lon, lat_min - pad_lat, lon_max + pad_lon, lat_max + pad_lat)
    except Exception:
        return None

def _open_ds_lazy(fp):
    """Open dataset lazily with chunks (dask) and normalize coords/time."""
    try:
        ds = xr.open_dataset(fp, decode_times=True, use_cftime=True, chunks={})
    except Exception:
        ds = xr.open_dataset(fp, decode_times=True, use_cftime=True)
    ds = _normalize_coords_names(ds)
    ds = _time_to_datetime(ds)
    return ds

# -------------------
# Backwards-compatible API functions (same names as original app expects)
# -------------------

@memory.cache
def process_and_clip_all_scenarios(aoi_name, var_name):
    """
    Build descriptor that maps scenario -> files (same as before).
    Keep this function lightweight (it only lists files), heavy lifting in analyze_data.
    """
    scenarios = ["historis"] + SCENARIOS
    descriptor = {"var_name": var_name, "aoi": aoi_name, "files": {}}
    for sc in scenarios:
        f_list = _list_nc_files_for(sc, var_name)
        filemap = {}
        for fp in f_list:
            key = os.path.splitext(os.path.basename(fp))[0]
            filemap[key] = fp
        descriptor["files"][sc] = filemap
    return descriptor

def _process_single_file_lazy(fp, var, aoi_bounds):
    """
    Open file lazily, normalize coords & time, find var, clip to AOI bounds (lon/lat) if present.
    Return DataArray (lazy) named by requested var.
    """
    try:
        ds = _open_ds_lazy(fp)
        var_actual = _find_varname(ds, var)
        if var_actual is None or var_actual not in ds.data_vars:
            return None
        da = ds[var_actual]
        da = _normalize_coords_names(da)
        da = _time_to_datetime(da)
        # ensure lon in -180..180 if present
        if "lon" in da.coords:
            try:
                da = da.assign_coords(lon=((da["lon"] + 180) % 360) - 180)
                da = da.sortby("lon")
            except Exception:
                pass
        # Clip spatially to AOI bounds (lazy selection)
        if aoi_bounds and ("lon" in da.coords) and ("lat" in da.coords):
            lon_min, lat_min, lon_max, lat_max = aoi_bounds
            try:
                da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
            except Exception:
                # try sorting lon
                try:
                    da = da.sortby("lon")
                    da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
                except Exception:
                    pass
        # rename to standard var key
        da = da.rename(var)
        return da
    except Exception:
        return None

def _ensemble_mean_from_list(da_list):
    """
    Safely compute ensemble mean from list of DataArray objects (lazy).
    Try xr.concat(...) first. If concat fails or dtype object issues, fallback to incremental mean.
    Returns DataArray (lazy or computed).
    """
    if not da_list:
        return None
    try:
        # try safe concat (lazy)
        stacked = xr.concat(da_list, dim="model", coords='minimal', compat='override')
        return stacked.mean(dim="model", skipna=True)
    except Exception:
        # fallback incremental mean using dask compute per file to avoid OOM
        total = None
        count = 0
        for da in da_list:
            try:
                # compute per-file mean over model dim (if any) or keep as is
                da_comp = da
                # convert to dask array if not already
                if not isinstance(da_comp.data, (np.ndarray,)):
                    # leave as lazy (dask) but we sum incrementally by calling .load() in small chunks
                    pass
                # accumulate lazily via xarray where possible:
                if total is None:
                    total = da_comp.copy(deep=False)
                else:
                    total = total + da_comp
                count += 1
            except Exception:
                continue
        if total is None:
            return None
        if count == 0:
            return None
        return total / count

@memory.cache
def analyze_data(descriptor):
    """
    Main analysis function (optimized).
    Returns same structure as original analyze_data but computed more memory-safely.
    """
    var = descriptor.get("var_name")
    aoi = descriptor.get("aoi")
    files = descriptor.get("files", {})

    clipped_data = {}

    # compute AOI bounds once
    aoi_bounds = _get_aoi_bounds(aoi)

    # For each scenario: open files lazily, clip spatially early, and compute ensemble mean (lazy)
    for sc in ["historis"] + SCENARIOS:
        filemap = files.get(sc, {})
        da_list = []
        for _, fp in filemap.items():
            da = _process_single_file_lazy(fp, var, aoi_bounds)
            if da is not None:
                da_list.append(da)
        if da_list:
            # ensemble mean across model/files (lazy)
            try:
                ens = _ensemble_mean_from_list(da_list)
                # keep as dataset-like by converting to Dataset (var as key) for backward compatibility
                if ens is not None:
                    ds_out = ens.to_dataset(name=var)
                    clipped_data[sc] = ds_out
                else:
                    clipped_data[sc] = None
            except Exception:
                clipped_data[sc] = None
        else:
            clipped_data[sc] = None

    # --- Time-series and climatology computations ---
    # For historical files: compute monthly climatology by spatial-mean then monthly average across files
    hist_monthly_ensembles = []
    hist_dfs = []
    for _, fp in files.get("historis", {}).items():
        da = _process_single_file_lazy(fp, var, aoi_bounds)
        if da is None:
            continue
        # spatial mean (lazy) -> results in DataArray over TIME
        if ("lat" in da.dims) and ("lon" in da.dims):
            da_spmean = da.mean(dim=["lat", "lon"], skipna=True)
        else:
            # mean over any non-TIME dims
            other_dims = [d for d in da.dims if d != "TIME"]
            if other_dims:
                da_spmean = da.mean(dim=other_dims, skipna=True)
            else:
                da_spmean = da
        # ensure TIME normalized
        da_spmean = _time_to_datetime(da_spmean)
        df = _da_to_clean_df(da_spmean, var)
        if df.empty:
            continue
        df['TIME'] = pd.to_datetime(df['TIME'])
        df['year'] = pd.to_datetime(df['TIME']).dt.year.astype(int)
        df['month'] = pd.to_datetime(df['TIME']).dt.month.astype(int)
        # monthly mean per file
        monthly = df.groupby('month')[var].mean(numeric_only=True)
        hist_monthly_ensembles.append(monthly)
        hist_dfs.append(df)

    if hist_monthly_ensembles:
        df_hist_ensemble = pd.concat(hist_monthly_ensembles, axis=1)
        mclim_historis = pd.DataFrame({
            "month": range(1,13),
            "mean": df_hist_ensemble.mean(axis=1).values
        })
    else:
        mclim_historis = pd.DataFrame()

    # Projections: for each decade & scenario compute monthly means (ensemble across files)
    mclim_proyeksi_decade = {}
    proj_dfs = {sc: [] for sc in SCENARIOS}
    for start, end in DECADES:
        decade_key = f"{start}-{end}"
        mclim_proyeksi_decade[decade_key] = {}
        for sc in SCENARIOS:
            filemap = files.get(sc, {})
            sc_monthly_list = []
            for _, fp in filemap.items():
                da = _process_single_file_lazy(fp, var, aoi_bounds)
                if da is None:
                    continue
                # spatial mean
                if ("lat" in da.dims) and ("lon" in da.dims):
                    da_spmean = da.mean(dim=["lat","lon"], skipna=True)
                else:
                    other_dims = [d for d in da.dims if d != "TIME"]
                    if other_dims:
                        da_spmean = da.mean(dim=other_dims, skipna=True)
                    else:
                        da_spmean = da
                da_spmean = _time_to_datetime(da_spmean)
                df = _da_to_clean_df(da_spmean, var)
                if df.empty:
                    continue
                df['TIME'] = pd.to_datetime(df['TIME'])
                df['year'] = pd.to_datetime(df['TIME']).dt.year.astype(int)
                df['month'] = pd.to_datetime(df['TIME']).dt.month.astype(int)
                proj_dfs[sc].append(df)
                df_dec = df[(df['year'] >= start) & (df['year'] <= end)]
                if not df_dec.empty:
                    monthly = df_dec.groupby('month')[var].mean(numeric_only=True)
                    sc_monthly_list.append(monthly)
            if sc_monthly_list:
                df_sc_ensemble = pd.concat(sc_monthly_list, axis=1)
                mclim_proyeksi_decade[decade_key][sc] = pd.DataFrame({
                    "month": range(1,13),
                    "mean": df_sc_ensemble.mean(axis=1).values,
                    "min": df_sc_ensemble.min(axis=1).values,
                    "max": df_sc_ensemble.max(axis=1).values
                })
            else:
                mclim_proyeksi_decade[decade_key][sc] = pd.DataFrame()

    # Percent change
    percent_change_decade = {}
    for start, end in DECADES:
        decade_key = f"{start}-{end}"
        percent_change_decade[decade_key] = {}
        for sc in SCENARIOS:
            df_proj = mclim_proyeksi_decade[decade_key].get(sc, pd.DataFrame())
            if df_proj.empty or mclim_historis.empty:
                percent_change_decade[decade_key][sc] = pd.DataFrame()
                continue
            hist_means = mclim_historis.set_index("month")["mean"]
            proj_means = df_proj.set_index("month")["mean"]
            with np.errstate(divide='ignore', invalid='ignore'):
                pct_change = ((proj_means - hist_means) / hist_means.replace(0, np.nan)) * 100
            percent_change_decade[decade_key][sc] = pd.DataFrame({
                "month": range(1,13),
                "pct": pct_change.values
            })

    # Annual stats
    annual_stats_combined = {}
    if hist_dfs:
        df_hist = pd.concat(hist_dfs, ignore_index=True)
        df_hist['TIME'] = pd.to_datetime(df_hist['TIME'])
        df_hist['year'] = df_hist['TIME'].dt.year
        annual_grp = df_hist.groupby('year')[var]
        annual_stats_combined['historis'] = pd.DataFrame({
            'year': annual_grp.mean().index,
            'mean': annual_grp.mean().values,
            'min': annual_grp.min().values,
            'max': annual_grp.max().values
        })
    else:
        annual_stats_combined['historis'] = pd.DataFrame()

    for sc in SCENARIOS:
        if proj_dfs[sc]:
            df_proj = pd.concat(proj_dfs[sc], ignore_index=True)
            df_proj['TIME'] = pd.to_datetime(df_proj['TIME'])
            df_proj['year'] = df_proj['TIME'].dt.year
            annual_grp = df_proj.groupby('year')[var]
            annual_stats_combined[sc] = pd.DataFrame({
                'year': annual_grp.mean().index,
                'mean': annual_grp.mean().values,
                'min': annual_grp.min().values,
                'max': annual_grp.max().values
            })
        else:
            annual_stats_combined[sc] = pd.DataFrame()

    # Prepare clipped_arrays (DataArray per scenario) for spatial plotting compatibility (keep lazy)
    clipped_arrays = {}
    for sc, ds in clipped_data.items():
        if ds is None:
            clipped_arrays[sc] = None
            continue
        # ds is dataset from earlier ens.to_dataset
        if isinstance(ds, xr.Dataset):
            if var in ds.data_vars:
                clipped_arrays[sc] = ds[var]
            else:
                clipped_arrays[sc] = ds[list(ds.data_vars)[0]]
        else:
            clipped_arrays[sc] = None

    return {
        "mclim_historis": mclim_historis,
        "mclim_proyeksi_decade": mclim_proyeksi_decade,
        "percent_change_decade": percent_change_decade,
        "annual_stats_combined": annual_stats_combined,
        "clipped_data": clipped_data,
        "clipped_arrays": clipped_arrays
    }

# -------------------
# Extreme variables helpers (keamanan: lazy open + AOI clip)
# -------------------
extreme_variables_config = {
    "cdd": {"label": "CDD (Hari Kering Berturut-turut)", "unit": "hari", "nc_var": "CONSECUTIVE_DRY_DAYS_INDEX_PER_TIME_PERIOD"},
    "cwd": {"label": "CWD (Hari Basah Berturut-turut)", "unit": "hari", "nc_var": "CONSECUTIVE_WET_DAYS_INDEX_PER_TIME_PERIOD"},
    "rx1day": {"label": "Rx1day (Hujan Harian Maksimum)", "unit": "mm", "nc_var": "HIGHEST_ONE_DAY_PRECIPITATION_AMOUNT_PER_TIME_PERIOD"},
    "rx5day": {"label": "Rx5day (Hujan Maksimum 5 Hari)", "unit": "mm", "nc_var": "HIGHEST_FIVE_DAY_PRECIPITATION_AMOUNT_PER_TIME_PERIOD"},
    "r20mm": {"label": "R20mm (Hari Hujan > 20mm)", "unit": "hari", "nc_var": "VERY_HEAVY_PRECIPITATION_DAYS"}
}

def process_and_clip_all_scenarios_extreme(aoi, var_key):
    """
    Memproses data ekstrem tahunan untuk semua skenario (historis + proyeksi),
    lalu mengembalikan dict scenario -> DataArray (ensemble mean).
    """
    var_info = extreme_variables_config.get(var_key)
    if var_info is None:
        return {}
    nc_var = var_info["nc_var"]

    scenarios = ["historis"] + SCENARIOS
    results = {}
    aoi_bounds = _get_aoi_bounds(aoi)

    for scn in scenarios:
        folder = os.path.join(BASE_DIR, "ekstrem", scn)
        if not os.path.exists(folder):
            results[scn] = None
            continue

        # ✅ ambil semua file .nc tanpa syarat nama variabel
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".nc")]
        if not files:
            results[scn] = None
            continue

        da_list = []
        for f in files:
            try:
                ds = _open_ds_lazy(f)

                # cari variabel dalam file
                var_found = None
                for v in ds.data_vars:
                    if v.lower() == nc_var.lower() or var_key.lower() in v.lower():
                        var_found = v
                        break
                if var_found is None:
                    var_found = list(ds.data_vars)[0]  # fallback ambil variabel pertama

                da = ds[var_found]
                da = _normalize_coords_names(da)

                # clip ke AOI
                if aoi_bounds and ("lon" in da.coords) and ("lat" in da.coords):
                    lon_min, lat_min, lon_max, lat_max = aoi_bounds
                    try:
                        da = da.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
                    except Exception:
                        pass

                # rename agar konsisten
                da = da.rename(var_key)
                da_list.append(da)
            except Exception as e:
                print(f"Gagal membaca {f}: {e}")
                continue

        # hitung ensemble mean
        if da_list:
            try:
                ens = _ensemble_mean_from_list(da_list)
                results[scn] = ens
            except Exception:
                results[scn] = da_list[0]
        else:
            results[scn] = None

    return results

def analyze_extreme_data(clipped_extreme):
    """
    Accepts dict of scenario->DataArray (annual), and returns:
      - annual_stats_combined (DataFrame per scenario)
      - clipped_arrays (for spatial plots)
    """
    results = {
        "annual_stats_combined": {},
        "clipped_arrays": clipped_extreme
    }

    for scn, da in clipped_extreme.items():
        if da is None:
            results["annual_stats_combined"][scn] = pd.DataFrame()
            continue
        try:
            # spatial mean → timeseries tahunan
            if ("lat" in da.dims) and ("lon" in da.dims):
                ts = da.mean(dim=["lat", "lon"], skipna=True)
            else:
                other_dims = [d for d in da.dims if d != "TIME"]
                ts = da.mean(dim=other_dims, skipna=True) if other_dims else da
            ts = _time_to_datetime(ts)
            df = ts.to_dataframe(name="value").reset_index()
            if df.empty:
                continue
            df["year"] = df["TIME"].dt.year
            annual = df.groupby("year")["value"].agg(["mean", "min", "max"]).reset_index()
            results["annual_stats_combined"][scn] = annual
        except Exception:
            results["annual_stats_combined"][scn] = pd.DataFrame()

    return results

# =========================================================
# Tambahan fungsi utilitas untuk kompatibilitas app.py
# =========================================================

# Load shapefile sekali saja, bisa dipakai di app.py
try:
    GDF_BATAS = gpd.read_file(SHAPEFILE).to_crs(epsg=4326)
except Exception:
    GDF_BATAS = gpd.GeoDataFrame()

def compute_spatial_mean_timeseries(aoi, var, scenario):
    """Hitung rata-rata spasial (lon/lat) -> time series untuk AOI tertentu"""
    descriptor = process_and_clip_all_scenarios(aoi, var)
    processed = analyze_data(descriptor)
    da = processed["clipped_arrays"].get(scenario)
    if da is None:
        return pd.DataFrame()
    try:
        ts = da.mean(dim=["lat", "lon"], skipna=True).to_dataframe(name=var).reset_index()
    except Exception:
        other_dims = [d for d in da.dims if d != "TIME"]
        ts = da.mean(dim=other_dims, skipna=True).to_dataframe(name=var).reset_index()
    return ts

def compute_spatial_map_mean(aoi, var, scenario, period=None):
    """
    Hitung peta rata-rata spasial (lon/lat) untuk 1 periode atau keseluruhan.
    Kalau period=tuple(start,end) → filter TIME.
    """
    descriptor = process_and_clip_all_scenarios(aoi, var)
    processed = analyze_data(descriptor)
    da = processed["clipped_arrays"].get(scenario)
    if da is None:
        return None
    if "TIME" in da.dims and period is not None:
        start, end = period
        da = da.sel(TIME=slice(start, end))
    return da.mean(dim="TIME", skipna=True)

def seasonally_aggregate(df, how="mean"):
    """Agregasi DataFrame time series per bulan -> musim/klimatologi"""
    if "TIME" not in df:
        return df
    df["month"] = pd.to_datetime(df["TIME"]).dt.month
    grouped = df.groupby("month")["mean"]
    if how == "mean":
        return grouped.mean().reset_index()
    else:
        return grouped.sum().reset_index()

