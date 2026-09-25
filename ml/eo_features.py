"""Earth-observation feature recipe for GEO-MN exploration — single source of truth.

Both the offline grid build (ml/build_exploration_dataset.py) and the live
coordinate query (services/exploration_service.py) call extract_cell_features(),
so training and inference share identical column names, units, transforms,
quality masks and spatial aggregation.

Sources (all public, read anonymously from Microsoft Planetary Computer):
  * Sentinel-2 L2A (sentinel-2-l2a)            -> NDVI, iron-oxide ratio, clay/hydroxyl ratio
  * MODIS Terra LST 8-day (modis-11A2-061)     -> mean daytime land-surface temperature (K)
  * NASADEM (SRTM reprocessing, nasadem)       -> elevation (m), slope (degrees)

Recipe (documented in models/model_manifest.json):
  * Observation window 2024-01-01/2024-12-31 (a fixed annual composite, NOT real-time).
  * Sentinel-2: scenes with tile cloud cover < 20 %, the 12 least-cloudy per MGRS
    tile; pixels kept only where SCL is 4 (vegetation), 5 (bare soil) or 7
    (unclassified); processing-baseline >= 04.00 offset (-1000 DN) removed;
    read at 200 m; per-pixel temporal median; a pixel needs >= 3 valid scenes.
  * MODIS: MOD11A2 (Terra) only; LST_Day_1km * 0.02 -> Kelvin; kept where QC
    mandatory QA == good, or == "other quality" with LST error <= 2 K.
  * NASADEM: 1 arc-second elevation; slope from finite differences in metres.
  * Sentinel-2 and NASADEM pixels are averaged into 0.01 degree cells (~1.1 km x
    1.0 km); MODIS LST takes the nearest 1 km pixel to each cell centre.
    The effective resolution of the prospectivity surface is therefore ~1 km,
    not the 10 m Sentinel-2 pixel size.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

# GDAL tuning for many small ranged reads of cloud-optimised GeoTIFFs.
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "1")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF,.tiff")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

GRID_RES_DEG = 0.01
OBSERVATION_WINDOW = "2024-01-01/2024-12-31"

S2_MAX_TILE_CLOUD = 20.0
S2_SCENES_PER_TILE = 12
S2_READ_RES_M = 200.0
S2_MIN_VALID_SCENES = 3
S2_SCL_VALID = (4, 5, 7)
S2_BANDS = ("B02", "B04", "B08", "B11", "B12")

LST_SCALE = 0.02

FEATURES = [
    "NDVI",
    "Iron_Oxide_Index",
    "Clay_Hydroxyl_Index",
    "LST_Day_K",
    "elevation",
    "slope",
]
QC_COLUMNS = ["s2_valid_scenes", "s2_pixels", "lst_valid_composites", "dem_pixels"]

RECIPE = {
    "grid_res_deg": GRID_RES_DEG,
    "observation_window": OBSERVATION_WINDOW,
    "sentinel2": {
        "collection": "sentinel-2-l2a",
        "max_tile_cloud_pct": S2_MAX_TILE_CLOUD,
        "scenes_per_tile": S2_SCENES_PER_TILE,
        "read_resolution_m": S2_READ_RES_M,
        "scl_valid_classes": list(S2_SCL_VALID),
        "min_valid_scenes_per_pixel": S2_MIN_VALID_SCENES,
        "offset_correction": "subtract 1000 DN for processing baseline >= 04.00",
        "composite": "per-pixel temporal median, indices computed on the composite",
        "indices": {
            "NDVI": "(B08-B04)/(B08+B04)",
            "Iron_Oxide_Index": "B04/B02",
            "Clay_Hydroxyl_Index": "B11/B12",
        },
    },
    "modis_lst": {
        "collection": "modis-11A2-061 (MOD11A2 Terra only)",
        "band": "LST_Day_1km",
        "scale": LST_SCALE,
        "unit": "K",
        "qc_rule": "QC_Day bits0-1 == 0, or bits0-1 == 1 and bits6-7 <= 1 (error <= 2 K)",
        "temporal_reduction": "mean of valid 8-day composites",
        "spatial_sampling": "nearest MODIS pixel to the cell centre",
    },
    "dem": {"collection": "nasadem", "slope": "degrees, finite differences in metres"},
    "cell_aggregation": "Sentinel-2 and NASADEM: mean of pixels whose centre falls in the 0.01 degree cell; MODIS LST: nearest 1 km pixel to the cell centre",
}


# ---------------------------------------------------------------------------
# Lattice accumulator
# ---------------------------------------------------------------------------

class Lattice:
    """Dense accumulator over the global 0.01 degree lattice restricted to a bbox."""

    def __init__(self, bbox, res=GRID_RES_DEG):
        lon_min, lat_min, lon_max, lat_max = bbox
        self.res = res
        self.i0 = int(math.floor(lat_min / res + 1e-9))
        self.j0 = int(math.floor(lon_min / res + 1e-9))
        self.ni = int(math.ceil(lat_max / res - 1e-9)) - self.i0
        self.nj = int(math.ceil(lon_max / res - 1e-9)) - self.j0
        self.ni, self.nj = max(self.ni, 1), max(self.nj, 1)
        n = self.ni * self.nj
        self.sums = {}
        self.counts = {}
        self._n = n

    def add(self, name, lats, lons, values):
        lats = np.asarray(lats, dtype="float64").ravel()
        lons = np.asarray(lons, dtype="float64").ravel()
        values = np.asarray(values, dtype="float64").ravel()
        ok = np.isfinite(values) & np.isfinite(lats) & np.isfinite(lons)
        ii = np.floor(lats[ok] / self.res).astype(np.int64) - self.i0
        jj = np.floor(lons[ok] / self.res).astype(np.int64) - self.j0
        inside = (ii >= 0) & (ii < self.ni) & (jj >= 0) & (jj < self.nj)
        flat = ii[inside] * self.nj + jj[inside]
        v = values[ok][inside]
        s = self.sums.setdefault(name, np.zeros(self._n))
        c = self.counts.setdefault(name, np.zeros(self._n))
        s += np.bincount(flat, weights=v, minlength=self._n)
        c += np.bincount(flat, minlength=self._n)

    def mean(self, name):
        if name not in self.sums:
            return np.full(self._n, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.counts[name] > 0, self.sums[name] / self.counts[name], np.nan)

    def count(self, name):
        return self.counts.get(name, np.zeros(self._n))

    def centres(self):
        ii, jj = np.divmod(np.arange(self._n), self.nj)
        lat = (self.i0 + ii + 0.5) * self.res
        lon = (self.j0 + jj + 0.5) * self.res
        return np.round(lat, 4), np.round(lon, 4)


def cell_bbox(lat, lon, res=GRID_RES_DEG):
    """Bounding box (lon_min, lat_min, lon_max, lat_max) of the lattice cell containing a point."""
    i = math.floor(lat / res)
    j = math.floor(lon / res)
    return (j * res, i * res, (j + 1) * res, (i + 1) * res)


def cell_centre(lat, lon, res=GRID_RES_DEG):
    b = cell_bbox(lat, lon, res)
    return round((b[1] + b[3]) / 2, 4), round((b[0] + b[2]) / 2, 4)


# ---------------------------------------------------------------------------
# STAC helpers
# ---------------------------------------------------------------------------

def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _search(collection, bbox, datetime=None):
    kw = {"collections": [collection], "bbox": list(bbox)}
    if datetime:
        kw["datetime"] = datetime
    return list(_catalog().search(**kw).items())


def select_s2_items(bbox, window=OBSERVATION_WINDOW):
    """Group Sentinel-2 items by MGRS tile; keep the least-cloudy scenes per tile."""
    items = _search("sentinel-2-l2a", bbox, window)
    by_tile = {}
    for it in items:
        cc = it.properties.get("eo:cloud_cover")
        if cc is None or cc >= S2_MAX_TILE_CLOUD:
            continue
        by_tile.setdefault(it.properties.get("s2:mgrs_tile"), []).append(it)
    out = {}
    for tile, its in by_tile.items():
        its.sort(key=lambda x: (x.properties["eo:cloud_cover"], x.datetime))
        out[tile] = its[:S2_SCENES_PER_TILE]
    return out


# ---------------------------------------------------------------------------
# Sentinel-2
# ---------------------------------------------------------------------------

def _snap_bounds(bounds, origin_x, origin_y, step):
    left, bottom, right, top = bounds
    left = origin_x + math.floor((left - origin_x) / step) * step
    right = origin_x + math.ceil((right - origin_x) / step) * step
    top = origin_y - math.floor((origin_y - top) / step) * step
    bottom = origin_y - math.ceil((origin_y - bottom) / step) * step
    return left, bottom, right, top


def _read_resampled(href, bounds, shape, resampling):
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(href) as src:
        win = from_bounds(*bounds, transform=src.transform)
        return src.read(1, window=win, out_shape=shape, resampling=resampling, boundless=True, fill_value=0)


def _s2_tile_composite(items, bbox, pool):
    """Median composite for one MGRS tile clipped to bbox. Returns (lats, lons, dict of arrays) or None."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform, transform_bounds

    ref = items[0].assets["B11"].href
    with rasterio.open(ref) as src:
        crs = src.crs
        tb = src.bounds
        ox, oy = src.transform.c, src.transform.f
    b = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    b = (max(b[0], tb.left), max(b[1], tb.bottom), min(b[2], tb.right), min(b[3], tb.top))
    if b[0] >= b[2] or b[1] >= b[3]:
        return None
    b = _snap_bounds(b, ox, oy, S2_READ_RES_M)
    w = max(1, int(round((b[2] - b[0]) / S2_READ_RES_M)))
    h = max(1, int(round((b[3] - b[1]) / S2_READ_RES_M)))

    jobs = []
    for k, it in enumerate(items):
        for band in S2_BANDS:
            jobs.append((k, band, it.assets[band].href, Resampling.average))
        jobs.append((k, "SCL", it.assets["SCL"].href, Resampling.nearest))

    def run(job):
        k, band, href, rs = job
        try:
            return k, band, _read_resampled(href, b, (h, w), rs)
        except Exception:
            return k, band, None

    stack = np.full((len(items), len(S2_BANDS), h, w), np.nan, dtype="float32")
    scl = np.zeros((len(items), h, w), dtype="uint8")
    ok_scene = np.ones(len(items), dtype=bool)
    for k, band, arr in pool.map(run, jobs):
        if arr is None:
            ok_scene[k] = False
            continue
        if band == "SCL":
            scl[k] = arr
        else:
            stack[k, S2_BANDS.index(band)] = arr.astype("float32")

    for k, it in enumerate(items):
        baseline = str(it.properties.get("s2:processing_baseline") or "00.00")
        offset = 1000.0 if baseline >= "04.00" else 0.0
        valid = np.isin(scl[k], S2_SCL_VALID) & np.all(stack[k] > 0, axis=0) & ok_scene[k]
        refl = (stack[k] - offset) / 10000.0
        refl = np.clip(refl, 1e-4, None)
        refl[:, ~valid] = np.nan
        stack[k] = refl

    n_valid = np.sum(np.isfinite(stack[:, 0]), axis=0)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            comp = np.nanmedian(stack, axis=0)
    comp[:, n_valid < S2_MIN_VALID_SCENES] = np.nan
    b2, b4, b8, b11, b12 = comp
    with np.errstate(all="ignore"):
        ndvi = (b8 - b4) / (b8 + b4)
        iron = b4 / b2
        clay = b11 / b12

    xs = b[0] + (np.arange(w) + 0.5) * S2_READ_RES_M
    ys = b[3] - (np.arange(h) + 0.5) * S2_READ_RES_M
    X, Y = np.meshgrid(xs, ys)
    lons, lats = transform(crs, "EPSG:4326", X.ravel().tolist(), Y.ravel().tolist())
    return np.array(lats), np.array(lons), {
        "NDVI": ndvi,
        "Iron_Oxide_Index": iron,
        "Clay_Hydroxyl_Index": clay,
        "s2_valid_scenes": np.where(n_valid >= S2_MIN_VALID_SCENES, n_valid, np.nan).astype("float64"),
    }


def add_sentinel2(lattice, bbox, pool, log=print):
    tiles = select_s2_items(bbox)
    for n, (tile, items) in enumerate(sorted(tiles.items())):
        res = _s2_tile_composite(items, bbox, pool)
        if res is None:
            continue
        lats, lons, arrs = res
        for name, arr in arrs.items():
            lattice.add(name, lats, lons, arr)
        lattice.add("s2_pixels", lats, lons, np.where(np.isfinite(arrs["NDVI"]), 1.0, np.nan))
        log(f"  sentinel-2 tile {tile} ({n + 1}/{len(tiles)}): {len(items)} scenes")
    return sorted(tiles)


# ---------------------------------------------------------------------------
# MODIS LST
# ---------------------------------------------------------------------------

def _lst_valid(qc):
    qa = qc & 0b11
    err = (qc >> 6) & 0b11
    return (qa == 0) | ((qa == 1) & (err <= 1))


def add_modis_lst(lattice, bbox, pool, log=print):
    import rasterio
    from rasterio.warp import transform, transform_bounds
    from rasterio.windows import from_bounds

    items = [it for it in _search("modis-11A2-061", bbox, OBSERVATION_WINDOW) if it.id.startswith("MOD11A2")]
    by_tile = {}
    for it in items:
        by_tile.setdefault(it.id.split(".")[2], []).append(it)

    for tile, its in sorted(by_tile.items()):
        with rasterio.open(its[0].assets["LST_Day_1km"].href) as src:
            crs, tr = src.crs, src.transform
            win = from_bounds(*transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21), transform=tr)
            # pad by 2 pixels so every cell centre has a nearest pixel inside the window
            win = rasterio.windows.Window(win.col_off - 2, win.row_off - 2, win.width + 4, win.height + 4)
            win = win.round_offsets(op="floor").round_lengths(op="ceil")
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            wtr = src.window_transform(win)

        def run(it):
            try:
                with rasterio.open(it.assets["LST_Day_1km"].href) as s:
                    lst = s.read(1, window=win).astype("float64")
                with rasterio.open(it.assets["QC_Day"].href) as s:
                    qc = s.read(1, window=win).astype("int64")
                ok = _lst_valid(qc) & (lst > 0)
                return np.where(ok, lst * LST_SCALE, np.nan)
            except Exception:
                return None

        layers = [a for a in pool.map(run, its) if a is not None]
        if not layers:
            continue
        cube = np.stack(layers)
        n = np.sum(np.isfinite(cube), axis=0)
        with np.errstate(all="ignore"):
            mean = np.nansum(cube, axis=0) / np.where(n > 0, n, np.nan)
        h, w = mean.shape
        # nearest MODIS pixel to each cell centre
        c_lat, c_lon = lattice.centres()
        xs, ys = transform("EPSG:4326", crs, c_lon.tolist(), c_lat.tolist())
        cols = np.floor((np.array(xs) - wtr.c) / wtr.a).astype(int)
        rows = np.floor((np.array(ys) - wtr.f) / wtr.e).astype(int)
        inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        vals = np.full(c_lat.shape, np.nan)
        cnt = np.full(c_lat.shape, np.nan)
        vals[inside] = mean[rows[inside], cols[inside]]
        cnt[inside] = n[rows[inside], cols[inside]]
        cnt[cnt == 0] = np.nan
        lattice.add("LST_Day_K", c_lat, c_lon, vals)
        lattice.add("lst_valid_composites", c_lat, c_lon, cnt)
        log(f"  modis tile {tile}: {len(layers)} composites")


# ---------------------------------------------------------------------------
# NASADEM
# ---------------------------------------------------------------------------

def add_dem(lattice, bbox, log=print):
    import rasterio
    from rasterio.windows import from_bounds

    items = _search("nasadem", bbox)
    for it in items:
        with rasterio.open(it.assets["elevation"].href) as src:
            # pad by 2 pixels so slope at the bbox edge uses real neighbours
            px = abs(src.transform.a)
            pb = (bbox[0] - 2 * px, bbox[1] - 2 * px, bbox[2] + 2 * px, bbox[3] + 2 * px)
            win = from_bounds(*pb, transform=src.transform).round_offsets(op="floor").round_lengths(op="ceil")
            win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            z = src.read(1, window=win).astype("float64")
            nod = src.nodata
            wtr = src.window_transform(win)
        if nod is not None:
            z[z == nod] = np.nan
        z[z < -500] = np.nan
        h, w = z.shape
        if h < 3 or w < 3:
            continue
        lats = wtr.f + (np.arange(h) + 0.5) * wtr.e
        lons = wtr.c + (np.arange(w) + 0.5) * wtr.a
        dy = abs(wtr.e) * 111320.0
        dx = abs(wtr.a) * 111320.0 * np.cos(np.radians(lats))[:, None]
        gy, gx = np.gradient(z)
        slope = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
        LON, LAT = np.meshgrid(lons, lats)
        lattice.add("elevation", LAT, LON, z)
        lattice.add("slope", LAT, LON, slope)
        lattice.add("dem_pixels", LAT, LON, np.where(np.isfinite(z), 1.0, np.nan))
    log(f"  nasadem: {len(items)} tiles")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_cell_features(bbox, threads=16, log=print):
    """Compute the exploration feature table for every 0.01 degree cell in bbox.

    bbox = (lon_min, lat_min, lon_max, lat_max). Returns a DataFrame with
    lat, lon (cell centres), FEATURES and QC_COLUMNS. Cells lacking any
    feature keep NaN — callers decide how to treat them (never mean-filled).
    """
    lat_ = Lattice(bbox)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        add_sentinel2(lat_, bbox, pool, log=log)
        add_modis_lst(lat_, bbox, pool, log=log)
    add_dem(lat_, bbox, log=log)

    lat, lon = lat_.centres()
    df = pd.DataFrame({"lat": lat, "lon": lon})
    for f in FEATURES:
        df[f] = lat_.mean(f)
    df["s2_valid_scenes"] = lat_.mean("s2_valid_scenes")
    df["s2_pixels"] = lat_.count("s2_pixels")
    df["lst_valid_composites"] = lat_.mean("lst_valid_composites")
    df["dem_pixels"] = lat_.count("dem_pixels")
    return df


def extract_point_features(lat, lon, threads=16, log=lambda *_: None):
    """Features of the single lattice cell containing (lat, lon)."""
    df = extract_cell_features(cell_bbox(lat, lon), threads=threads, log=log)
    c_lat, c_lon = cell_centre(lat, lon)
    row = df[(np.isclose(df["lat"], c_lat)) & (np.isclose(df["lon"], c_lon))]
    if row.empty:
        row = df.iloc[[0]]
    return row.iloc[0].to_dict()
