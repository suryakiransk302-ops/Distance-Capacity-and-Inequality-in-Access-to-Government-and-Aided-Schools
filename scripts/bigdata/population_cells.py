"""Step D2 worker code: WorldPop 100 m raster blocks -> populated cells tagged with their area units.

Runs inside Spark tasks (notebook 17 ships this file to the workers with addPyFile). One task = a few raster
blocks of 1024 x 1024 cells. For each block it reads the total population and the three age bands, keeps
cells with population > 0 and burns the area polygons onto the same block grid (rasterio rasterize, default
rule: a cell belongs to the polygon that contains its centre, the same rule used to clip the Kerala rasters in
notebook 11). Nothing is shuffled between tasks.

Cell id: row * WIDTH + col on the WorldPop India grid (EPSG:4326, 3 arc-seconds); lon/lat of the cell centre
come back from the id with lonlat_sql().
"""

import os

if "postgis" in os.environ.get("PROJ_LIB", "").lower():  # an old PROJ database from a PostGIS install breaks rasterio
    os.environ.pop("PROJ_LIB")

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box

BANDS = {"pop": "ind_pop_2026_CN_100m_R2025A_v1.tif", "pop_5_9": "ind_t_05_2026_CN_100m_R2025A_v1.tif",
         "pop_10_14": "ind_t_10_2026_CN_100m_R2025A_v1.tif", "pop_15_19": "ind_t_15_2026_CN_100m_R2025A_v1.tif"}
# code column in the cells table -> (silver table, id column, name column)
UNITS = {"state_code": ("states", "shapeID", "shapeName"), "district_code": ("districts", "shapeID", "shapeName"),
         "subdistrict_code": ("subdistricts", "shapeID", "shapeName"),
         "local_body_code": ("kerala_local_bodies", "osm_id", "name")}
SCHEMA = ("cell_id long, pop float, pop_5_9 float, pop_10_14 float, pop_15_19 float, "
          "state_code int, district_code int, subdistrict_code int, local_body_code int, age_nodata boolean")
BLOCK = 1024

_cache = {}


def load_units(silver, code_col):
    """Polygons of one area table with an integer code 1..n (sorted by the table's id); cached per worker."""
    if code_col not in _cache:
        table, id_col, name_col = UNITS[code_col]
        g = gpd.read_parquet(Path(silver) / table).sort_values(id_col, kind="stable").reset_index(drop=True)
        g["code"] = np.arange(1, len(g) + 1, dtype="int32")
        _cache[code_col] = g[["code", id_col, name_col, "geometry"]].rename(columns={id_col: "unit_id", name_col: "unit_name"})
    return _cache[code_col]


def grid(raster_dir):
    with rasterio.open(Path(raster_dir) / BANDS["pop"]) as r:
        return r.width, r.height, r.transform


def blocks(raster_dir, rows=None, cols=None):
    """(row0, col0, height, width) of every block, optionally only inside a row/col range."""
    width, height, _ = grid(raster_dir)
    r_lo, r_hi = rows or (0, height)
    c_lo, c_hi = cols or (0, width)
    return [(r, c, min(BLOCK, r_hi - r), min(BLOCK, c_hi - c))
            for r in range(r_lo, r_hi, BLOCK) for c in range(c_lo, c_hi, BLOCK)]


def lonlat_sql(raster_dir, id_col="cell_id"):
    """Spark SQL expressions for the lon and lat of a cell centre."""
    width, _, t = grid(raster_dir)
    lon = f"({t.c!r} + (({id_col} % {width}) + 0.5) * {t.a!r})"
    lat = f"({t.f!r} + (floor({id_col} / {width}) + 0.5) * {t.e!r})"
    return lon, lat


def block_cells(raster_dir, silver, row0, col0, height, width):
    """Populated cells of one raster block, with their area codes (0 = centre in no polygon)."""
    win = Window(col0, row0, width, height)
    with rasterio.open(Path(raster_dir) / BANDS["pop"]) as r:
        pop = r.read(1, window=win)
        transform, full_width = r.window_transform(win), r.width
        win_box = box(*window_bounds(win, r.transform))
    keep = pop > 0  # nodata is -99999
    if not keep.any():
        return pd.DataFrame()
    rr, cc = np.nonzero(keep)
    out = {"cell_id": (rr + row0).astype("int64") * full_width + (cc + col0), "pop": pop[keep]}
    age_nodata = np.zeros(len(rr), dtype=bool)
    for name, f in BANDS.items():
        if name == "pop":
            continue
        with rasterio.open(Path(raster_dir) / f) as r:
            a = r.read(1, window=win)[keep]
        age_nodata |= ~(a >= 0)
        out[name] = np.where(a >= 0, a, 0).astype("float32")
    for code_col in UNITS:
        g = load_units(silver, code_col)
        hit = g.iloc[g.sindex.query(win_box, predicate="intersects")]
        if len(hit):
            burnt = rasterize(zip(hit.geometry, hit.code), out_shape=(height, width), transform=transform,
                              fill=0, dtype="int32")
            out[code_col] = burnt[keep]
        else:
            out[code_col] = np.zeros(len(rr), dtype="int32")
    out["age_nodata"] = age_nodata
    return pd.DataFrame(out)


def map_blocks(batches, raster_dir, silver):
    """mapInPandas body: input rows are blocks, output rows are cells."""
    for b in batches:
        for row0, col0, height, width in b[["row0", "col0", "height", "width"]].itertuples(index=False):
            df = block_cells(raster_dir, silver, int(row0), int(col0), int(height), int(width))
            if len(df):
                yield df
