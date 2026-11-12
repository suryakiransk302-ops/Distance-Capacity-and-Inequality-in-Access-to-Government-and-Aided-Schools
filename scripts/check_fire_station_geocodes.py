"""Check geocoded fire stations and build a satellite review map.

Checks per station:
  district_check   point lies inside the district it is listed under (geoBoundaries ADM2 2021)
  nearest_km       distance to the nearest other geocoded station (very small = likely duplicate)

Outputs (data/interim/fire_stations/):
  kerala_fire_stations_<provider>_checked.csv
  kerala_fire_stations_<provider>_checked.geojson   (open in QGIS over satellite imagery)
  review_map_<provider>.html                         (open in a browser)

Usage: python scripts/check_fire_station_geocodes.py --provider nominatim
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FS_DIR = ROOT / "data" / "interim" / "fire_stations"
DISTRICTS = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries-IND-ADM2.geojson"
METRIC_CRS = "EPSG:32643"  # UTM 43N covers Kerala
DUPLICATE_KM = 0.2


def check(provider):
    df = pd.read_csv(FS_DIR / f"kerala_fire_stations_{provider}.csv")
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    pts = pts[pts["lat"].notna()].copy()

    districts = gpd.read_file(DISTRICTS)[["shapeName", "geometry"]].rename(columns={"shapeName": "gb_district"})
    pts = gpd.sjoin(pts, districts, how="left", predicate="within").drop(columns="index_right")
    pts["district_check"] = np.where(pts["gb_district"].isna(), "outside_india_polygons",
                                     np.where(pts["gb_district"] == pts["district"], "ok", "wrong_district"))

    m = pts.to_crs(METRIC_CRS)
    xy = np.c_[m.geometry.x, m.geometry.y]
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) / 1000
    np.fill_diagonal(d, np.inf)
    pts["nearest_km"] = d.min(1).round(2)
    pts["nearest_station"] = pts["station_name"].to_numpy()[d.argmin(1)]

    extra = np.where(pts["district_check"] != "ok", pts["district_check"], "")
    extra = np.where(pts["nearest_km"] < DUPLICATE_KM, np.char.add(extra.astype(str), " near_duplicate"), extra)
    pts["review_reason"] = (pts["review_reason"].fillna("") + " " + pd.Series(extra, index=pts.index)).str.strip()
    pts["needs_review"] = pts["review_reason"] != ""

    out = pd.concat([pts.drop(columns="geometry"), df[df["lat"].isna()]]).sort_values("sl_no")
    out.to_csv(FS_DIR / f"kerala_fire_stations_{provider}_checked.csv", index=False, encoding="utf-8")
    pts.to_file(FS_DIR / f"kerala_fire_stations_{provider}_checked.geojson", driver="GeoJSON")
    write_map(pts, FS_DIR / f"review_map_{provider}.html")

    print("district check:", pts["district_check"].value_counts().to_dict())
    print("stations without coordinates:", int(df["lat"].isna().sum()))
    print("closer than", DUPLICATE_KM, "km to another station:", int((pts["nearest_km"] < DUPLICATE_KM).sum()))
    print("accepted:", int((~pts["needs_review"]).sum()), "| needs review:", int(pts["needs_review"].sum()))
    bad = pts[pts["district_check"] != "ok"]
    if len(bad):
        print(bad[["sl_no", "station_name", "district", "gb_district", "match_level"]].to_string(index=False))


def write_map(pts, path):
    """Leaflet map over Esri World Imagery: green = accepted, red = needs review."""
    feats = [{"lat": r.lat, "lon": r.lon, "ok": not r.needs_review,
              "label": f"{r.sl_no}. {r.station_name} ({r.district})<br>matched: {r.matched_name}"
                       f"<br>level: {r.match_level}<br>review: {r.review_reason or '-'}"}
             for r in pts.itertuples()]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Fire station review map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0}}</style></head><body><div id="map"></div><script>
const pts = {json.dumps(feats)};
const map = L.map('map').setView([10.5, 76.3], 8);
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
  {{maxZoom: 19, attribution: 'Imagery: Esri, Maxar, Earthstar Geographics'}}).addTo(map);
const osm = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
  {{maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'}});
L.control.layers({{'Satellite': sat, 'OpenStreetMap': osm}}).addTo(map);
pts.forEach(p => L.circleMarker([p.lat, p.lon], {{radius: 7, color: p.ok ? '#1a9850' : '#d73027', weight: 2,
  fillOpacity: 0.6}}).bindPopup(p.label).addTo(map));
</script></body></html>"""
    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["nominatim", "google"], default="nominatim")
    check(ap.parse_args().provider)
