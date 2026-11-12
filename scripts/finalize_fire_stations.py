"""Build the final Kerala fire and rescue station dataset (129 stations).

Location per station, in order of precedence:
  manual             researcher located the station in Google Maps
                     (data/raw/fire_stations_manual/manual_locations_google_maps_*.csv)
  osm_official_agree OSM point; the department's official map agrees within 100 m
  official_map_pin   pin from the station's official page on fire.kerala.gov.in

Checks: every station has a location; each point lies in its listed district
(geoBoundaries ADM2 2021); no two stations closer than 200 m; manual points are
compared with the OSM point and official pin where those exist.

Outputs (FINAL COMPLETE DATASET/fire_stations/):
  kerala_fire_stations.csv / .geojson / .gpkg
Usage: python scripts/finalize_fire_stations.py
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from merge_official_locations import haversine_m

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "data" / "interim" / "fire_stations" / "kerala_fire_stations_merged.csv"
MANUAL_DIR = ROOT / "data" / "raw" / "fire_stations_manual"
DISTRICTS = ROOT / "data" / "raw" / "boundaries" / "geoBoundaries-IND-ADM2.geojson"
OUT_DIR = ROOT / "FINAL COMPLETE DATASET" / "fire_stations"
LIST_RETRIEVED_ON = "2026-10-27"
NEAR_DUPLICATE_M = 200
SPOT_CHECK_M = 250  # official pin vs OSM disagreement above which a spot check is advised


def main():
    m = pd.read_csv(MERGED)
    manual = pd.concat([pd.read_csv(f) for f in sorted(MANUAL_DIR.glob("manual_locations_*.csv"))])
    manual = manual.drop_duplicates("sl_no", keep="last").set_index("sl_no")

    mismatch = manual.join(m.set_index("sl_no")[["station_name", "district"]], rsuffix="_list")
    mismatch = mismatch[(mismatch["station_name"] != mismatch["station_name_list"]) |
                        (mismatch["district"] != mismatch["district_list"])]
    if len(mismatch):
        raise SystemExit(f"Manual rows do not match the station list:\n{mismatch}")

    is_manual = m["sl_no"].isin(manual.index)
    tier_a = m["auto_method"].eq("osm_and_official_map_agree") & ~is_manual
    tier_b = m["auto_method"].eq("official_map_pin") & ~is_manual
    m["osm_lat"] = m["lat"].where(m["match_level"].eq("fire_station"))
    m["osm_lon"] = m["lon"].where(m["match_level"].eq("fire_station"))

    m["final_lat"] = np.select([is_manual, tier_a, tier_b],
                               [m["sl_no"].map(manual["lat"]), m["lat"], m["official_lat"]], np.nan)
    m["final_lon"] = np.select([is_manual, tier_a, tier_b],
                               [m["sl_no"].map(manual["lon"]), m["lon"], m["official_lon"]], np.nan)
    m["location_method"] = np.select([is_manual, tier_a, tier_b],
                                     ["manual_google_maps", "osm_official_agree", "official_map_pin"], "")
    m["location_source"] = np.select(
        [is_manual, tier_a, tier_b],
        ["Located manually by the researcher in Google Maps",
         "OpenStreetMap (Nominatim), confirmed by the official station map within 100 m",
         "Map pin on the official station page, fire.kerala.gov.in"], "")
    m["location_date"] = np.select([is_manual, tier_a, tier_b],
                                   [m["sl_no"].map(manual["entered_on"]), m["geocoded_on"], m["retrieved_on"]], "")
    m["manual_check_group"] = m["sl_no"].map(manual["check_group"])
    if m["final_lat"].isna().any():
        raise SystemExit(f"Stations without a location: {m.loc[m['final_lat'].isna(), 'station_name'].tolist()}")

    # Manual points vs the automatic candidates (large gaps are expected where those were wrong).
    m["manual_vs_osm_m"] = haversine_m(m["final_lat"], m["final_lon"], m["osm_lat"], m["osm_lon"]).where(is_manual).round(0)
    m["manual_vs_official_m"] = haversine_m(m["final_lat"], m["final_lon"], m["official_lat"], m["official_lon"]).where(is_manual).round(0)

    g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m["final_lon"], m["final_lat"]), crs="EPSG:4326")
    districts = gpd.read_file(DISTRICTS)[["shapeName", "geometry"]].cx[74.5:77.8, 7.9:13.1]
    g = gpd.sjoin(g, districts.rename(columns={"shapeName": "district_spatial"}), how="left", predicate="within")
    g = g.drop(columns="index_right")
    g["district_check"] = np.where(g["district_spatial"].eq(g["district"]), "ok", "mismatch")

    xy = g.to_crs("EPSG:32643").geometry
    d = np.sqrt((xy.x.values[:, None] - xy.x.values) ** 2 + (xy.y.values[:, None] - xy.y.values) ** 2)
    np.fill_diagonal(d, np.inf)
    g["nearest_station_m"] = d.min(1).round(0)
    g["nearest_station"] = g["station_name"].to_numpy()[d.argmin(1)]

    flags = pd.DataFrame({
        "district": np.where(g["district_check"].ne("ok"), "outside_listed_district", ""),
        "dup": np.where(g["nearest_station_m"].lt(NEAR_DUPLICATE_M), "near_another_station", ""),
        # only where the OSM point itself was a credible match (right name, not shared with another station)
        "spot": np.where(g["location_method"].eq("official_map_pin") & g["osm_vs_official_m"].gt(SPOT_CHECK_M)
                         & g["name_score"].ge(0.8) & ~g["review_reason"].fillna("").str.contains("shared_place"),
                         "spot_check_recommended", ""),
    }, index=g.index)
    g["qa_flags"] = flags.apply(lambda r: " ".join(x for x in r if x), axis=1)

    g = g.sort_values("sl_no")
    g["facility_id"] = [f"KL_FRS_{n:03d}" for n in g["sl_no"]]
    g["service"] = "fire_and_rescue"
    g["state"] = "Kerala"
    g["station_list_source"] = g["source_url"]
    g["station_list_retrieved_on"] = LIST_RETRIEVED_ON
    g["lat"], g["lon"] = g["final_lat"].round(6), g["final_lon"].round(6)
    cols = ["facility_id", "sl_no", "station_name", "station_official", "district", "state", "service",
            "lat", "lon", "location_method", "location_source", "location_date", "manual_check_group",
            "official_address", "official_phone", "page_url", "station_list_source", "station_list_retrieved_on",
            "place_id", "osm_vs_official_m", "manual_vs_osm_m", "manual_vs_official_m",
            "district_check", "nearest_station", "nearest_station_m", "qa_flags", "geometry"]
    out =gpd.GeoDataFrame(g[cols].rename(columns={"place_id": "osm_place_id"}), geometry="geometry", crs="EPSG:4326")
    out.loc[~out["location_method"].eq("osm_official_agree"), "osm_place_id"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.drop(columns="geometry").to_csv(OUT_DIR / "kerala_fire_stations.csv", index=False, encoding="utf-8")
    out.to_file(OUT_DIR / "kerala_fire_stations.geojson", driver="GeoJSON")
    out.to_file(OUT_DIR / "kerala_fire_stations.gpkg", layer="fire_stations", driver="GPKG")

    print("stations:", len(out), "| with location:", int(out["lat"].notna().sum()))
    print("location method:", out["location_method"].value_counts().to_dict())
    print("district check:", out["district_check"].value_counts().to_dict())
    print("per district:", out.groupby("district").size().to_dict())
    flagged = out[out["qa_flags"] != ""]
    print(f"\nQA flags ({len(flagged)}):")
    print(flagged[["sl_no", "station_name", "district", "location_method", "qa_flags", "nearest_station",
                   "nearest_station_m"]].to_string(index=False))
    man = out[out["location_method"].eq("manual_google_maps")]
    print("\nManual points vs automatic candidates (m):")
    print(man[["sl_no", "station_name", "manual_check_group", "manual_vs_osm_m", "manual_vs_official_m"]].to_string(index=False))
    print("\nwritten:", OUT_DIR)


if __name__ == "__main__":
    main()
