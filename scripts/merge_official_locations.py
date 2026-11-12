"""Combine the geocoded stations with the department's own station pages.

For each of the 129 stations:
  - match its official page on fire.kerala.gov.in (name within district)
  - official map position: centre of the department's embedded Google Map. Embeds at the
    standard place zoom (view span ~3.9 km) are centred ~240 m west of the pin, because the
    place card covers the left of the map; that shift is removed. Close zooms (10-20 km view)
    are centred on the pin. Other zooms are rough, and zoomed-out maps (> 20 km) are unusable.
  - agreement: distance between the OSM fire-station point and the official map position

Accepted automatically:
  tier A  OSM and the official map agree within AGREE_M (two independent sources)
  tier B  no agreement, but the official pin is at a usable zoom and its name matches the station
All others go to manual verification.

Output: data/interim/fire_stations/kerala_fire_stations_merged.csv
Usage:  python scripts/merge_official_locations.py
"""

import difflib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from geocode_fire_stations import NAME_MATCH_MIN, name_score, query_name

ROOT = Path(__file__).resolve().parents[1]
FS_DIR = ROOT / "data" / "interim" / "fire_stations"
PAGE_MATCH_MIN = 0.8
STANDARD_SPAN_M = (3800, 4100)
CLOSE_SPAN_M = (10000, 20000)
ZOOMED_OUT_SPAN_M = 20000
EAST_SHIFT_PER_SPAN = 0.0612  # fitted on stations where OSM and the official map clearly coincide
AGREE_M = 100
# District main stations whose pages are named after their locality.
PAGE_ALIASES = {"Kollam": "kadappakada", "Ernakulam": "gandhi-nagar", "Kozhikkode": "meenchanda"}


def key(s):
    return re.sub(r"[^a-z]", "", str(s).lower())


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(h))


def match_page(row, pages):
    if row.station_name in PAGE_ALIASES:
        idx = pages.index[pages["slug"] == PAGE_ALIASES[row.station_name]]
        return pd.Series({"page_idx": idx[0] if len(idx) else None, "page_score": 1.0 if len(idx) else 0.0})
    scores =[(max(difflib.SequenceMatcher(None, key(query_name(row.station_name)), key(t)).ratio(),
                   difflib.SequenceMatcher(None, key(row.station_name), key(t)).ratio()), i)
              for i, p in pages[pages["district"] == row.district].iterrows() for t in (p.page_title, p.slug)]
    score, idx = max(scores) if scores else (0.0, None)
    return pd.Series({"page_idx": idx if score >= PAGE_MATCH_MIN else None, "page_score": round(score, 2)})


def main():
    st = pd.read_csv(FS_DIR / "kerala_fire_stations_nominatim_checked.csv")
    pages = pd.read_csv(FS_DIR / "official_station_pages.csv")
    pages["slug"] = pages["page_url"].str.replace("https://fire.kerala.gov.in/en/", "").str.strip("/")
    pages = pages[~pages["slug"].str.startswith("category")]

    m = st.join(st.apply(match_page, axis=1, pages=pages))
    cols = ["page_url", "official_address", "official_phone", "pin_code", "embed_lat", "embed_lon",
            "embed_place_name", "embed_feature_id", "embed_view_span_m", "embed_is_hq_map", "retrieved_on"]
    m = m.join(pages[cols], on="page_idx")

    own_map = m["embed_lat"].notna() & ~m["embed_is_hq_map"].fillna(True).astype(bool)
    span = m["embed_view_span_m"]
    standard = own_map & span.between(*STANDARD_SPAN_M)
    close = own_map & span.between(*CLOSE_SPAN_M)
    usable = own_map & span.le(ZOOMED_OUT_SPAN_M)
    shift_deg = EAST_SHIFT_PER_SPAN * span / (111320 * np.cos(np.radians(m["embed_lat"])))
    m["official_lat"] = m["embed_lat"].where(usable)
    m["official_lon"] = (m["embed_lon"] + shift_deg.where(standard, 0)).where(usable)
    m["official_pos_quality"] = np.select(
        [m["page_idx"].isna(), ~own_map, standard, close, usable],
        ["no_page", "no_own_map", "standard_zoom_corrected", "close_zoom", "other_zoom_rough"], "zoomed_out_unusable")
    m["embed_name_score"] = [round(name_score(s, n), 2) if isinstance(n, str) else np.nan
                             for s, n in zip(m["station_name"], m["embed_place_name"])]

    osm_ok = m["match_level"].eq("fire_station")
    m["osm_vs_official_m"] = haversine_m(m["lat"], m["lon"], m["official_lat"], m["official_lon"]).where(
        osm_ok & usable).round(0)
    tier_a = m["osm_vs_official_m"].le(AGREE_M)
    pin_is_station = m["embed_name_score"].ge(NAME_MATCH_MIN) | m["station_name"].isin(PAGE_ALIASES)
    tier_b = ~tier_a & (standard | close) & pin_is_station
    m["auto_accept"] = tier_a | tier_b
    m["auto_method"] = np.select([tier_a, tier_b], ["osm_and_official_map_agree", "official_map_pin"], "")

    out = FS_DIR / "kerala_fire_stations_merged.csv"
    m.drop(columns="page_idx").to_csv(out, index=False, encoding="utf-8")
    print("official page found:", int(m["page_url"].notna().sum()), "of", len(m))
    print("official map position:", m["official_pos_quality"].value_counts().to_dict())
    d = m["osm_vs_official_m"].dropna()
    print(f"OSM vs official map (n={len(d)}): median {d.median():.0f} m; "
          + ", ".join(f"<= {t} m: {(d <= t).sum()}" for t in (25, 50, 100, 1000)))
    print("accepted automatically:", m["auto_method"].replace("", "manual").value_counts().to_dict())
    print("written:", out)


if __name__ == "__main__":
    main()
