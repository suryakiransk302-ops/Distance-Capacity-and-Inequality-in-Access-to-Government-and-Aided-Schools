"""Download the NIN Health Facilities directory (all India) from the Esri India Living Atlas.

Layer:   "India: Hospital Directory" (item f8ebe747d6634b6cbe26718b953bb1b5, owner esri_IN_content)
         https://livingatlas.esri.in/server/rest/services/LivingAtlas/IND_Hospital_Directory/MapServer/0
Origin:  MoHFW National Health Portal, "NIN Health Facilities Geo-code and Additional Parameters"
         https://data.gov.in/resources/nin-health-facilties-geo-code-and-additional-parameters-updated-till-last-month
Licence: data.gov.in terms of use (NDSAP / GODL-India), as stated on the item. This is NOT the
         "Healthcare Facilities in India" item (194e18cc...), whose terms forbid export.

The server often answers "Service not started", so each page is retried and saved once
(data/raw/health/nin_hospital_directory/pages/), and a re-run resumes.

Output: data/raw/health/nin_hospital_directory/nin_hospital_directory.csv
Usage:  python scripts/download_nin_hospital_directory.py
"""

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
LAYER = "https://livingatlas.esri.in/server/rest/services/LivingAtlas/IND_Hospital_Directory/MapServer/0"
OUT_DIR = ROOT / "data" / "raw" / "health" / "nin_hospital_directory"
PAGE_DIR = OUT_DIR / "pages"
PAGE = 2000
TRIES = 15


def get(session, url, params):
    for attempt in range(TRIES):
        try:
            r = session.get(url, params=params, timeout=120)
            d = r.json()
            if "error" not in d:
                return d
            msg = d["error"].get("message")
        except (requests.RequestException, ValueError) as e:
            msg = e
        print(f"  retry {attempt + 1}: {msg}", flush=True)
        time.sleep(3 + attempt)
    raise SystemExit(f"gave up on {url} {params}")


def main():
    PAGE_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "msc-spatial-inequality-study/0.1 (academic project)"
    ids = sorted(get(s, LAYER + "/query", {"where": "1=1", "returnIdsOnly": "true", "f": "json"})["objectIds"])
    print("features:", len(ids), flush=True)
    for n, start in enumerate(range(0, len(ids), PAGE)):
        f = PAGE_DIR / f"page_{n:03d}.json"
        if f.exists():
            continue
        chunk = ids[start:start + PAGE]
        d = get(s, LAYER + "/query", {"where": f"objectid >= {chunk[0]} AND objectid <= {chunk[-1]}",
                                      "outFields": "*", "outSR": 4326, "returnGeometry": "true", "f": "json"})
        if len(d["features"]) != len(chunk):
            raise SystemExit(f"page {n}: expected {len(chunk)} features, got {len(d['features'])}")
        f.write_text(json.dumps(d["features"]), encoding="utf-8")
        print(f"page {n + 1}/{-(-len(ids) // PAGE)}", flush=True)

    rows = []
    for f in sorted(PAGE_DIR.glob("page_*.json")):
        for feat in json.loads(f.read_text(encoding="utf-8")):
            g = feat.get("geometry") or {}
            rows.append(feat["attributes"] | {"geom_x": g.get("x"), "geom_y": g.get("y")})
    df = pd.DataFrame(rows).drop(columns=["globalid", "shape"], errors="ignore").drop_duplicates("objectid")
    df["source_layer"] = LAYER
    out = OUT_DIR / "nin_hospital_directory.csv"
    # keep the original retrieval date of rows already saved; only newly fetched rows get today's date
    prev = pd.read_csv(out, usecols=["objectid", "retrieved_on"], dtype=str).set_index("objectid")["retrieved_on"] if out.exists() else {}
    df["retrieved_on"] = df["objectid"].astype(str).map(prev).fillna(dt.date.today().isoformat())
    df.to_csv(out, index=False, encoding="utf-8")
    print("written:", len(df), "rows ->", OUT_DIR / "nin_hospital_directory.csv")


if __name__ == "__main__":
    main()
