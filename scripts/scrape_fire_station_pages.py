"""Collect station locations published by Kerala Fire and Rescue Services.

Every station page on https://fire.kerala.gov.in/en/<station>/ shows the station's
address and phone, and embeds a Google Map of that station placed by the department.
The embed URL carries the map centre (!3d = latitude, !2d = longitude) and the
place name (!2s). The footer of every page embeds the Headquarters map, which is
skipped unless the page is the Headquarters' own station.

Raw pages are kept in data/raw/fire_kerala_pages/ (with the retrieval date) as evidence.

Output: data/interim/fire_stations/official_station_pages.csv
Usage:  python scripts/scrape_fire_station_pages.py
"""

import datetime as dt
import html
import re
import time
from pathlib import Path
from urllib.parse import unquote, urljoin

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "fire_kerala_pages"
OUT_CSV = ROOT / "data" / "interim" / "fire_stations" / "official_station_pages.csv"

BASE = "https://fire.kerala.gov.in"
DISTRICT_PAGES = {
    "Thiruvananthapuram": "/en/thiruvananthapuram/", "Kollam": "/en/kadappakada/",
    "Pathanamthitta": "/en/pathanamthitta/", "Alappuzha": "/en/alappuzha/", "Kottayam": "/en/kottayam/",
    "Idukki": "/en/idukki/", "Ernakulam": "/en/gandhi-nagar/", "Thrissur": "/en/thrissur/",
    "Palakkad": "/en/palakkad/", "Malappuram": "/en/malappuram/", "Kozhikode": "/en/meenchanda/",
    "Wayanad": "/en/kalpetta/", "Kannur": "/en/kannur/", "Kasaragod": "/en/kasargode/",
}
HQ_FEATURE = "0xd5619dc644e2ec52"  # Headquarters map, embedded in every page footer
DELAY_S = 1.5


def fetch(session, path):
    """Fetch a page once; reuse the saved copy on later runs."""
    f = RAW_DIR / (path.strip("/").replace("/", "__") + ".html")
    if not f.exists():
        time.sleep(DELAY_S)
        r = session.get(urljoin(BASE, path), timeout=90)
        r.raise_for_status()
        f.write_text(r.text, encoding="utf-8")
    return f.read_text(encoding="utf-8")


def station_links(page):
    """Links listed between 'Our Stations' and 'Stations Near Me'."""
    i = page.find("Our Stations")
    j = page.find("Stations Near Me", i)
    seg = page[i:j] if i >= 0 and j > i else ""
    return list(dict.fromkeys(re.findall(r'href="(/en/[^"#]+/)"', seg)))


def text_of(page):
    page = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", " ", page)
    t = html.unescape(re.sub(r"<[^>]+>", " | ", page))
    return re.sub(r"(\|\s*)+", "| ", re.sub(r"\s+", " ", t))


def parse_station(page, path):
    embeds = [html.unescape(e) for e in re.findall(r'google\.com/maps/embed\?pb=([^"\']+)', page)]
    own = [e for e in embeds if HQ_FEATURE not in e] or embeds[:1]
    rec = {"page_url": urljoin(BASE, path)}
    if own:
        e = own[0]
        lat, lon = re.search(r"!3d(-?[\d.]+)", e), re.search(r"!2d(-?[\d.]+)", e)
        name, feat, span = re.search(r"!2s([^!]+)", e), re.search(r"!1s(0x[0-9a-f]+(?:%3A|:)0x[0-9a-f]+)", e), re.search(r"!1d([\d.]+)", e)
        rec |= {"embed_lat": float(lat.group(1)) if lat else None, "embed_lon": float(lon.group(1)) if lon else None,
                "embed_place_name": unquote(name.group(1)).replace("+", " ") if name else None,
                "embed_feature_id": unquote(feat.group(1)) if feat else None,
                "embed_view_span_m": float(span.group(1)) if span else None,
                "embed_is_hq_map": HQ_FEATURE in e}
    t = text_of(page)
    i = t.find("Contact Information")
    contact = t[i + len("Contact Information"): t.find("Right to Service Act", i)] if i >= 0 else ""
    parts = [p.strip(" ,") for p in contact.split("|") if p.strip(" ,")]
    rec["page_title"] = parts[0] if parts else None
    rec["official_address"] = ", ".join(p for p in parts[1:] if not re.match(r"(?i)^ph", p)) or None
    phone = re.search(r"(?i)ph\s*:?\s*([\d\s/,-]{6,})", contact)
    rec["official_phone"] = phone.group(1).strip(" ,") if phone else None
    pin = re.search(r"\b(6[789]\d{4})\b", contact)
    rec["pin_code"] = pin.group(1) if pin else None
    return rec


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "msc-spatial-inequality-study/0.1 (academic project)"
    rows = []
    for district, dpath in DISTRICT_PAGES.items():
        links = station_links(fetch(session, dpath))
        links = list(dict.fromkeys([dpath] + links))  # the district page is itself a station page
        print(f"{district}: {len(links)} station pages")
        for path in links:
            rows.append({"district": district} | parse_station(fetch(session, path), path))
    df = pd.DataFrame(rows).drop_duplicates("page_url")
    # keep the original retrieval date of pages already saved; only newly fetched pages get today's date
    prev = pd.read_csv(OUT_CSV, usecols=["page_url", "retrieved_on"], dtype=str).set_index("page_url")["retrieved_on"] if OUT_CSV.exists() else {}
    df["retrieved_on"] = df["page_url"].map(prev).fillna(dt.date.today().isoformat())
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("pages:", len(df), "| with own station map:", int((~df["embed_is_hq_map"].fillna(True)).sum()))
    print("written:", OUT_CSV)


if __name__ == "__main__":
    main()
