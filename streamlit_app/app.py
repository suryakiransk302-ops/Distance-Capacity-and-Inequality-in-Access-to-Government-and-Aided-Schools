"""School Access Explorer: the Streamlit version of the dashboard (notebook 29).

It reads exactly the same data as the HTML dashboard (school_access_dashboard_data.json, written by notebook 29)
and uses the same rules: 7 equal-count colour classes, children-weighted totals, the same wording.
Run:  streamlit run streamlit_app/app.py
"""
import json
import math
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="School Access Explorer", page_icon="🏫", layout="wide")

DATA = Path(__file__).resolve().parent.parent / "FINAL COMPLETE DATASET" / "results" / "dashboard" / "school_access_dashboard_data.json"

# ---------- constants: identical to the HTML dashboard ----------
UNITS = {"in": [("in_state", "States"), ("in_district", "Districts"), ("in_sub", "Sub-districts")],
         "k": [("k_district", "Districts"), ("k_taluk", "Taluks"), ("k_lb", "Local bodies")]}
LEVELS = {"in": ["primary", "upper_primary", "secondary", "higher_secondary"], "k": ["lp", "up", "hs", "hss"]}
MEAS = {"b": ("% of children beyond the distance norm", 1), "n": ("mean distance to nearest school (km)", 1),
        "m": ("seats per child (norm-based)", -1), "e": ("seats per child (enrolment check)", -1), "type": ("area type (K-Means)", 0)}
LDESC = {"b": "Share of children who live farther than the distance norm from the nearest school of this level.",
         "n": "Average straight-line distance from a child to the nearest school of this level.",
         "m": "School seats a child can reach, shared with the other children nearby. 1 = a seat for every child.",
         "e": "Same as seats per child, but seats = the pupils enrolled now (check run)."}
PAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NODATA, CLUSTER = "#b4b3ac", "#eb6834"
TYPEC = {"Well served": "#1baf7a", "Crowded: near but few places": "#eda100", "Far from primary school": "#eb6834",
         "Higher secondary far": "#2a78d6", "Remote: far at every level": "#6250d6", "Not classified": NODATA}
NORM = {"primary": "1 km", "upper_primary": "3 km", "secondary": "5 km", "higher_secondary": "7 km",
        "lp": "1 km", "up": "3 km", "hs": "5 km (3 km in hilly areas)", "hss": "7 km"}
SHORT = {"primary": "Primary", "upper_primary": "Upper primary", "secondary": "Secondary", "higher_secondary": "Higher sec.",
         "lp": "LP", "up": "UP", "hs": "HS", "hss": "HSS"}
TYPED = {"Well served": "Schools are near and there are enough seats at most levels.",
         "Crowded: near but few places": "Schools are close, but there are too few seats for the number of children.",
         "Far from primary school": "Many young children live more than 1 km from a primary school.",
         "Higher secondary far": "Lower levels are fine, but many teenagers live far from a higher secondary school.",
         "Remote: far at every level": "Children live far from schools at every level.",
         "Not classified": "Not grouped (too few children or no data)."}
ATTRIB = "Boundaries: geoBoundaries (CC BY 4.0) · Kerala local bodies © OpenStreetMap contributors (ODbL) · Population: WorldPop (CC BY 4.0)"


@st.cache_data(show_spinner="Loading the results…")
def load():
    D = json.loads(DATA.read_text(encoding="utf-8"))
    for fc in D["layers"].values():
        for f in fc["features"]:
            if str(f["properties"].get("name", "")).lower() == "data not available":
                f["properties"]["name"] = "Unnamed area"
    return D


def fmt(x, d):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:,.{d}f}"


def rgb(h, a=235):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)] + [a]


def agg(feats, lv):
    """Children-weighted totals over a set of areas (same as the HTML dashboard)."""
    C = B = CB = N = CN = M = CM = 0.0
    for f in feats:
        p = f["properties"]; c = p.get("c_" + lv)
        if not c:
            continue
        C += c
        if p.get("b_" + lv) is not None: B += c * p["b_" + lv]; CB += c
        if p.get("n_" + lv) is not None: N += c * p["n_" + lv]; CN += c
        if p.get("m_" + lv) is not None: M += c * p["m_" + lv]; CM += c
    return {"c": C, "b": B / CB if CB else None, "n": N / CN if CN else None, "m": M / CM if CM else None, "far": B / 100 if CB else None}


def breaks(vals):
    v = sorted(x for x in vals if x is not None and math.isfinite(x))
    if not v:
        return []
    out = []
    for i in range(1, 7):
        x = v[math.floor(i * len(v) / 7)]
        if x not in out:
            out.append(x)
    return out


def colour_index(x, b, d):
    c = 0
    while c < len(b) and x >= b[c]:
        c += 1
    c = math.floor(c * (len(PAL) - 1) / max(len(b), 1) + 0.5)
    return len(PAL) - 1 - c if d < 0 else c


def val(p, m, lv):
    if m == "type":
        return p.get("type") or "Not classified"
    x = p.get(f"{m}_{lv}")
    if p.get("f_" + lv) or x is None:
        return "no data"
    return f"{fmt(x, 1)} %" if m == "b" else f"{fmt(x, 2)} km" if m == "n" else fmt(x, 2)


def label(p, u):
    if u == "in_state":
        return p["name"]
    rest = (p.get("district") or "Kerala") if u.startswith("k") else ", ".join(x for x in [p.get("district"), p.get("state")] if x)
    return f'{p["name"]} — {rest}'


def seat_word(x):
    return "enough seats" if x >= 1 else "slightly too few seats" if x >= 0.8 else "too few seats" if x >= 0.5 else "far too few seats"


def far_word(b):
    return ("almost every child is within reach" if b < 5 else "most children are within reach" if b < 20
            else "many children live too far" if b < 50 else "most children live too far")


def describe(p, u):
    """The plain-language description of one area (same wording as the HTML dashboard)."""
    ls = LEVELS["k" if u.startswith("k") else "in"]
    out = [f"### {p['name']}", ", ".join(x for x in [p.get("district"), p.get("state") if u != "in_state" else None] if x)]
    if p.get("note"):
        out.append(f"*{p['note']}*")
    if p.get("type"):
        out.append(f"**Area type: {p['type']}.** {TYPED.get(p['type'], '')}")
    ok = [l for l in ls if not p.get("f_" + l) and p.get("b_" + l) is not None]
    if ok:
        far = max(ok, key=lambda l: p["b_" + l])
        ms = [l for l in ok if p.get("m_" + l) is not None]
        few = min(ms, key=lambda l: p["m_" + l]) if ms else None
        s = (f"the biggest distance problem is at **{LV[far]}**: {fmt(p['b_' + far], 0)} % of children live more than {NORM[far]} from the nearest school."
             if p["b_" + far] >= 20 else "at every level, most children live within the distance norm.")
        if few:
            m = p["m_" + few]
            if m < 0.95:
                k = fmt(m * 10, 0)
                s += f" The fewest seats are at **{LV[few]}**: {'less than 1 seat' if m < 0.05 else 'about ' + k + (' seat' if k == '1' else ' seats')} for every 10 children."
            elif m < 1:
                s += f" Seats are slightly fewer than children at **{LV[few]}**."
            else:
                s += " There are enough seats at every level."
        out.append(f"> **In short:** {s}")
    for l in ls:
        c, b, n, m = (p.get(f"{k}_{l}") for k in "cbnm")
        head = f"**{LV[l]}**"
        if p.get("hf_" + l): head += " · :orange[in a cluster of areas where children live far]"
        if p.get("hp_" + l): head += " · :orange[in a cluster of areas with few seats]"
        if p.get("f_" + l) or b is None:
            out.append(head + "  \nNo data for this level."); continue
        t = f"About {fmt(c, 0)} children. {fmt(b, 1)} % live more than {NORM[l]} from the nearest school (on average {fmt(n, 1)} km away): {far_word(b)}."
        t += " No school of this level is within reach, so no seats." if m is None else f" {fmt(m, 2)} seats per child: {seat_word(m)}."
        out.append(head + "  \n" + t)
    if p.get("types"):
        out.append("Its sub-districts by area type: " + ", ".join(f"{k} {v}" for k, v in p["types"].items()) + ".")
    out.append("*Children = those expected in Government + Aided schools. Seats per child: 1 = a seat for every child. Distances are straight-line, so real journeys are longer.*")
    return "\n\n".join(out)


def bars(df, x, y, colour="#2a78d6", fmt_x=",.1f", ref=None, height=None, sort=None, colour_col=None):
    base = alt.Chart(df).encode(y=alt.Y(f"{y}:N", sort=sort or "-x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=220)), x=alt.X(f"{x}:Q", title=None),
                                tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt_x)])
    bar = base.mark_bar(cornerRadiusEnd=3).encode(color=alt.Color(f"{colour_col}:N", scale=None)) if colour_col else base.mark_bar(color=colour, cornerRadiusEnd=3)
    ch = bar + base.mark_text(align="left", dx=4).encode(text=alt.Text(f"{x}:Q", format=fmt_x))
    if ref is not None:
        ch += alt.Chart(pd.DataFrame({"r": [ref]})).mark_rule(color="#888", strokeDash=[4, 3]).encode(x="r:Q")
    return ch.properties(height=height or max(150, 30 * len(df)))


def kpi_row(items):
    """KPI cards that show the full caption (st.metric cuts long labels)."""
    cards = "".join(f"<div style='flex:1;min-width:150px;border:1px solid rgba(128,128,128,.25);border-left:4px solid #2a78d6;border-radius:12px;padding:10px 14px'>"
                    f"<div style='font-size:30px;font-weight:700;line-height:1.15'>{v}</div><div style='font-size:12.5px;opacity:.8;margin-top:4px'>{c}</div></div>"
                    for v, c in items)
    st.markdown(f"<div style='display:flex;gap:12px;flex-wrap:wrap;margin:6px 0 14px'>{cards}</div>", unsafe_allow_html=True)


D = load()
LV = D["lv"]

# ---------- sidebar: filters (same choices as the HTML dashboard) ----------
with st.sidebar:
    st.markdown("### School Access Explorer")
    st.caption("Government + Aided schools · India and Kerala · distance and seats for every 100 m cell")
    view = st.radio("Where", ["India", "Kerala"], horizontal=True)
    v = "in" if view == "India" else "k"
    states = sorted(f["properties"]["name"] for f in D["layers"]["in_state"]["features"])
    state = st.selectbox("State / UT", ["All India"] + states) if v == "in" else "All India"
    st_sel = "" if state == "All India" else state
    units = UNITS["in"][1:] if st_sel else UNITS[v]
    u = dict((t, k) for k, t in units)[st.selectbox("Areas", [t for _, t in units], index=0 if st_sel else 1)]
    lv = dict((LV[l], l) for l in LEVELS[v])[st.selectbox("School level", [LV[l] for l in LEVELS[v]])]
    ms = ["type", "b", "n", "m", "e"] if u == "in_sub" else ["b", "n", "m", "e"] + (["type"] if u == "k_taluk" else [])
    m = dict((MEAS[k][0], k) for k in ms)[st.selectbox("Show on map", [MEAS[k][0] for k in ms])]
    hot_on = st.toggle("Outline poor-access clusters (Gi*)", value=True)
    pri_on = st.toggle("Priority areas for new schools", value=True) if v == "k" else False

all_f = D["layers"][u]["features"]
feats = [f for f in all_f if not st_sel or f["properties"].get("state") == st_sel]
labels = [label(f["properties"], u) for f in feats]

tab_map, tab_find, tab_about = st.tabs(["Explore map", "Key findings", "About the data"])

# ---------- Explore map ----------
with tab_map:
    scope = "Kerala" if v == "k" else (st_sel or "All India")
    kf = D["layers"]["k_district"]["features"] if v == "k" else [f for f in D["layers"]["in_state"]["features"] if not st_sel or f["properties"]["name"] == st_sel]
    a = agg(kf, lv)
    C = a["c"]
    big = (fmt(C / 1e6, 1), "M") if C >= 1e6 else (fmt(C / 1e3, 0), "k") if C >= 1e3 else (fmt(C, 0), "")
    st.markdown(f"Showing **{scope}** · {LV[lv]}")
    kpi_row([(f"{big[0]} {big[1]}".strip(), "children who need a Govt + Aided school"), (f"{fmt(a['b'], 1)} %", f"live more than {NORM[lv]} from a school"),
             (f"{fmt(a['n'], 2)} km", "average distance to the nearest school"), (fmt(a["m"], 2), "seats per child (1 = a seat for every child)")])

    key, d = f"{m}_{lv}", MEAS[m][1]
    b = [] if m == "type" else breaks([f["properties"].get(key) for f in all_f if not f["properties"].get("f_" + lv)])  # all-India classes
    fine = u in ("in_sub", "k_lb")
    for f, lab in zip(feats, labels):
        p = f["properties"]
        if m == "type":
            col = TYPEC.get(p.get("type"), NODATA)
        elif p.get("f_" + lv) or p.get(key) is None:
            col = NODATA
        else:
            col = PAL[colour_index(p[key], b, d)]
        hot = hot_on and (p.get("hf_" + lv) or p.get("hp_" + lv))
        p["_fill"] = rgb(col)
        p["_line"] = rgb("#0b0b0b" if m == "type" else CLUSTER, 255) if hot else [255, 255, 255, 200]
        p["_w"] = (1.1 if fine else 2.4) if hot else (0.25 if fine else 0.6)
        p["_tip"] = f"<b>{p['name']}</b>" + (f"<br/>{p['district']}" if p.get("district") and u != "k_district" else "") + f"<br/>{'Area type' if m == 'type' else MEAS[m][0]}: <b>{val(p, m, lv)}</b>"
        p["_label"] = lab

    pick = st.selectbox("Find an area", ["—"] + sorted(labels), index=0, placeholder="Type a name, e.g. Wayanad")

    def bounds(fs):
        lo1 = la1 = 1e9; lo2 = la2 = -1e9
        def walk(c):
            nonlocal lo1, la1, lo2, la2
            if isinstance(c[0], (int, float)):
                lo1, la1, lo2, la2 = min(lo1, c[0]), min(la1, c[1]), max(lo2, c[0]), max(la2, c[1])
            else:
                for cc in c: walk(cc)
        for f in fs: walk(f["geometry"]["coordinates"])
        return lo1, la1, lo2, la2

    if pick != "—" and pick != st.session_state.get("last_pick"):
        st.session_state["sel_label"] = pick
    st.session_state["last_pick"] = pick
    sel = st.session_state.get("sel_label")
    focus = [f for f in feats if f["properties"]["_label"] == sel] if sel else []
    lo1, la1, lo2, la2 = bounds(focus or feats)
    zw = math.log2(1000 * 360 / (512 * max(lo2 - lo1, 0.01)))            # zoom that fits the width of the map (about 1000 px)
    zh = math.log2(620 * 360 / (512 * max((la2 - la1) * 1.15, 0.01)))     # and its height (620 px)
    view_state = pdk.ViewState(longitude=(lo1 + lo2) / 2, latitude=(la1 + la2) / 2, zoom=max(3.2, min(11, min(zw, zh) - 0.2)))

    layers = [pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": feats}, id="areas", pickable=True, stroked=True, filled=True,
                        get_fill_color="properties._fill", get_line_color=[255, 255, 255, 200], line_width_min_pixels=0.25 if fine else 0.6,
                        auto_highlight=True, highlight_color=[11, 11, 11, 60])]
    hot_f = [f for f in feats if f["properties"]["_w"] > 1]   # Gi* poor-access clusters, drawn as outlines on top
    if hot_f:
        layers.append(pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": hot_f}, id="clusters", filled=False, stroked=True,
                                get_line_color=rgb("#0b0b0b" if m == "type" else CLUSTER, 255), line_width_min_pixels=1.1 if fine else 2.4))
    if u == "k_lb" and D.get("gaps"):
        layers.append(pdk.Layer("GeoJsonLayer", D["gaps"], id="gaps", filled=True, stroked=True, get_fill_color=[158, 158, 158, 140],
                                get_line_color=[82, 81, 78, 255], line_width_min_pixels=1))
    if focus:
        layers.append(pdk.Layer("GeoJsonLayer", {"type": "FeatureCollection", "features": focus}, id="selected", filled=False, stroked=True,
                                get_line_color=[11, 11, 11, 255], line_width_min_pixels=3))
    if pri_on:
        pts = pd.DataFrame([p for p in D["pts"] if p["level"] == lv])
        if len(pts):
            pts["r"] = 3 + pts["children"] ** 0.5 / 6
            pts["_tip"] = pts.apply(lambda r: f"<b>Priority area · {LV[lv]}</b><br/>{fmt(r.children, 0)} children with no {LV[lv]} school within the norm<br/>{r.cells} cells of 100 m · {r.local_body}, {r.district}", axis=1)
            pts["_label"] = ""
            layers.append(pdk.Layer("ScatterplotLayer", pts, id="priority", pickable=True, get_position=["lon", "lat"], get_radius="r", radius_units=pdk.types.String("pixels"),
                                    get_fill_color=[232, 123, 164, 180], get_line_color=[11, 11, 11, 255], stroked=True, line_width_min_pixels=1))
    deck = pdk.Deck(layers=layers, initial_view_state=view_state, map_style="", map_provider=None,  # no background map, like the HTML dashboard
                    tooltip={"html": "{_tip}", "style": {"fontSize": "12.5px"}})

    left, right = st.columns([3, 1])
    with left:
        ev = st.pydeck_chart(deck, height=620, on_select="rerun", selection_mode="single-object", key="map")
        st.caption(ATTRIB)
        try:
            objs = ((ev or {}).get("selection") or {}).get("objects", {}).get("areas")
        except AttributeError:
            objs = None
        if objs:
            got = objs[0].get("properties", objs[0]).get("_label")
            if got and got != st.session_state.get("sel_label"):
                st.session_state["sel_label"] = got
                st.rerun()
    with right:
        st.markdown(f"**{MEAS[m][0] if m != 'type' else 'Area type (K-Means)'}**  \n{LV[lv] if m != 'type' else ''}")
        rows = []
        if m == "type":
            st.caption("Sub-districts grouped by the kind of school-access problem they have.")
            for t, c in TYPEC.items():
                rows.append(f"<div style='display:flex;gap:8px;align-items:center;margin:3px 0'><i style='width:22px;height:12px;background:{c};border-radius:3px;display:inline-block'></i><b>{t}</b></div><div style='font-size:11px;color:#777;margin:0 0 4px 30px'>{TYPED[t]}</div>")
        else:
            st.caption(LDESC[m])
            lo, hi = [None] + b, b + [None]
            dd = 1 if m == "b" else 2 if m == "n" else 3
            for i, x in enumerate(lo):
                c = math.floor(i * (len(PAL) - 1) / max(len(b), 1) + 0.5); c = len(PAL) - 1 - c if d < 0 else c
                tag = " <i>good</i>" if c == 0 else " <i>poor</i>" if c == len(PAL) - 1 else ""
                txt = f"&lt; {fmt(hi[i], dd)}" if x is None else f"≥ {fmt(x, dd)}" if hi[i] is None else f"{fmt(x, dd)} – {fmt(hi[i], dd)}"
                rows.append(f"<div style='display:flex;gap:8px;align-items:center;margin:2px 0'><i style='width:22px;height:12px;background:{PAL[c]};border-radius:3px;display:inline-block'></i>{txt}{tag}</div>")
            rows.append(f"<div style='display:flex;gap:8px;align-items:center;margin:8px 0 2px'><i style='width:22px;height:12px;background:{NODATA};border-radius:3px;display:inline-block'></i>no data (school list incomplete or too few children)</div>")
        if hot_on and u in ("in_sub", "k_lb"):
            rows.append(f"<div style='display:flex;gap:8px;align-items:center;margin:2px 0'><i style='width:22px;height:12px;outline:2.5px solid {'#0b0b0b' if m == 'type' else CLUSTER};outline-offset:-2.5px;display:inline-block'></i>outline: part of a poor-access cluster (Gi*)</div>")
        if u == "k_lb":
            rows.append("<div style='display:flex;gap:8px;align-items:center;margin:2px 0'><i style='width:22px;height:12px;background:#9e9e9e;outline:1px dashed #52514e;display:inline-block'></i>no local-body boundary in OSM</div>")
        if pri_on:
            rows.append("<div style='display:flex;gap:8px;align-items:center;margin:2px 0'><i style='width:13px;height:13px;background:#e87ba4;border-radius:50%;display:inline-block;margin:0 5px 0 4px'></i>priority area for new schools (size = children)</div>")
        if m != "type":
            rows.append(f"<div style='color:#888;margin-top:4px'>light = good, dark = poor · 7 equal-count groups{' · all-India scale' if st_sel else ''}</div>")
        st.markdown("<div style='font-size:12.5px;line-height:18px'>" + "".join(rows) + "</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Selected area")
        if focus:
            st.markdown(describe(focus[0]["properties"], u))
            if st.button("Clear selection"):
                st.session_state.pop("sel_label", None); st.rerun()
        else:
            st.info("Click an area on the map, or use “Find an area”, to read what we found there.")
    with c2:
        if m == "type":
            st.markdown("##### Area types here")
            cnt = pd.Series([f["properties"].get("type") or "Not classified" for f in feats]).value_counts()
            df = pd.DataFrame({"type": cnt.index, "areas": cnt.values, "col": [TYPEC.get(t, NODATA) for t in cnt.index]})
            st.altair_chart(bars(df, "areas", "type", fmt_x=",.0f", colour_col="col"), width="stretch")
        else:
            dirn = MEAS[m][1]
            rows = [f["properties"] for f in feats if f["properties"].get(key) is not None and not f["properties"].get("f_" + lv) and (f["properties"].get("c_" + lv) or 0) >= 100]
            rows.sort(key=lambda p: -p[key] if dirn > 0 else p[key])
            top = rows[:10]
            st.markdown(f"##### {'Highest' if dirn > 0 else 'Lowest'} 10 here: {MEAS[m][0]}")
            if top:
                df = pd.DataFrame({"area": [f"{i + 1}. {p['name']}" for i, p in enumerate(top)], "value": [p[key] for p in top]})
                st.altair_chart(bars(df, "value", "area", colour="#184f95" if dirn > 0 else "#256abf", fmt_x=",.1f" if m == "b" else ",.2f",
                                     ref=1 if m in ("m", "e") else None, sort=list(df.area)), width="stretch")
                st.caption(f"{'Worst first' if dirn > 0 else 'Fewest seats first'} · areas with at least 100 children{' · line = 1 seat per child' if m in ('m', 'e') else ''}")
            else:
                st.info("No areas with data.")

# ---------- Key findings ----------
with tab_find:
    S, IN, K = D["layers"]["in_state"]["features"], LEVELS["in"], D["layers"]["k_district"]["features"]
    ker = [f for f in S if f["properties"]["name"] == "Kerala"]
    nat = {l: agg(S, l) for l in IN}; kin = {l: agg(ker, l) for l in IN}
    types = pd.Series([f["properties"].get("type") for f in D["layers"]["in_sub"]["features"] if f["properties"].get("type") not in (None, "Not classified")]).value_counts()
    st.subheader("Key findings")
    st.write("What the 140 million population cells tell us about Government and Aided schools. All charts are computed from the same results as the map.")
    kpi_row([(f"{fmt(nat['primary']['far'] / 1e6, 1)} million", "primary children live more than 1 km from a primary school"),
             (fmt(nat["higher_secondary"]["m"], 2), f"higher-secondary seats per child in India: about {fmt(nat['higher_secondary']['m'] * 10, 0)} for every 10 children"),
             (fmt(int(types.sum()), 0), "sub-districts grouped into 5 area types by the kind of problem"),
             (fmt(len(D["pts"]), 0), "priority areas in Kerala where children live beyond the norm")])

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Distance hurts primary; seats are short at higher secondary**  \nIndia, Government + Aided schools, by school level")
        df = pd.DataFrame({"level": [f"{SHORT[l]} ({NORM[l]})" for l in IN], "beyond norm (%)": [nat[l]["b"] for l in IN]})
        st.altair_chart(bars(df, "beyond norm (%)", "level", colour="#184f95", sort=list(df.level)), width="stretch")
        df = pd.DataFrame({"level": [SHORT[l] for l in IN], "seats per child": [nat[l]["m"] for l in IN]})
        st.altair_chart(bars(df, "seats per child", "level", colour="#3987e5", fmt_x=",.2f", ref=1, sort=list(df.level)), width="stretch")
        st.caption("Primary has a 1 km norm, so distance is the main gap there. At higher secondary there are only about 4 seats for every 10 children.")
    with g2:
        st.markdown("**Kerala against India, same school levels**  \nKerala values from the all-India run, so the levels and data match")
        rows = []
        for l in IN:
            rows += [{"level": LV[l], "who": "India", "beyond norm (%)": nat[l]["b"], "seats per child": nat[l]["m"]},
                     {"level": LV[l], "who": "Kerala", "beyond norm (%)": kin[l]["b"], "seats per child": kin[l]["m"]}]
        df = pd.DataFrame(rows)
        for col, f2 in [("beyond norm (%)", ",.1f"), ("seats per child", ",.2f")]:
            st.altair_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
                y=alt.Y("level:N", title=None, sort=[LV[l] for l in IN], axis=alt.Axis(labelOverlap=False, labelLimit=220)), yOffset=alt.YOffset("who:N", sort=["India", "Kerala"]), x=alt.X(f"{col}:Q"),
                color=alt.Color("who:N", scale=alt.Scale(domain=["India", "Kerala"], range=["#2a78d6", "#eb6834"]), legend=alt.Legend(title=None, orient="top")),
                tooltip=["level", "who", alt.Tooltip(col, format=f2)]).properties(height=260), width="stretch")
        st.caption("Kerala has far more seats per child at every level and is close to universal above primary, but slightly more of its primary children live beyond 1 km (hilly Idukki and Wayanad).")

    stp = [f["properties"] for f in S if (f["properties"].get("c_primary") or 0) >= 1e5]
    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**States with the most primary children far from a school**  \nShare of primary children beyond 1 km · top 10 states")
        t = sorted([p for p in stp if not p.get("f_primary") and p.get("b_primary") is not None], key=lambda p: -p["b_primary"])[:10]
        df = pd.DataFrame({"state": [p["name"] for p in t], "beyond 1 km (%)": [p["b_primary"] for p in t]})
        st.altair_chart(bars(df, "beyond 1 km (%)", "state", colour="#184f95", sort=list(df.state)), width="stretch")
        st.caption("States with incomplete school lists are left out.")
    with g4:
        st.markdown("**States with the fewest higher-secondary seats**  \nSeats per child at higher secondary (classes 11–12) · 10 lowest states")
        t = sorted([p for p in stp if not p.get("f_higher_secondary") and p.get("m_higher_secondary") is not None], key=lambda p: p["m_higher_secondary"])[:10]
        df = pd.DataFrame({"state": [p["name"] for p in t], "seats per child": [p["m_higher_secondary"] for p in t]})
        st.altair_chart(bars(df, "seats per child", "state", colour="#256abf", fmt_x=",.2f", ref=1, sort=list(df.state)), width="stretch")
        st.caption("Line = 1 seat for every child.")

    g5, g6 = st.columns(2)
    with g5:
        st.markdown("**Most of the inequality is inside states**  \nTheil index of seats per child, split into between-state and within-state parts")
        iq = {r["level"]: r for r in D.get("ineq", []) if r["variable"] == "a_main"}
        rows = []
        for l in IN:
            if l in iq:
                rows += [{"level": f"{SHORT[l]} · Gini {fmt(iq[l]['gini'], 2)}", "part": "between states", "share (%)": iq[l]["between"]},
                         {"level": f"{SHORT[l]} · Gini {fmt(iq[l]['gini'], 2)}", "part": "within states", "share (%)": 100 - iq[l]["between"]}]
        df = pd.DataFrame(rows)
        st.altair_chart(alt.Chart(df).mark_bar().encode(
            y=alt.Y("level:N", title=None, sort=list(dict.fromkeys(df.level)), axis=alt.Axis(labelOverlap=False, labelLimit=220)), x=alt.X("share (%):Q", stack="zero", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("part:N", scale=alt.Scale(domain=["between states", "within states"], range=["#eb6834", "#2a78d6"]), legend=alt.Legend(title=None, orient="top")),
            order=alt.Order("part:N"), tooltip=["level", "part", alt.Tooltip("share (%):Q", format=",.0f")]).properties(height=alt.Step(34)), width="stretch")
        st.caption("At primary only 9 % lies between states, so planning by district and block matters most.")
    with g6:
        st.markdown("**Five kinds of problem areas**  \nK-Means groups of India sub-districts (silhouette 0.42)")
        df = pd.DataFrame({"type": types.index, "sub-districts": types.values, "col": [TYPEC[t] for t in types.index]})
        st.altair_chart(bars(df, "sub-districts", "type", fmt_x=",.0f", colour_col="col"), width="stretch")
        st.caption("The crowded group is the biggest: schools are near, but seats are short.")

    g7, g8 = st.columns(2)
    with g7:
        st.markdown("**Kerala: lower primary children beyond 1 km, by district**  \nThe main gap in Kerala is distance for its youngest pupils")
        kd = sorted([f["properties"] for f in K if f["properties"].get("b_lp") is not None], key=lambda p: -p["b_lp"])
        df = pd.DataFrame({"district": [p["name"] for p in kd], "LP beyond 1 km (%)": [p["b_lp"] for p in kd]})
        st.altair_chart(bars(df, "LP beyond 1 km (%)", "district", colour="#184f95", sort=list(df.district)), width="stretch")
        st.caption("Hilly districts (Idukki, Wayanad) are far above the rest.")
    with g8:
        st.markdown("**Kerala priority areas for new schools**  \nConnected 100 m cells beyond the norm with at least 50 children")
        pts = pd.DataFrame(D["pts"])
        by = pts.groupby("level").agg(areas=("children", "size"), children=("children", "sum")).reindex(LEVELS["k"]).dropna()
        df = pd.DataFrame({"level": [f"{SHORT[l]} · {fmt(r.children, 0)} children" for l, r in by.iterrows()], "areas": by["areas"].values})
        st.altair_chart(bars(df, "areas", "level", colour="#e87ba4", fmt_x=",.0f", sort=list(df.level)), width="stretch")
        big5 = pts[pts.level == "lp"].nlargest(5, "children")
        df = pd.DataFrame({"area": [f"{r.local_body}, {r.district}" for r in big5.itertuples()], "children": big5["children"].values})
        st.altair_chart(bars(df, "children", "area", colour="#e87ba4", fmt_x=",.0f", sort=list(df.area)), width="stretch")
        st.caption("They show where the gap is, not the exact site: choosing a site needs land and cost data.")

    st.markdown("#### Detailed tables")
    for t in [k for k in D["panels"] if k != "Notes"]:
        with st.expander(t):
            st.markdown(D["panels"][t], unsafe_allow_html=True)

# ---------- About ----------
with tab_about:
    st.subheader("About the data")
    st.write("How the numbers on this site were made, in plain words.")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("""**How it was calculated**

1. **Children (demand):** WorldPop 2026 population in 100 m cells, split into school ages, times the share of pupils in Government + Aided schools (UDISE).
2. **Seats (supply):** teachers × the official pupils-per-teacher norm; higher secondary from the real HSCAP and VHSCAP batches. A check run uses the pupils enrolled now.
3. **Catchment:** a child may travel 1 km (primary), 3 km (upper primary), 5 km (secondary; 3 km in hilly Kerala) or 7 km (higher secondary).
4. **Nearer counts more:** a Gaussian weight, 1 at the school and 0 at the edge of the catchment.
5. **E2SFCA:** each school's seats are shared among the children around it; each cell adds up what it can reach. That is “seats per child”.
6. **Areas:** cells are added up into states, districts, sub-districts, taluks and local bodies, weighted by children.
7. **Patterns:** Gi* hot spots, K-Means area types, FP-Growth rules and Kerala priority areas.

*Computed with Apache Spark and Apache Sedona over 140 million populated cells and about 1.07 million schools.*""")
    with a2:
        st.markdown("**Data notes**")
        st.markdown(D["panels"]["Notes"], unsafe_allow_html=True)
