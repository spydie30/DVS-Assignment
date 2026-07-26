"""
VAHAN RTO Vehicle Registration Analytics
========================================
Streamlit + Altair dashboard over the cleaned VAHAN/DVS registration dataset
(3,700 registrations, 12 Indian states, 2018-2024).

Structure follows the refactor spec: three topic tabs -
    1. Macro Fuel Transition   (CFAR and market shifts)
    2. OEM & Powertrain Strategy
    3. Regulatory & Data Quality Audit

Metric definitions as implemented
---------------------------------
CFAR - Clean-Fuel Adoption Rate.
       100 * (clean-fuel vehicles / total). Clean fuels are Electric, CNG and
       Hybrid, which are the clean powertrains present in this dataset. The
       shipped `is_clean` column encodes exactly this.

FMI  - Fleet Modernization Index.
       100 * (BS6-compliant vehicles / total), i.e. the shipped `is_compliant`
       rate. NOTE: the refactor spec described FMI as a Herfindahl fuel-
       diversity score, but the FMI column in this dataset is the BS6
       compliance rate. Per project decision the dataset definition governs and
       the diversity index is not implemented.

`is_clean` semantics
--------------------
`is_clean` flags CLEAN FUEL (electric / CNG / hybrid). It is NOT a data-quality
flag. The dataset was cleaned before delivery, so analytics are not partitioned
on it; Tab 3's governance section instead audits genuine integrity defects
(RTO/state disagreement, electric vehicles carrying Bharat Stage norms, and
electric-only manufacturers recorded against fossil fuels).

Design compliance (Knaflic, 2015; Han, Kamber & Pei, 2012)
----------------------------------------------------------
Gestalt   - Proximity, Similarity (global colour maps), Enclosure (section
            containers), Closure (no gridlines/borders), Continuity (labelAngle
            fixed at 0), Connection (trends drawn as lines).
Hard bans - no pie/donut (a waterfall carries the share-shift story instead),
            no 3D, no secondary y-axes, zero baseline on every bar.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VAHAN RTO Registration Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

_HERE = Path(__file__).parent
DATA_PATH = next(
    (p for p in (_HERE / "cleaned_dvs_data_with_cfar_fmi.csv",
                 _HERE / "cleaned_dvs_data.csv") if p.exists()),
    _HERE / "cleaned_dvs_data_with_cfar_fmi.csv",
)

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
# Grey-first, one accent, colourblind-safe. Declared once and obeyed
# everywhere, which is what satisfies the Gestalt "Similarity" rule.
# ---------------------------------------------------------------------------

ACCENT = "#1F6FB2"        # single high-contrast accent (blue)
ACCENT_ALT = "#E08214"    # orange, for two-way contrast (never red/green)
GREY_DARK = "#54595F"
GREY_MID = "#8C9196"
GREY_LIGHT = "#C9CDD1"
GREY_FAINT = "#E8EAEC"
INK = "#25292E"

CLEAN_FUELS = ["Electric", "CNG", "Hybrid"]
FUEL_ORDER = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"]
FUEL_COLOR = {
    "Petrol": GREY_MID,
    "Diesel": GREY_DARK,
    "CNG": "#9CC3DE",
    "Hybrid": ACCENT_ALT,
    "Electric": ACCENT,
}
NORM_ORDER = ["BS3", "BS4", "BS6"]
NORM_COLOR = {"BS3": GREY_DARK, "BS4": GREY_MID, "BS6": ACCENT}
CATEGORY_ORDER = ["2W", "3W", "4W", "LCV", "HCV", "OTH"]
CATEGORY_PLAIN = {
    "2W": "Two-wheelers (scooters, motorcycles)",
    "3W": "Three-wheelers (auto-rickshaws)",
    "4W": "Cars and passenger vehicles",
    "LCV": "Light goods vehicles (vans, mini-trucks)",
    "HCV": "Heavy goods vehicles (trucks, buses)",
    "OTH": "Other vehicles (tractors, cranes)",
}

alt.data_transformers.disable_max_rows()


def base_theme():
    """Global Vega-Lite config: no gridlines, no borders, horizontal labels."""
    return {
        "config": {
            "view": {"strokeWidth": 0, "continuousHeight": 300},
            "axis": {
                "grid": False,
                "domainColor": GREY_LIGHT,
                "tickColor": GREY_LIGHT,
                "labelColor": GREY_DARK,
                "titleColor": GREY_DARK,
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
                "labelAngle": 0,          # Continuity: rotated text is banned
                "labelLimit": 400,
            },
            "legend": {
                "labelColor": GREY_DARK,
                "titleColor": GREY_DARK,
                "labelFontSize": 12,
                "titleFontSize": 12,
                "titleFontWeight": "normal",
                "symbolType": "square",
            },
            "title": {
                "color": INK,
                "fontSize": 15,
                "fontWeight": 600,
                "anchor": "start",
                "subtitleColor": GREY_MID,
                "subtitleFontSize": 12,
            },
            "range": {"category": [ACCENT, GREY_MID, "#9CC3DE", GREY_DARK,
                                   ACCENT_ALT, GREY_LIGHT]},
        }
    }


alt.themes.register("brief", base_theme)
alt.themes.enable("brief")


def chart(df_, title=None, subtitle=None):
    c = alt.Chart(df_)
    if title:
        c = c.properties(title=alt.TitleParams(
            text=title, subtitle=subtitle or "", anchor="start", align="left"))
    return c


def qx(field, title=None, fmt=None):
    """Quantitative X axis, zero baseline enforced, horizontal labels."""
    return alt.X(field, title=title,
                 scale=alt.Scale(zero=True, nice=True),
                 axis=alt.Axis(labelAngle=0, grid=False,
                               format=fmt if fmt is not None else alt.Undefined))


def qy(field, title=None, fmt=None):
    """Quantitative Y axis, zero baseline enforced."""
    return alt.Y(field, title=title,
                 scale=alt.Scale(zero=True, nice=True),
                 axis=alt.Axis(labelAngle=0, grid=False,
                               format=fmt if fmt is not None else alt.Undefined))


def section(title, caption=None):
    """Enclosure: light grey container grouping a section (Gestalt)."""
    st.markdown(
        f"""
        <div style="background:{GREY_FAINT};border-radius:8px;
                    padding:10px 14px;margin:6px 0 10px 0;">
          <div style="color:{INK};font-size:15px;font-weight:600;">{title}</div>
          {f'<div style="color:{GREY_MID};font-size:12.5px;margin-top:2px;">{caption}</div>' if caption else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(label, value, note=None, accent=True):
    """Simple text callout - preferred over tables for headline numbers."""
    color = ACCENT if accent else GREY_DARK
    st.markdown(
        f"""
        <div style="padding:4px 0 14px 0;">
          <div style="color:{GREY_MID};font-size:13px;">{label}</div>
          <div style="color:{color};font-size:36px;font-weight:700;
                      line-height:1.15;">{value}</div>
          {f'<div style="color:{GREY_MID};font-size:12.5px;">{note}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_int(n):
    """No trailing decimals - formatting legibility rule."""
    return f"{int(round(n)):,}"


def fmt_pct(x, dp=0):
    return f"{x * 100:.{dp}f}%"


def fmt_score(x, dp=1):
    return f"{x:.{dp}f}%"


def fmt_pp(x, dp=1):
    """Percentage-POINT difference; a gap between shares is not a % change."""
    return f"{abs(x) * 100:.{dp}f} percentage points"


FLAT_THRESHOLD_PP = 0.02


def describe_trend(values, noun, plain=False):
    """Derive an honest action title from a series instead of asserting one."""
    vals = list(values)
    if len(vals) < 2:
        return f"{noun} in the selected period", 0.0
    delta = vals[-1] - vals[0]
    steps = [b - a for a, b in zip(vals, vals[1:])]
    all_up, all_down = all(s > 0 for s in steps), all(s < 0 for s in steps)
    many = len(vals) >= 4
    if abs(delta) < FLAT_THRESHOLD_PP:
        return (f"{noun} has stayed about the same" if plain
                else f"{noun} has held roughly flat"), delta
    if delta > 0:
        if all_up and many:
            return (f"{noun} has gone up every year" if plain
                    else f"{noun} has risen every year"), delta
        if all_up:
            return (f"{noun} has gone up" if plain
                    else f"{noun} is up {fmt_pp(delta)}"), delta
        return (f"{noun} is higher than it was, but it has moved up and down" if plain
                else f"{noun} is up {fmt_pp(delta)}, though not steadily"), delta
    if all_down and many:
        return (f"{noun} has gone down every year" if plain
                else f"{noun} has fallen every year"), delta
    if all_down:
        return (f"{noun} has gone down" if plain
                else f"{noun} is down {fmt_pp(delta)}"), delta
    return (f"{noun} is lower than it was, but it has moved up and down" if plain
            else f"{noun} is down {fmt_pp(delta)}, though not steadily"), delta


# ---------------------------------------------------------------------------
# Data loading, derived fields and integrity flags
# ---------------------------------------------------------------------------

RTO_CODE_TO_STATE = {
    "KA": "Karnataka", "KL": "Kerala", "MP": "Madhya Pradesh", "PB": "Punjab",
    "TN": "Tamil Nadu", "DL": "Delhi", "RJ": "Rajasthan", "GJ": "Gujarat",
    "UP": "Uttar Pradesh", "WB": "West Bengal", "TS": "Telangana",
    "TG": "Telangana", "MH": "Maharashtra",
    # Chandigarh is a union territory: a valid RTO code that belongs to no
    # state in this 12-state dataset.
    "CH": "Chandigarh (UT)",
}
EV_ONLY_BRANDS = ["Ola Electric", "Ather"]


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Read the registration file and derive analysis + integrity columns."""
    try:                                   # PyArrow engine per spec
        df = pd.read_csv(path, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path)

    df["Registration_Date"] = pd.to_datetime(
        df["Registration_Date"], format="%d-%m-%Y", errors="coerce")
    df["Year_Month"] = df["Registration_Date"].dt.to_period("M").dt.to_timestamp()

    # "EV" is an acronym; spell it out once, globally.
    df["Fuel_Type"] = df["Fuel_Type"].replace({"EV": "Electric"})

    df["Category_Plain"] = df["Vehicle_Category"].map(CATEGORY_PLAIN)
    df["Is_Electric"] = df["Fuel_Type"].eq("Electric")

    # is_clean = CLEAN FUEL flag (not a data-quality flag).
    df["Is_Clean"] = (df["is_clean"].astype(bool) if "is_clean" in df.columns
                      else df["Fuel_Type"].isin(CLEAN_FUELS))
    # is_compliant = meets the current BS6 standard. FMI is its rate.
    df["Is_Compliant"] = (df["is_compliant"].astype(bool)
                          if "is_compliant" in df.columns
                          else df["Emission_Norm"].eq("BS6"))

    df["Emission_Norm_Display"] = np.where(
        df["Is_Electric"], "Not applicable (electric)", df["Emission_Norm"])

    df["RTO_Code"] = df["RTO_Office"].str.extract(r"\(([A-Z]{2})-")
    df["RTO_State"] = df["RTO_Code"].map(RTO_CODE_TO_STATE)

    # --- Genuine integrity defects (the dataset was pre-cleaned, so these are
    # the residual issues worth governing; `is_clean` is NOT one of them).
    df["QF_RTO_Mismatch"] = ~df["RTO_State"].eq(df["State"])
    df["QF_EV_Has_Norm"] = df["Is_Electric"] & df["Emission_Norm"].isin(NORM_ORDER)
    df["QF_EVBrand_Fossil"] = (df["Manufacturer_Brand"].isin(EV_ONLY_BRANDS)
                               & ~df["Is_Electric"])
    QF = ["QF_RTO_Mismatch", "QF_EV_Has_Norm", "QF_EVBrand_Fossil"]
    df["Has_Defect"] = df[QF].any(axis=1)
    df["Defect_Count"] = df[QF].sum(axis=1)

    def _reasons(r):
        out = []
        if r["QF_RTO_Mismatch"]:
            out.append("RTO office belongs to a different state")
        if r["QF_EV_Has_Norm"]:
            out.append("Electric vehicle carries a Bharat Stage norm")
        if r["QF_EVBrand_Fossil"]:
            out.append("Electric-only brand recorded as fossil fuel")
        return "; ".join(out)

    df["Defect_Reasons"] = df.apply(_reasons, axis=1)
    return df


QUALITY_RULES = [
    ("QF_RTO_Mismatch", "Consistency",
     "RTO office code matches the declared State",
     "State-to-RTO drill-down is unreliable for these records."),
    ("QF_EV_Has_Norm", "Validity",
     "Electric vehicles carry no Bharat Stage norm",
     "Bharat Stage rates tailpipe emissions; electric vehicles have none."),
    ("QF_EVBrand_Fossil", "Accuracy",
     "Electric-only manufacturers sell only electric vehicles",
     "Ola Electric and Ather produce electric vehicles exclusively."),
]


@st.cache_data(show_spinner=False)
def audit_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Data governance tracker over genuine defects (Han, Kamber & Pei, 2012)."""
    n = max(len(df), 1)
    rows = []
    for col, dim, check, detail in QUALITY_RULES:
        bad = int(df[col].sum())
        rows.append({"Dimension": dim, "Check": check, "Failing records": bad,
                     "Pass rate": 1 - bad / n, "Detail": detail})

    source_cols = [c for c in [
        "Registration_Number", "Registration_Date", "Registration_Year", "State",
        "RTO_Office", "Vehicle_Category", "Vehicle_Sub_Type", "Manufacturer_Brand",
        "Fuel_Type", "Emission_Norm", "Engine_CC", "Seating_Capacity",
        "Vehicle_Age_Years", "is_clean", "is_compliant", "CFAR", "FMI"]
        if c in df.columns]
    missing = int(df[source_cols].isna().sum().sum())
    rows.append({"Dimension": "Completeness",
                 "Check": "No missing values in any source field",
                 "Failing records": missing,
                 "Pass rate": 1 - missing / (n * max(len(source_cols), 1)),
                 "Detail": "All source fields populated across every record."})

    dupes = int(df["Registration_Number"].duplicated().sum())
    rows.append({"Dimension": "Uniqueness",
                 "Check": "Registration numbers are unique",
                 "Failing records": dupes, "Pass rate": 1 - dupes / n,
                 "Detail": "Primary key holds; no duplicate registrations."})
    return pd.DataFrame(rows)


def index_by(df: pd.DataFrame, *group_cols) -> pd.DataFrame:
    """CFAR and FMI recomputed live at an arbitrary grain, honouring filters."""
    return (df.groupby(list(group_cols), dropna=False)
              .agg(Registrations=("Registration_Number", "size"),
                   CFAR=("Is_Clean", lambda s: s.mean() * 100),
                   FMI=("Is_Compliant", lambda s: s.mean() * 100))
              .reset_index())


df_all = load_data(DATA_PATH)

# ---------------------------------------------------------------------------
# SIDEBAR - multi-level dynamic filters
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")

year_min = int(df_all["Registration_Year"].min())
year_max = int(df_all["Registration_Year"].max())
date_min = df_all["Registration_Date"].min().date()
date_max = df_all["Registration_Date"].max().date()

# Two time filters offered side by side: the year slider (fast, coarse) and an
# exact from/to calendar. A mode selector keeps them from silently fighting -
# only the chosen one is applied - while both remain available.
time_mode = st.sidebar.radio(
    "Filter time by", ["Year range", "Exact dates"], horizontal=True)

if time_mode == "Year range":
    year_range = st.sidebar.slider(
        "Registration year", year_min, year_max, (year_min, year_max), step=1)
    start_date = pd.Timestamp(year_range[0], 1, 1)
    end_date = pd.Timestamp(year_range[1], 12, 31)
else:
    # Two independent single-date calendars rather than one range picker.
    # A range picker re-runs the app the moment the FIRST date is clicked,
    # leaving the range half-defined, and it forces both ends to be re-picked
    # just to nudge one of them. Separate From/To fields keep each calendar
    # self-contained: picking a date commits only that end, and the other side
    # is untouched.
    d_from, d_to = st.sidebar.columns(2)
    with d_from:
        d0 = st.date_input(
            "From", value=date_min,
            min_value=date_min, max_value=date_max,
            format="YYYY-MM-DD", key="date_from",
            help="Start of the window (inclusive).")
    with d_to:
        d1 = st.date_input(
            "To", value=date_max,
            min_value=date_min, max_value=date_max,
            format="YYYY-MM-DD", key="date_to",
            help="End of the window (inclusive).")

    # Bounds are deliberately NOT cross-linked: constraining "To" by "From"
    # would change that widget's identity and silently reset it. Instead an
    # inverted pair is swapped and reported.
    swapped = d0 > d1
    if swapped:
        d0, d1 = d1, d0

    start_date = pd.Timestamp(d0)
    end_date = pd.Timestamp(d1) + pd.Timedelta(hours=23, minutes=59)
    year_range = (start_date.year, end_date.year)

    if swapped:
        st.sidebar.warning(
            f"'From' was later than 'To', so the dates were swapped. "
            f"Showing {d0} to {d1}.")
    span_days = (d1 - d0).days + 1
    st.sidebar.caption(
        f"Showing {d0} to {d1} (inclusive) · {span_days:,} day"
        f"{'s' if span_days != 1 else ''}.")

# State -> RTO office (RTO options depend on the states chosen)
all_states = sorted(df_all["State"].unique())
states = st.sidebar.multiselect("State", all_states, default=all_states)
_state_pool = df_all[df_all["State"].isin(states)] if states else df_all
rto_options = sorted(_state_pool["RTO_Office"].unique())
rtos = st.sidebar.multiselect(
    "RTO office", rto_options, default=[],
    help="Leave empty to include every RTO office in the selected states.")

# Category -> sub-type (sub-type options depend on the categories chosen)
cats_present = [c for c in CATEGORY_ORDER if c in set(df_all["Vehicle_Category"])]
categories = st.sidebar.multiselect("Vehicle category", cats_present,
                                    default=cats_present)
_cat_pool = (df_all[df_all["Vehicle_Category"].isin(categories)]
             if categories else df_all)
sub_options = sorted(_cat_pool["Vehicle_Sub_Type"].unique())
sub_types = st.sidebar.multiselect(
    "Vehicle sub-type", sub_options, default=[],
    help="Leave empty to include every sub-type in the selected categories.")

brand_options = sorted(df_all["Manufacturer_Brand"].unique())
brands = st.sidebar.multiselect(
    "Manufacturer (OEM)", brand_options, default=[],
    help="Leave empty to include every manufacturer.")

st.sidebar.divider()
# Global data-quality toggle. `is_clean` is a fuel flag, so it cannot drive
# this; the toggle acts on genuine integrity defects instead.
exclude_defects = st.sidebar.checkbox(
    "Exclude records failing data-quality checks", value=False,
    help="Removes rows with an RTO/state disagreement, an electric vehicle "
         "carrying a Bharat Stage norm, or an electric-only brand recorded "
         "against fossil fuel. Off by default so headline totals match the "
         "delivered dataset.")

mask = (
    df_all["Registration_Date"].between(start_date, end_date)
    & df_all["State"].isin(states)
    & df_all["Vehicle_Category"].isin(categories)
)
if rtos:
    mask &= df_all["RTO_Office"].isin(rtos)
if sub_types:
    mask &= df_all["Vehicle_Sub_Type"].isin(sub_types)
if brands:
    mask &= df_all["Manufacturer_Brand"].isin(brands)
if exclude_defects:
    mask &= ~df_all["Has_Defect"]

df = df_all[mask].copy()

st.sidebar.divider()
st.sidebar.metric("Records in selection", fmt_int(len(df)),
                  f"of {fmt_int(len(df_all))} total")
st.sidebar.caption(
    "CFAR = clean-fuel adoption rate. FMI = fleet modernization index "
    "(share meeting the current BS6 standard)."
)

# ---------------------------------------------------------------------------
# Header + empty-state handling
# ---------------------------------------------------------------------------

st.title("VAHAN RTO Registration Analytics")
st.caption(
    f"{fmt_int(len(df_all))} registrations · {df_all['State'].nunique()} states · "
    f"{year_min} to {year_max}")

if df.empty:
    st.warning("No data available for the selected filters. "
               "Widen your selection in the sidebar.")
    st.stop()

tab_macro, tab_oem, tab_audit = st.tabs(
    ["📊 Macro Fuel Transition",
     "🏎️ OEM & Powertrain Strategy",
     "🚨 Regulatory & Data Quality Audit"])

# ===========================================================================
# TAB 1 - MACRO FUEL TRANSITION
# ===========================================================================

with tab_macro:
    cfar = df["Is_Clean"].mean() * 100
    fmi = df["Is_Compliant"].mean() * 100
    non_compliant = (~df["Is_Compliant"]).mean() * 100

    vol_by_year = (df.groupby("Registration_Year")
                     .size().reset_index(name="Registrations"))
    yoy_delta = None
    if len(vol_by_year) >= 2:
        prev, last = vol_by_year["Registrations"].iloc[-2], vol_by_year["Registrations"].iloc[-1]
        if prev:
            yoy_delta = (last - prev) / prev * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        callout("Total registrations", fmt_int(len(df)),
                (f"{yoy_delta:+.1f}% vs previous year" if yoy_delta is not None
                 else "Single year selected"))
    with k2:
        callout("Clean-Fuel Adoption Rate (CFAR)", fmt_score(cfar),
                "Electric, CNG or hybrid")
    with k3:
        callout("Fleet Modernization Index (FMI)", fmt_score(fmi),
                "Meets the current BS6 standard", accent=False)
    with k4:
        callout("Non-compliant fleet share", fmt_score(non_compliant),
                "The exact complement of FMI", accent=False)

    st.divider()

    # --- Chart 1: fuel share trajectory (100% stacked) ---------------------
    fuel_year = (df.groupby(["Registration_Year", "Fuel_Type"])
                   .size().reset_index(name="Registrations"))
    clean_by_year = df.groupby("Registration_Year")["Is_Clean"].mean()
    macro_title, macro_delta = describe_trend(clean_by_year, "Clean-fuel adoption")

    section(macro_title,
            "Fuel mix as a share of registrations each year. Connection: bands "
            "are continuous, so each fuel reads as one trajectory.")
    fuels_here = [f for f in FUEL_ORDER if f in set(fuel_year["Fuel_Type"])]
    area = (
        chart(fuel_year)
        .mark_area()
        .encode(
            x=alt.X("Registration_Year:O", title="Registration year",
                    axis=alt.Axis(labelAngle=0, grid=False)),
            y=alt.Y("Registrations:Q", stack="normalize",
                    title="Share of registrations",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(labelAngle=0, grid=False, format="%")),
            color=alt.Color("Fuel_Type:N", title="Fuel",
                            scale=alt.Scale(domain=fuels_here,
                                            range=[FUEL_COLOR[f] for f in fuels_here]),
                            sort=fuels_here),
            order=alt.Order("color_Fuel_Type_sort_index:Q"),
            tooltip=[alt.Tooltip("Registration_Year:O", title="Year"),
                     alt.Tooltip("Fuel_Type:N", title="Fuel"),
                     alt.Tooltip("Registrations:Q", format=",")],
        ).properties(height=330)
    )
    st.altair_chart(area, width="stretch")

    st.divider()

    c1, c2 = st.columns([1, 1])

    # --- Chart 2: CFAR ranking by state (or RTO office) --------------------
    with c1:
        rank_dim = st.radio("Rank clean-fuel adoption by",
                            ["State", "RTO office"], horizontal=True)
        dim_col = "State" if rank_dim == "State" else "RTO_Office"
        ranked = index_by(df, dim_col).sort_values("CFAR", ascending=False)
        if rank_dim == "RTO office":
            ranked = ranked.head(20)
        best = ranked.iloc[0][dim_col] if len(ranked) else None
        ranked["Highlight"] = ranked[dim_col].eq(best)

        section(f"Clean-fuel adoption hotspots by {rank_dim.lower()}",
                "Preattentive: only the leading row carries the accent colour.")
        # Positional-only base. Text layers must NOT inherit the colour
        # condition, or unhighlighted labels render in near-invisible grey.
        rank_base = chart(ranked).encode(
            x=qx("CFAR:Q", "CFAR (% clean fuel)"),
            y=alt.Y(f"{dim_col}:N", sort="-x", title=None,
                    axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260)),
        )
        bars = rank_base.mark_bar(cornerRadiusEnd=3).encode(
            color=alt.condition(alt.datum.Highlight,
                                alt.value(ACCENT), alt.value(GREY_LIGHT)),
            tooltip=[alt.Tooltip(f"{dim_col}:N"),
                     alt.Tooltip("CFAR:Q", title="CFAR %", format=".1f"),
                     alt.Tooltip("FMI:Q", title="FMI %", format=".1f"),
                     alt.Tooltip("Registrations:Q", format=",")],
        )
        lbl = rank_base.mark_text(align="left", dx=4, fontSize=11,
                                  color=GREY_DARK).encode(
            text=alt.Text("CFAR:Q", format=".1f"))
        st.altair_chart(
            (bars + lbl).properties(height=max(260, min(560, len(ranked) * 26))),
            width="stretch")

    # --- Chart 3: waterfall of net share shift (replaces banned donut) -----
    with c2:
        yrs = sorted(df["Registration_Year"].unique())
        section("Net fuel share shift, in basis points",
                "A waterfall, not a donut: angles cannot be compared "
                "quantitatively, so shifts are shown on a common baseline.")
        if len(yrs) < 2:
            st.info("Select at least two years to measure a share shift.")
        else:
            y0, y1 = int(yrs[0]), int(yrs[-1])
            s0 = df[df["Registration_Year"] == y0]["Fuel_Type"].value_counts(normalize=True)
            s1 = df[df["Registration_Year"] == y1]["Fuel_Type"].value_counts(normalize=True)
            allf = [f for f in FUEL_ORDER if f in set(s0.index) | set(s1.index)]
            shift = pd.DataFrame({
                "Fuel": allf,
                "Shift_bps": [(s1.get(f, 0) - s0.get(f, 0)) * 10000 for f in allf],
            }).sort_values("Shift_bps")
            # Floating bars: each starts where the previous ended.
            shift["End"] = shift["Shift_bps"].cumsum()
            shift["Start"] = shift["End"] - shift["Shift_bps"]
            shift["Direction"] = np.where(shift["Shift_bps"] >= 0, "Gain", "Loss")

            wf = (
                chart(shift)
                .mark_bar(cornerRadius=2, size=26)
                .encode(
                    x=alt.X("Fuel:N", sort=list(shift["Fuel"]), title=None,
                            axis=alt.Axis(labelAngle=0, grid=False)),
                    y=alt.Y("Start:Q", title=f"Share shift {y0} to {y1} (bps)",
                            scale=alt.Scale(zero=True),
                            axis=alt.Axis(labelAngle=0, grid=False)),
                    y2="End:Q",
                    color=alt.Color("Direction:N", title=None,
                                    scale=alt.Scale(domain=["Gain", "Loss"],
                                                    range=[ACCENT, ACCENT_ALT]),
                                    legend=alt.Legend(orient="top")),
                    tooltip=[alt.Tooltip("Fuel:N"),
                             alt.Tooltip("Shift_bps:Q", title="Shift (bps)",
                                         format="+,.0f")],
                ).properties(height=max(260, min(560, len(ranked) * 26)))
            )
            zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                color=GREY_MID).encode(y="y:Q")
            st.altair_chart(wf + zero_rule, width="stretch")
            st.caption(
                f"100 basis points = 1 percentage point. Gains and losses sum to "
                f"zero because these are shares of the same total. Comparing "
                f"{y0} with {y1} only; intermediate years are not shown."
            )

# ===========================================================================
# TAB 2 - OEM & POWERTRAIN STRATEGY
# ===========================================================================

with tab_oem:
    oem = index_by(df, "Manufacturer_Brand")
    oem = oem.rename(columns={"Manufacturer_Brand": "Brand"})

    if oem.empty:
        st.warning("No data available for the selected filters.")
    else:
        top_vol = oem.loc[oem["Registrations"].idxmax()]
        top_cfar = oem.loc[oem["CFAR"].idxmax()]
        top_fmi = oem.loc[oem["FMI"].idxmax()]

        k1, k2, k3 = st.columns(3)
        with k1:
            callout("Top OEM by volume", str(top_vol["Brand"]),
                    f"{fmt_int(top_vol['Registrations'])} registrations")
        with k2:
            callout("Highest clean-fuel OEM", str(top_cfar["Brand"]),
                    f"CFAR {top_cfar['CFAR']:.1f}%")
        with k3:
            # The spec called this "most diversified (highest FMI)", but FMI in
            # this dataset is BS6 compliance, not a diversity score.
            callout("Most BS6-compliant OEM", str(top_fmi["Brand"]),
                    f"FMI {top_fmi['FMI']:.1f}%", accent=False)

        st.divider()

        # --- Chart 1: CFAR vs FMI quadrant scatter -------------------------
        section("Where each manufacturer sits on both indices",
                "Bubble size is registration volume. Dashed lines split each "
                "axis at its mean.")
        cfar_spread = oem["CFAR"].max() - oem["CFAR"].min()
        fmi_spread = oem["FMI"].max() - oem["FMI"].min()

        if len(oem) < 2 or cfar_spread < 1 or fmi_spread < 1:
            st.info(
                "Too few manufacturers, or too little spread between them, for a "
                "two-dimensional comparison. Widen the OEM or year filters.")
        else:
            x_mid, y_mid = oem["CFAR"].mean(), oem["FMI"].mean()
            # Quadrant names describe the actual axes (clean fuel x BS6
            # compliance), not the diversity framing in the original spec.
            oem["Quadrant"] = np.where(
                oem["CFAR"] >= x_mid,
                np.where(oem["FMI"] >= y_mid, "Green pioneers", "Clean-fuel specialists"),
                np.where(oem["FMI"] >= y_mid, "Compliance leaders", "Fossil dependent"))

            rule_v = alt.Chart(pd.DataFrame({"x": [x_mid]})).mark_rule(
                color=GREY_LIGHT, strokeDash=[4, 4]).encode(x="x:Q")
            rule_h = alt.Chart(pd.DataFrame({"y": [y_mid]})).mark_rule(
                color=GREY_LIGHT, strokeDash=[4, 4]).encode(y="y:Q")
            quad_domain = ["Green pioneers", "Compliance leaders",
                           "Clean-fuel specialists", "Fossil dependent"]
            quad_range = [ACCENT, "#9CC3DE", ACCENT_ALT, GREY_MID]

            # Labelling every OEM makes the plot unreadable, so name only the
            # largest by volume unless the user asks for all of them.
            max_lab = min(12, len(oem))
            label_all = st.checkbox(
                "Label every manufacturer", value=False,
                help="Off by default: only the largest manufacturers are named, "
                     "so the labels stay legible. Hover any bubble for details.")
            lab_cut = (oem["Registrations"].min() if label_all
                       else oem["Registrations"].nlargest(max_lab).iloc[-1])
            oem["Label"] = np.where(oem["Registrations"] >= lab_cut, oem["Brand"], "")

            # Positional-only base. The text layer MUST NOT inherit the `size`
            # encoding: on a text mark `size` sets the font size, so bubble
            # scaling (60-900) would render labels hundreds of points tall.
            scatter_base = chart(oem).encode(
                # Relationship scatter: axes zoom to the data so the cloud is
                # visible. The zero-baseline rule governs bar heights.
                x=alt.X("CFAR:Q", title="CFAR (% clean fuel)",
                        scale=alt.Scale(zero=False, nice=True, padding=14),
                        axis=alt.Axis(labelAngle=0, grid=False)),
                y=alt.Y("FMI:Q", title="FMI (% BS6 compliant)",
                        scale=alt.Scale(zero=False, nice=True, padding=14),
                        axis=alt.Axis(labelAngle=0, grid=False)),
            )
            pts = scatter_base.mark_circle(opacity=0.75).encode(
                size=alt.Size("Registrations:Q", title="Volume",
                              scale=alt.Scale(range=[60, 900])),
                color=alt.Color("Quadrant:N", title=None,
                                scale=alt.Scale(domain=quad_domain,
                                                range=quad_range),
                                legend=alt.Legend(orient="top", columns=2)),
                tooltip=[alt.Tooltip("Brand:N", title="Manufacturer"),
                         alt.Tooltip("Quadrant:N"),
                         alt.Tooltip("CFAR:Q", title="CFAR %", format=".1f"),
                         alt.Tooltip("FMI:Q", title="FMI %", format=".1f"),
                         alt.Tooltip("Registrations:Q", format=",")],
            )
            labels = scatter_base.mark_text(
                dx=10, dy=-8, align="left", fontSize=11, color=INK,
            ).encode(text=alt.Text("Label:N"))

            st.altair_chart((rule_v + rule_h + pts + labels).properties(height=470),
                            width="stretch")
            st.caption(
                "Green pioneers lead on both axes. Compliance leaders meet BS6 "
                "but still sell fossil powertrains. Clean-fuel specialists sell "
                "clean fuel yet carry older norms on the rest of their range. "
                + ("All manufacturers are labelled."
                   if label_all else
                   f"Only the {max_lab} largest manufacturers are labelled; "
                   "hover any bubble for the rest."))

        st.divider()

        # --- Chart 2: fuel mix per brand (100% stacked horizontal) ---------
        # A slider needs min < max, so only offer one when the selection holds
        # enough manufacturers to be worth narrowing.
        n_brands = len(oem)
        if n_brands <= 5:
            n_show = n_brands
            st.caption(f"Showing all {n_brands} manufacturer"
                       f"{'s' if n_brands != 1 else ''} in the current selection.")
        else:
            n_show = st.slider("Manufacturers to compare (by volume)",
                               5, int(min(30, n_brands)), int(min(15, n_brands)))
        keep = oem.nlargest(n_show, "Registrations")["Brand"].tolist()
        mix = (df[df["Manufacturer_Brand"].isin(keep)]
               .groupby(["Manufacturer_Brand", "Fuel_Type"])
               .size().reset_index(name="Registrations"))
        fuels_mix = [f for f in FUEL_ORDER if f in set(mix["Fuel_Type"])]

        section("Fuel mix within each manufacturer",
                "Shares within each bar; bars are ordered by total volume.")
        stacked = (
            chart(mix)
            .mark_bar()
            .encode(
                x=alt.X("Registrations:Q", stack="normalize", title="Share of the OEM's registrations",
                        scale=alt.Scale(zero=True),
                        axis=alt.Axis(labelAngle=0, grid=False, format="%")),
                y=alt.Y("Manufacturer_Brand:N", sort=keep, title=None,
                        axis=alt.Axis(labelAngle=0, grid=False, labelLimit=220)),
                color=alt.Color("Fuel_Type:N", title="Fuel",
                                scale=alt.Scale(domain=fuels_mix,
                                                range=[FUEL_COLOR[f] for f in fuels_mix]),
                                sort=fuels_mix),
                order=alt.Order("color_Fuel_Type_sort_index:Q"),
                tooltip=[alt.Tooltip("Manufacturer_Brand:N", title="OEM"),
                         alt.Tooltip("Fuel_Type:N", title="Fuel"),
                         alt.Tooltip("Registrations:Q", format=",")],
            ).properties(height=max(280, n_show * 26))
        )
        st.altair_chart(stacked, width="stretch")

        st.divider()

        # --- Chart 3: Engine_CC distribution -------------------------------
        section("Engine displacement by sub-type and fuel",
                "Boxes span the interquartile range; whiskers extend to 1.5x IQR.")
        drop_ev = st.checkbox(
            "Exclude electric vehicles (they report 0 cc by definition)",
            value=True,
            help="Electric vehicles have no combustion displacement; keeping "
                 "them as zeros drags every box downward.")
        cc = df[~df["Is_Electric"]] if drop_ev else df
        if cc.empty:
            st.info("No records left after that exclusion.")
        else:
            split = st.radio("Break down by", ["Vehicle sub-type", "Fuel type"],
                             horizontal=True)
            if split == "Vehicle sub-type":
                order = (cc.groupby("Vehicle_Sub_Type")["Engine_CC"].median()
                           .sort_values().index.tolist())
                box = (
                    chart(cc)
                    .mark_boxplot(extent=1.5, size=18,
                                  median={"color": INK},
                                  outliers={"color": ACCENT_ALT, "size": 14})
                    .encode(
                        y=alt.Y("Vehicle_Sub_Type:N", sort=order, title=None,
                                axis=alt.Axis(labelAngle=0, grid=False, labelLimit=220)),
                        x=qx("Engine_CC:Q", "Engine displacement (cc)"),
                        color=alt.value(GREY_LIGHT),
                    ).properties(height=max(320, cc["Vehicle_Sub_Type"].nunique() * 26))
                )
            else:
                fuels_cc = [f for f in FUEL_ORDER if f in set(cc["Fuel_Type"])]
                box = (
                    chart(cc)
                    .mark_boxplot(extent=1.5, size=30,
                                  median={"color": INK},
                                  outliers={"color": ACCENT_ALT, "size": 14})
                    .encode(
                        x=alt.X("Fuel_Type:N", sort=fuels_cc, title=None,
                                axis=alt.Axis(labelAngle=0, grid=False)),
                        y=qy("Engine_CC:Q", "Engine displacement (cc)"),
                        color=alt.Color("Fuel_Type:N", legend=None,
                                        scale=alt.Scale(domain=fuels_cc,
                                                        range=[FUEL_COLOR[f] for f in fuels_cc])),
                    ).properties(height=340)
                )
            st.altair_chart(box, width="stretch")

# ===========================================================================
# TAB 3 - REGULATORY & DATA QUALITY AUDIT
# ===========================================================================

with tab_audit:
    sec_a, sec_b = st.tabs(["Section A · Compliance & scrappage risk",
                            "Section B · Data governance"])

    # ---------------- Section A: non-compliant fleet -----------------------
    with sec_a:
        risk = df[~df["Is_Compliant"]].copy()

        k1, k2, k3 = st.columns(3)
        with k1:
            callout("Non-compliant vehicles", fmt_int(len(risk)),
                    f"{(len(risk) / len(df) * 100):.1f}% of the selection")
        with k2:
            callout("Fleet Modernization Index", fmt_score(df["Is_Compliant"].mean() * 100),
                    "Share meeting BS6", accent=False)
        with k3:
            oldest = int(risk["Vehicle_Age_Years"].max()) if len(risk) else 0
            callout("Oldest non-compliant vehicle", f"{oldest} yrs",
                    "Scrappage exposure grows with age", accent=False)

        if risk.empty:
            st.success("Every vehicle in the current selection meets the BS6 standard.")
        else:
            section("Ageing non-compliant fleet: age against emission norm",
                    "Darker cells hold more vehicles. Older cohorts on the "
                    "oldest norms carry the greatest scrappage exposure.")
            grid = (risk.groupby(["Vehicle_Age_Years", "Emission_Norm"])
                        .size().reset_index(name="Vehicles"))
            norms_here = [n for n in NORM_ORDER if n in set(grid["Emission_Norm"])]
            # Positional-only base so the text layer does not inherit the blues
            # colour scale, which would tint the numbers and bury them in the
            # darker cells.
            hm_base = chart(grid).encode(
                x=alt.X("Emission_Norm:N", sort=norms_here, title="Emission norm",
                        axis=alt.Axis(labelAngle=0, grid=False)),
                y=alt.Y("Vehicle_Age_Years:O", title="Vehicle age (years)",
                        axis=alt.Axis(labelAngle=0, grid=False)),
            )
            hm = hm_base.mark_rect().encode(
                color=alt.Color("Vehicles:Q", title="Vehicles",
                                scale=alt.Scale(scheme="blues")),
                tooltip=[alt.Tooltip("Vehicle_Age_Years:O", title="Age (yrs)"),
                         alt.Tooltip("Emission_Norm:N", title="Norm"),
                         alt.Tooltip("Vehicles:Q", format=",")],
            )
            # Flip label colour on dark cells so the count stays readable.
            cutoff = grid["Vehicles"].max() * 0.6
            txt = hm_base.mark_text(fontSize=11).encode(
                text=alt.Text("Vehicles:Q", format=","),
                color=alt.condition(alt.datum.Vehicles > cutoff,
                                    alt.value("white"), alt.value(INK)),
            )
            st.altair_chart((hm + txt).properties(height=320), width="stretch")

            section("Risk fleet by RTO office")
            by_rto = (risk.groupby("RTO_Office")
                          .agg(Non_compliant=("Registration_Number", "size"),
                               Mean_age=("Vehicle_Age_Years", "mean"))
                          .reset_index()
                          .sort_values("Non_compliant", ascending=False))
            by_rto["Mean_age"] = by_rto["Mean_age"].round(1)
            top_rto = by_rto.head(20).copy()
            top_rto["Highlight"] = top_rto["RTO_Office"].eq(top_rto.iloc[0]["RTO_Office"])
            rbar = (
                chart(top_rto)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=qx("Non_compliant:Q", "Non-compliant vehicles"),
                    y=alt.Y("RTO_Office:N", sort="-x", title=None,
                            axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260)),
                    color=alt.condition(alt.datum.Highlight,
                                        alt.value(ACCENT), alt.value(GREY_LIGHT)),
                    tooltip=[alt.Tooltip("RTO_Office:N", title="RTO office"),
                             alt.Tooltip("Non_compliant:Q", format=","),
                             alt.Tooltip("Mean_age:Q", title="Mean age (yrs)",
                                         format=".1f")],
                ).properties(height=max(280, len(top_rto) * 26))
            )
            st.altair_chart(rbar, width="stretch")

            rto_pick = st.multiselect(
                "Filter the risk table by RTO office", sorted(risk["RTO_Office"].unique()),
                default=[])
            risk_view = risk[risk["RTO_Office"].isin(rto_pick)] if rto_pick else risk
            risk_cols = ["Registration_Number", "Registration_Date", "State",
                         "RTO_Office", "Vehicle_Category", "Vehicle_Sub_Type",
                         "Manufacturer_Brand", "Fuel_Type", "Emission_Norm",
                         "Vehicle_Age_Years", "Engine_CC"]
            st.dataframe(risk_view[risk_cols].sort_values("Vehicle_Age_Years",
                                                          ascending=False),
                         width="stretch", height=340, hide_index=True)
            st.download_button(
                "Download risk fleet as CSV",
                data=risk_view[risk_cols].to_csv(index=False).encode("utf-8"),
                file_name="non_compliant_risk_fleet.csv", mime="text/csv")

    # ---------------- Section B: data governance ---------------------------
    with sec_b:
        quality = audit_quality(df)
        defects = df[df["Has_Defect"]]
        cleanliness = (1 - len(defects) / len(df)) * 100

        st.caption(
            "This dataset was cleaned before delivery, so `is_clean` marks clean "
            "FUEL (electric, CNG, hybrid) rather than record quality. Governance "
            "below therefore audits genuine integrity defects instead of "
            "partitioning on that flag.")

        k1, k2, k3 = st.columns(3)
        with k1:
            callout("Data cleanliness score", fmt_score(cleanliness),
                    "Records passing every integrity check")
        with k2:
            callout("Records with a defect", fmt_int(len(defects)),
                    f"of {fmt_int(len(df))} in selection", accent=False)
        with k3:
            failing = int((quality["Failing records"] > 0).sum())
            callout("Checks failing", f"{failing} of {len(quality)}",
                    accent=(failing == 0))

        section("Integrity checks")
        qv = quality.copy()
        qv["Status"] = np.where(qv["Failing records"] == 0, "Pass", "Fail")
        qbar = (
            chart(qv)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=qx("Pass rate:Q", "Share of records passing", fmt="%"),
                y=alt.Y("Check:N", sort="x", title=None,
                        axis=alt.Axis(labelAngle=0, grid=False, labelLimit=420)),
                color=alt.condition(alt.datum.Status == "Fail",
                                    alt.value(ACCENT_ALT), alt.value(GREY_LIGHT)),
                tooltip=[alt.Tooltip("Dimension:N"), alt.Tooltip("Check:N"),
                         alt.Tooltip("Failing records:Q", format=","),
                         alt.Tooltip("Pass rate:Q", format=".1%")],
            ).properties(height=240)
        )
        st.altair_chart(qbar, width="stretch")
        st.dataframe(
            quality.assign(**{"Pass rate": (quality["Pass rate"] * 100).round(1)}),
            width="stretch", hide_index=True,
            column_config={
                "Pass rate": st.column_config.NumberColumn("Pass rate", format="%.1f%%"),
                "Failing records": st.column_config.NumberColumn(format="%d")})

        section("RTO offices ranked by error rate",
                "Share of each office's records failing at least one check.")
        rto_err = (df.groupby("RTO_Office")
                     .agg(Records=("Registration_Number", "size"),
                          Defects=("Has_Defect", "sum"))
                     .reset_index())
        rto_err["Error_rate"] = rto_err["Defects"] / rto_err["Records"] * 100
        rto_err = rto_err[rto_err["Records"] >= 5].sort_values(
            "Error_rate", ascending=False).head(20)
        if rto_err.empty:
            st.info("No RTO office has enough records to rank reliably.")
        else:
            rto_err["Highlight"] = rto_err["Error_rate"].eq(rto_err["Error_rate"].max())
            ebar = (
                chart(rto_err)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=qx("Error_rate:Q", "Records failing a check (%)"),
                    y=alt.Y("RTO_Office:N", sort="-x", title=None,
                            axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260)),
                    color=alt.condition(alt.datum.Highlight,
                                        alt.value(ACCENT_ALT), alt.value(GREY_LIGHT)),
                    tooltip=[alt.Tooltip("RTO_Office:N", title="RTO office"),
                             alt.Tooltip("Error_rate:Q", title="Error rate %",
                                         format=".1f"),
                             alt.Tooltip("Defects:Q", format=","),
                             alt.Tooltip("Records:Q", format=",")],
                ).properties(height=max(280, len(rto_err) * 26))
            )
            st.altair_chart(ebar, width="stretch")
            st.caption("Offices with fewer than 5 records are excluded, since a "
                       "single bad row would dominate their rate.")

        section("Data hygiene drill-down",
                "Every record failing at least one integrity check, with the "
                "reason attached, for audit export.")
        reason_opts = ["RTO office belongs to a different state",
                       "Electric vehicle carries a Bharat Stage norm",
                       "Electric-only brand recorded as fossil fuel"]
        picked = st.multiselect("Filter by defect", reason_opts, default=[])
        view = defects
        if picked:
            pattern = "|".join(pd.Series(picked).str.replace(
                r"([().\[\]*+?^$|\\])", r"\\\1", regex=True))
            view = defects[defects["Defect_Reasons"].str.contains(pattern, regex=True)]

        if view.empty:
            st.success("No records fail any integrity check in this selection.")
        else:
            drill_cols = ["Registration_Number", "State", "RTO_Office",
                          "Vehicle_Category", "Vehicle_Sub_Type",
                          "Manufacturer_Brand", "Fuel_Type", "Emission_Norm",
                          "Engine_CC", "Seating_Capacity", "Defect_Reasons"]
            st.dataframe(view[drill_cols], width="stretch", height=380,
                         hide_index=True)
            st.download_button(
                "Download flagged records as CSV",
                data=view[drill_cols].to_csv(index=False).encode("utf-8"),
                file_name="data_quality_exceptions.csv", mime="text/csv")
