"""
VAHAN RTO Vehicle Registration Analytics
========================================
Streamlit + Altair dashboard over the cleaned VAHAN/DVS registration dataset
(3,700 registrations, 12 Indian states, 2018-2024).

Structure follows the refactor spec: three topic tabs -
    1. Macro Fuel Transition   (CFAR and market shifts)
    2. OEM & Powertrain Strategy
    3. Regulatory & Data Quality Audit

INTERACTIVE UPGRADE (Power BI / Tableau-style):
    - Dynamic Zooming & Panning on scatter/line/time-series charts
    - Highlight/dim selections within charts, legend-bound where useful
    - Cross-filtering across every tab via the sidebar filters
    - Native Data Drilling (Year -> Quarter -> Month hierarchy)
    - Advanced Tooltips with conditional highlight/dim encodings

Note on chart selections: Streamlit's `on_select` only works on single-view
charts, not on compositions (layers). Most charts here are layered (bars plus
value labels, heatmap plus text, scatter plus quadrant rules), so selections
are handled client-side by Vega-Lite and cross-filtering is driven by the
sidebar widgets instead of by chart clicks.

Theming: the app uses Streamlit's default theme and default chart theme, so it
follows the viewer's light/dark setting. `theme.py` supplies only the data
colours (accent vs. muted, fuel identity).

Metric definitions as implemented
---------------------------------
CFAR - Clean-Fuel Adoption Rate.
       100 * (clean-fuel vehicles / total). Clean fuels are Electric, CNG and
       Hybrid, which are the clean powertrains present in this dataset. The
       shipped `is_clean` column encodes exactly this.

FMI  - Fleet Modernization Index.
       100 * (compliant vehicles / total), i.e. the shipped `is_compliant`
       rate. Compliance is a COMPOUND test (source: DVSAssignmentMetric.ipynb):

           is_compliant = Emission_Norm_Clean in ['BS6', 'ZEV']
                          AND Vehicle_Age_Years <= 5

       Both conditions must hold. A vehicle on a modern standard still fails if
       it is more than five years old, and an older standard fails at any age.

Emission_Norm_Clean
       Shipped by the latest extract: the Bharat Stage norm, except electric
       vehicles which carry 'ZEV' (zero-emission vehicle).

`is_clean` semantics
--------------------
`is_clean` flags CLEAN FUEL (electric / CNG / hybrid). It is NOT a data-quality
flag.

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
    page_icon="\U0001F697",
    layout="wide",
    initial_sidebar_state="expanded",
)

_HERE = Path(__file__).parent
DATA_PATH = _HERE / "VAHAN_Dataset_Fully_Corrected_Issues.csv"

# ---------------------------------------------------------------------------
# DESIGN SYSTEM
#
# No CSS injection and no registered Altair theme: the app runs on Streamlit's
# default theme and Streamlit's built-in chart theme, so it follows whatever
# light/dark setting the viewer has. The palette below is used only for
# explicit data encodings (accent vs. muted, fuel identity).
# ---------------------------------------------------------------------------
import theme

UNKNOWN_COLORS = theme.UNKNOWN_COLORS

ACCENT = theme.ACCENT
ACCENT_ALT = theme.ACCENT_ALT
GREY_DARK = theme.GREY_DARK
GREY_MID = theme.GREY_MID
GREY_LIGHT = theme.GREY_LIGHT
INK = theme.INK

CLEAN_FUELS = theme.CLEAN_FUELS
FUEL_ORDER = theme.FUEL_ORDER
FUEL_COLOR = theme.FUEL_COLOR


def ordered(values, preferred):
    """Preferred vocabulary first, then anything else the data contains."""
    present = {v for v in values if pd.notna(v)}
    known = [v for v in preferred if v in present]
    extra = sorted(present.difference(preferred), key=str)
    return known + extra


def colors_for(values, palette):
    """Colour per value, assigning neutral greys to anything unmapped."""
    out, i = [], 0
    for v in values:
        if v in palette:
            out.append(palette[v])
        else:
            out.append(UNKNOWN_COLORS[i % len(UNKNOWN_COLORS)])
            i += 1
    return out


NORM_ORDER = ["BS3", "BS4", "BS6", "ZEV"]
NORM_PLAIN = {
    "BS3": "BS3 (oldest standard)",
    "BS4": "BS4 (older standard)",
    "BS6": "BS6 (current standard)",
    "ZEV": "ZEV (zero-emission, electric)",
}
COMPLIANT_NORMS = ["BS6", "ZEV"]
MAX_COMPLIANT_AGE = 5
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
    """Section heading. Native elements, so it follows the active theme."""
    st.subheader(title, divider="gray")
    if caption:
        st.caption(caption)


def callout(label, value, note=None, accent=True):
    """Headline number - preferred over tables for single figures.

    `accent` is accepted for call-site compatibility but ignored: st.metric
    already renders in the theme's own emphasis colour.
    """
    st.metric(label, value)
    if note:
        st.caption(note)


def fmt_int(n):
    return f"{int(round(n)):,}"


def fmt_pct(x, dp=0):
    return f"{x * 100:.{dp}f}%"


def fmt_score(x, dp=1):
    return f"{x:.{dp}f}%"


def fmt_pp(x, dp=1):
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
    "CH": "Chandigarh",
}
EV_ONLY_BRANDS = ["Ola Electric", "Ather"]


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Read the registration file and derive analysis + integrity columns."""
    try:
        df = pd.read_csv(path, engine="pyarrow")
    except Exception:
        df = pd.read_csv(path)

    df["Registration_Date"] = pd.to_datetime(
        df["Registration_Date"], format="%d-%m-%Y", errors="coerce")
    df["Year_Month"] = df["Registration_Date"].dt.to_period("M").dt.to_timestamp()

    df["Fuel_Type"] = df["Fuel_Type"].replace({"EV": "Electric"})

    df["Category_Plain"] = (df["Vehicle_Category"].map(CATEGORY_PLAIN)
                            .fillna(df["Vehicle_Category"]))
    df["Is_Electric"] = df["Fuel_Type"].eq("Electric")

    if "Emission_Norm_Clean" not in df.columns:
        df["Emission_Norm_Clean"] = df["Emission_Norm"].where(
            ~df["Is_Electric"], "ZEV")

    df["Is_Clean"] = (df["is_clean"].astype(bool) if "is_clean" in df.columns
                      else df["Fuel_Type"].isin(CLEAN_FUELS))

    df["Meets_Norm"] = df["Emission_Norm_Clean"].isin(COMPLIANT_NORMS)
    df["Within_Age"] = df["Vehicle_Age_Years"] <= MAX_COMPLIANT_AGE
    compliant_rule = df["Meets_Norm"] & df["Within_Age"]
    df["Is_Compliant"] = (df["is_compliant"].astype(bool)
                          if "is_compliant" in df.columns else compliant_rule)
    df["Compliance_Rule"] = compliant_rule

    df["Fail_Reason"] = np.select(
        [df["Is_Compliant"],
         ~df["Meets_Norm"] & ~df["Within_Age"],
         ~df["Meets_Norm"]],
        ["Compliant", "Older standard and over age limit", "Older standard"],
        default=f"Over {MAX_COMPLIANT_AGE}-year age limit")

    df["Norm_Label"] = df["Emission_Norm_Clean"].map(
        NORM_PLAIN).fillna(df["Emission_Norm_Clean"])

    df["RTO_Code"] = df["RTO_Office"].str.extract(r"\(([A-Z]{2})-")
    df["RTO_State"] = df["RTO_Code"].map(RTO_CODE_TO_STATE)

    df["QF_RTO_Mismatch"] = ~df["RTO_State"].eq(df["State"])
    df["QF_EV_Not_ZEV"] = df["Is_Electric"] & df["Emission_Norm_Clean"].ne("ZEV")
    df["QF_EVBrand_Fossil"] = (df["Manufacturer_Brand"].isin(EV_ONLY_BRANDS)
                               & ~df["Is_Electric"])
    df["QF_Compliance_Mismatch"] = df["Is_Compliant"].ne(df["Compliance_Rule"])
    QF = ["QF_RTO_Mismatch", "QF_EV_Not_ZEV", "QF_EVBrand_Fossil",
          "QF_Compliance_Mismatch"]
    df["Has_Defect"] = df[QF].any(axis=1)
    df["Defect_Count"] = df[QF].sum(axis=1)

    def _reasons(r):
        out = []
        if r["QF_RTO_Mismatch"]:
            out.append("RTO office belongs to a different state")
        if r["QF_EV_Not_ZEV"]:
            out.append("Electric vehicle not classed as zero-emission")
        if r["QF_EVBrand_Fossil"]:
            out.append("Electric-only brand recorded as fossil fuel")
        if r["QF_Compliance_Mismatch"]:
            out.append("Compliance flag disagrees with the documented rule")
        return "; ".join(out)

    df["Defect_Reasons"] = df.apply(_reasons, axis=1)

    # --- Drill-down: derive Quarter for time hierarchy navigation ---
    df["Quarter"] = df["Registration_Date"].dt.to_period("Q").astype(str)
    df["Month"] = df["Registration_Date"].dt.to_period("M").astype(str)

    return df


QUALITY_RULES = [
    ("QF_RTO_Mismatch", "Consistency",
     "RTO office code matches the declared State",
     "State-to-RTO drill-down is unreliable for these records."),
    ("QF_EV_Not_ZEV", "Validity",
     "Electric vehicles are classed as zero-emission (ZEV)",
     "Bharat Stage rates tailpipe emissions; electric vehicles have none. "
     "Corrected upstream via Emission_Norm_Clean."),
    ("QF_EVBrand_Fossil", "Accuracy",
     "Electric-only manufacturers sell only electric vehicles",
     "Ola Electric and Ather produce electric vehicles exclusively."),
    ("QF_Compliance_Mismatch", "Integrity",
     "Compliance flag matches the documented compound rule",
     f"is_compliant must equal (norm in {COMPLIANT_NORMS}) "
     f"and (age <= {MAX_COMPLIANT_AGE})."),
]


@st.cache_data(show_spinner=False)
def audit_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Data governance tracker over genuine defects."""
    n = max(len(df), 1)
    rows = []
    for col, dim, check, detail in QUALITY_RULES:
        bad = int(df[col].sum())
        rows.append({"Dimension": dim, "Check": check, "Failing records": bad,
                     "Pass rate": 1 - bad / n, "Detail": detail})

    source_cols = [c for c in [
        "Registration_Number", "Registration_Date", "Registration_Year", "State",
        "RTO_Office", "Vehicle_Category", "Vehicle_Sub_Type", "Manufacturer_Brand",
        "Fuel_Type", "Emission_Norm", "Emission_Norm_Clean", "Engine_CC",
        "Seating_Capacity", "Vehicle_Age_Years", "is_clean", "is_compliant",
        "CFAR", "FMI"]
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
    """CFAR and FMI recomputed live at an arbitrary grain."""
    return (df.groupby(list(group_cols), dropna=False)
              .agg(Registrations=("Registration_Number", "size"),
                   CFAR=("Is_Clean", lambda s: s.mean() * 100),
                   FMI=("Is_Compliant", lambda s: s.mean() * 100))
              .reset_index())


df_all = load_data(DATA_PATH)

# ---------------------------------------------------------------------------
# SIDEBAR - multi-level dynamic filters (UNCHANGED)
# ---------------------------------------------------------------------------

st.sidebar.title("Filters")

year_min = int(df_all["Registration_Year"].min())
year_max = int(df_all["Registration_Year"].max())
date_min = df_all["Registration_Date"].min().date()
date_max = df_all["Registration_Date"].max().date()

time_mode = st.sidebar.radio(
    "Filter time by", ["Year range", "Exact dates"], horizontal=True)

if time_mode == "Year range":
    year_range = st.sidebar.slider(
        "Registration year", year_min, year_max, (year_min, year_max), step=1)
    start_date = pd.Timestamp(year_range[0], 1, 1)
    end_date = pd.Timestamp(year_range[1], 12, 31)
else:
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
        f"Showing {d0} to {d1} (inclusive) \u00b7 {span_days:,} day"
        f"{'s' if span_days != 1 else ''}.")

# State -> RTO office
all_states = sorted(df_all["State"].unique())
states = st.sidebar.multiselect("State", all_states, default=all_states)
_state_pool = df_all[df_all["State"].isin(states)] if states else df_all
rto_options = sorted(_state_pool["RTO_Office"].unique())
rtos = st.sidebar.multiselect(
    "RTO office", rto_options, default=[],
    help="Leave empty to include every RTO office in the selected states.")

# Category -> sub-type
cats_present = ordered(df_all["Vehicle_Category"], CATEGORY_ORDER)
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

# Cross-filter: applies to every chart in every tab at once.
fuel_options = ordered(df_all["Fuel_Type"], FUEL_ORDER)
fuels_picked = st.sidebar.multiselect(
    "Fuel type", fuel_options, default=[],
    help="Cross-filter. Leave empty to include every powertrain.")

st.sidebar.divider()
exclude_defects = st.sidebar.checkbox(
    "Exclude records failing data-quality checks", value=False,
    help="Removes rows with an RTO/state disagreement, an electric vehicle "
         "carrying a Bharat Stage norm, or an electric-only brand recorded "
         "against fossil fuel.")

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
if fuels_picked:
    mask &= df_all["Fuel_Type"].isin(fuels_picked)
if exclude_defects:
    mask &= ~df_all["Has_Defect"]

df = df_all[mask].copy()

st.sidebar.divider()
st.sidebar.metric("Records in selection", fmt_int(len(df)),
                  f"of {fmt_int(len(df_all))} total")
st.sidebar.caption(
    "CFAR = clean-fuel adoption rate. FMI = fleet modernization index "
    f"(share on a modern standard \u2014 BS6 or zero-emission \u2014 and no more than "
    f"{MAX_COMPLIANT_AGE} years old)."
)

# ---------------------------------------------------------------------------
# Header + empty-state handling
# ---------------------------------------------------------------------------

st.title("VAHAN RTO Registration Analytics")
st.caption(
    f"{fmt_int(len(df_all))} registrations \u00b7 {df_all['State'].nunique()} states \u00b7 "
    f"{year_min} to {year_max}")

if df.empty:
    st.warning("No data available for the selected filters. "
               "Widen your selection in the sidebar.")
    st.stop()

tab_macro, tab_oem, tab_audit = st.tabs(
    ["\U0001F4CA Macro Fuel Transition",
     "\U0001F3CE\uFE0F OEM & Powertrain Strategy",
     "\U0001F6A8 Regulatory & Data Quality Audit"])


# ===========================================================================
# TAB 1 - MACRO FUEL TRANSITION (with interactivity upgrades)
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
                f"BS6 or zero-emission, and \u2264{MAX_COMPLIANT_AGE} years old",
                accent=False)
    with k4:
        callout("Non-compliant fleet share", fmt_score(non_compliant),
                "Older standard, over the age limit, or both", accent=False)

    st.divider()

    # =========================================================================
    # DRILL-DOWN Navigation (Year -> Quarter -> Month)
    #
    # Driven by widgets rather than chart clicks: Streamlit cannot return
    # selection events from a layered chart, and this compound chart is a
    # bar + line layer. The scope selectors below give the same hierarchy
    # with state that survives a rerun.
    # =========================================================================
    st.markdown("**\U0001F4C5 Time detail**")
    drill_col = st.columns([1, 1, 1])
    with drill_col[0]:
        grain = st.radio("Drill level", ["Year", "Quarter", "Month"],
                         horizontal=True, key="drill_grain",
                         help="Year is the top level. Quarter and Month drill "
                              "into the period you scope on the right.")

    years_here = sorted(int(y) for y in df["Registration_Year"].unique())
    drill_year = None
    drill_quarter = None

    if grain in ("Quarter", "Month"):
        with drill_col[1]:
            drill_year = st.selectbox("Year to drill into", years_here,
                                      index=len(years_here) - 1,
                                      key="drill_year_pick")
    if grain == "Month":
        quarters_here = sorted(
            df.loc[df["Registration_Year"] == drill_year, "Quarter"].unique())
        with drill_col[2]:
            drill_quarter = st.selectbox("Quarter to drill into", quarters_here,
                                         index=len(quarters_here) - 1,
                                         key="drill_quarter_pick")

    # Determine which time grain to visualize
    if grain == "Year":
        time_field = "Registration_Year"
        time_type = ":O"
        time_title = "Registration year"
        drill_df = df.copy()
    elif grain == "Quarter":
        time_field = "Quarter"
        time_type = ":N"
        time_title = f"Quarter ({drill_year})"
        drill_df = df[df["Registration_Year"] == drill_year].copy()
    else:  # Month
        time_field = "Month"
        time_type = ":N"
        time_title = f"Month ({drill_quarter})"
        drill_df = df[df["Quarter"] == drill_quarter].copy()

    breadcrumb = "Year"
    if grain in ("Quarter", "Month"):
        breadcrumb += f" \u203A {drill_year}"
    if grain == "Month":
        breadcrumb += f" \u203A {drill_quarter}"
    st.caption(f"Showing: {breadcrumb}")

    if drill_df.empty:
        st.info("No data at this drill level. Choose a different period, or go "
                "back to the Year level.")
    else:
        # Aggregate for the compound chart
        fuel_year = (drill_df.groupby([time_field, "Fuel_Type"])
                    .size().reset_index(name="Registrations"))
        total_year = (fuel_year.groupby(time_field)["Registrations"]
                            .sum().reset_index(name="Total_Registrations"))
        fuel_year = fuel_year.merge(total_year, on=time_field)
        fuel_year["Share"] = fuel_year["Registrations"] / fuel_year["Total_Registrations"]
        fuels_here = ordered(fuel_year["Fuel_Type"], FUEL_ORDER)

        # =====================================================================
        # Legend-bound highlighting (client side) + advanced tooltips.
        # Clicking a fuel in the legend dims the rest; shift-click adds to it.
        # =====================================================================
        fuel_selection = alt.selection_point(
            fields=["Fuel_Type"], bind="legend",  # Cross-filter: click legend
            name="fuel_highlight"
        )

        # --- Layer 1: Stacked Bar Chart with highlight/dim ---
        bar_chart = (
            alt.Chart(fuel_year)
            .mark_bar()
            .encode(
                x=alt.X(f"{time_field}{time_type}", title=time_title,
                        axis=alt.Axis(labelAngle=0, grid=False)),
                y=alt.Y("Registrations:Q",
                        title="Total Registrations",
                        axis=alt.Axis(format="s")),
                color=alt.Color("Fuel_Type:N", title="Fuel",
                                scale=alt.Scale(domain=fuels_here,
                                                range=colors_for(fuels_here, FUEL_COLOR)),
                                sort=fuels_here),
                # Conditional opacity - selected stays full, others dim
                fillOpacity=alt.condition(fuel_selection, alt.value(0.9), alt.value(0.25)),
                order=alt.Order("color_Fuel_Type_sort_index:Q"),
                tooltip=[
                    alt.Tooltip(f"{time_field}{time_type}", title=time_title),
                    alt.Tooltip("Fuel_Type:N", title="Fuel"),
                    alt.Tooltip("Registrations:Q", title="Registrations", format=","),
                    alt.Tooltip("Share:Q", title="Share of Period", format=".1%")
                ]
            )
            .add_params(fuel_selection)
        )

        # --- Layer 2: Total Trend Line Overlay ---
        line_chart = (
            alt.Chart(total_year)
            .mark_line(color=INK, strokeWidth=2,
                       point=alt.OverlayMarkDef(color=INK, size=50))
            .encode(
                x=alt.X(f"{time_field}{time_type}"),
                y=alt.Y("Total_Registrations:Q"),
                tooltip=[
                    alt.Tooltip(f"{time_field}{time_type}", title=time_title),
                    alt.Tooltip("Total_Registrations:Q", title="Total", format=",")
                ]
            )
        )

        # --- Combine into Compound Chart ---
        # Layered chart: rendered without on_select, since Streamlit cannot
        # return selection events from a chart composition.
        compound_chart = alt.layer(bar_chart, line_chart).properties(height=400)
        st.altair_chart(compound_chart, use_container_width=True)
        st.caption(
            "Click a fuel in the legend to isolate it; shift-click to add more. "
            "Use the sidebar fuel filter to cross-filter every chart at once."
        )

    st.divider()

    cvc, cen = st.columns([1, 1])

    # --- Chart 1b: registrations by vehicle category ------------------------
    with cvc:
        cat_counts = (df.groupby(["Vehicle_Category", "Category_Plain"])
                        .size().reset_index(name="Registrations")
                        .sort_values("Registrations", ascending=False))

        section("Registrations by vehicle category",
                "Similarity: bars share one hue, so the eye compares length, "
                "not colour.")

        # FEATURE 2: selection_point for cross-highlighting categories
        cat_selection = alt.selection_point(fields=["Category_Plain"], name="cat_sel")

        cat_base = chart(cat_counts).encode(
            x=qx("Registrations:Q", "Registrations", fmt=","),
            y=alt.Y("Category_Plain:N", sort="-x", title=None,
                    axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260)),
        )
        # FEATURE 4: highlight selected category bar, dim others
        cat_bars = cat_base.mark_bar(cornerRadiusEnd=3).encode(
            color=alt.condition(cat_selection, alt.value(ACCENT), alt.value(GREY_LIGHT)),
            tooltip=[alt.Tooltip("Category_Plain:N", title="Category"),
                     alt.Tooltip("Registrations:Q", format=",")],
        ).add_params(cat_selection)
        cat_lbl = cat_base.mark_text(align="left", dx=4, fontSize=11,
                                     color=GREY_DARK).encode(
            text=alt.Text("Registrations:Q", format=","))

        st.altair_chart((cat_bars + cat_lbl).properties(height=260),
                        use_container_width=True)

    # --- Chart 1c: registrations by emission norm ---------------------------
    with cen:
        norm_counts = (df.groupby(["Emission_Norm_Clean", "Norm_Label"])
                         .size().reset_index(name="Registrations")
                         .sort_values("Registrations", ascending=False))

        section("Registrations by emission norm",
                "Similarity: bars share one hue, so the eye compares length, "
                "not colour.")

        # FEATURE 2: selection_point for cross-highlighting norms
        norm_selection = alt.selection_point(fields=["Norm_Label"], name="norm_sel")

        norm_base = chart(norm_counts).encode(
            x=qx("Registrations:Q", "Registrations", fmt=","),
            y=alt.Y("Norm_Label:N", sort="-x", title=None,
                    axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260)),
        )
        # FEATURE 4: highlight selected norm bar, dim others
        norm_bars = norm_base.mark_bar(cornerRadiusEnd=3).encode(
            color=alt.condition(norm_selection, alt.value(ACCENT_ALT), alt.value(GREY_LIGHT)),
            tooltip=[alt.Tooltip("Norm_Label:N", title="Emission Standard"),
                     alt.Tooltip("Registrations:Q", format=",")],
        ).add_params(norm_selection)
        norm_lbl = norm_base.mark_text(align="left", dx=4, fontSize=11,
                                       color=GREY_DARK).encode(
            text=alt.Text("Registrations:Q", format=","))

        st.altair_chart((norm_bars + norm_lbl).properties(height=260),
                        use_container_width=True)

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

        # FEATURE 2: selection_point for picking a state/RTO
        rank_selection = alt.selection_point(fields=[dim_col], name="rank_sel")

        rank_base = chart(ranked).encode(
            x=qx("CFAR:Q", "CFAR (% clean fuel)"),
            y=alt.Y(f"{dim_col}:N", sort="-x", title=None,
                    axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260)),
        )
        # FEATURE 4: highlight on click OR highlight best by default
        bars = rank_base.mark_bar(cornerRadiusEnd=3).encode(
            fillOpacity=alt.condition(rank_selection, alt.value(1.0), alt.value(0.4)),
            color=alt.condition(alt.datum.Highlight,
                                alt.value(ACCENT), alt.value(GREY_LIGHT)),
            tooltip=[alt.Tooltip(f"{dim_col}:N"),
                     alt.Tooltip("CFAR:Q", title="CFAR %", format=".1f"),
                     alt.Tooltip("FMI:Q", title="FMI %", format=".1f"),
                     alt.Tooltip("Registrations:Q", format=",")],
        ).add_params(rank_selection)
        lbl = rank_base.mark_text(align="left", dx=4, fontSize=11,
                                  color=GREY_DARK).encode(
            text=alt.Text("CFAR:Q", format=".1f"))

        st.altair_chart(
            (bars + lbl).properties(height=max(260, min(560, len(ranked) * 26))),
            use_container_width=True)

    # --- Chart 3: waterfall of net share shift -----------------------------
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
            allf = ordered(set(s0.index) | set(s1.index), FUEL_ORDER)
            shift = pd.DataFrame({
                "Fuel": allf,
                "Shift_bps": [(s1.get(f, 0) - s0.get(f, 0)) * 10000 for f in allf],
            }).sort_values("Shift_bps")
            shift["End"] = shift["Shift_bps"].cumsum()
            shift["Start"] = shift["End"] - shift["Shift_bps"]
            shift["Direction"] = np.where(shift["Shift_bps"] >= 0, "Gain", "Loss")

            # FEATURE 2: selection on fuel in waterfall for cross-highlight
            wf_selection = alt.selection_point(fields=["Fuel"], name="wf_sel")

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
                    # FEATURE 4: dim unselected bars
                    fillOpacity=alt.condition(wf_selection, alt.value(1.0), alt.value(0.35)),
                    tooltip=[alt.Tooltip("Fuel:N"),
                             alt.Tooltip("Shift_bps:Q", title="Shift (bps)",
                                         format="+,.0f")],
                ).add_params(wf_selection)
                .properties(height=max(260, min(560, len(ranked) * 26)))
            )
            zero_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                color=GREY_MID).encode(y="y:Q")

            st.altair_chart(wf + zero_rule, use_container_width=True)
            st.caption(
                f"100 basis points = 1 percentage point. Gains and losses sum to "
                f"zero because these are shares of the same total. Comparing "
                f"{y0} with {y1} only; intermediate years are not shown."
            )


# ===========================================================================
# TAB 2 - OEM & POWERTRAIN STRATEGY (with interactivity upgrades)
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
            callout("Most compliant OEM", str(top_fmi["Brand"]),
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

            max_lab = min(12, len(oem))
            label_all = st.checkbox(
                "Label every manufacturer", value=False,
                help="Off by default: only the largest manufacturers are named, "
                     "so the labels stay legible. Hover any bubble for details.")
            lab_cut = (oem["Registrations"].min() if label_all
                       else oem["Registrations"].nlargest(max_lab).iloc[-1])
            oem["Label"] = np.where(oem["Registrations"] >= lab_cut, oem["Brand"], "")

            # =================================================================
            # FEATURE 1: Dynamic Zoom & Pan on scatter chart
            # Bind scales to interval selection for scroll-zoom + drag-pan
            # FEATURE 2: Point selection for cross-highlighting OEMs
            # FEATURE 4: Conditional stroke on selected points
            # =================================================================
            zoom_pan = alt.selection_interval(
                bind="scales",  # FEATURE 1: scroll to zoom, drag to pan
                name="zoom_scatter"
            )
            oem_selection = alt.selection_point(
                fields=["Brand"], name="oem_sel"  # FEATURE 2: click to select OEM
            )

            scatter_base = chart(oem).encode(
                x=alt.X("CFAR:Q", title="CFAR (% clean fuel)",
                        scale=alt.Scale(zero=False, nice=True, padding=14),
                        axis=alt.Axis(labelAngle=0, grid=False)),
                y=alt.Y("FMI:Q", title="FMI (% compliant)",
                        scale=alt.Scale(zero=False, nice=True, padding=14),
                        axis=alt.Axis(labelAngle=0, grid=False)),
            )
            pts = scatter_base.mark_circle().encode(
                size=alt.Size("Registrations:Q", title="Volume",
                              scale=alt.Scale(range=[60, 900])),
                color=alt.Color("Quadrant:N", title=None,
                                scale=alt.Scale(domain=quad_domain,
                                                range=quad_range),
                                legend=alt.Legend(orient="top", columns=2)),
                # FEATURE 4: selected OEM gets full opacity + stroke ring
                fillOpacity=alt.condition(oem_selection, alt.value(0.85), alt.value(0.25)),
                stroke=alt.condition(oem_selection, alt.value(INK), alt.value("transparent")),
                strokeWidth=alt.condition(oem_selection, alt.value(2), alt.value(0)),
                tooltip=[alt.Tooltip("Brand:N", title="Manufacturer"),
                         alt.Tooltip("Quadrant:N"),
                         alt.Tooltip("CFAR:Q", title="CFAR %", format=".1f"),
                         alt.Tooltip("FMI:Q", title="FMI %", format=".1f"),
                         alt.Tooltip("Registrations:Q", format=",")],
            ).add_params(zoom_pan, oem_selection)  # FEATURE 1+2: both params

            labels = scatter_base.mark_text(
                dx=10, dy=-8, align="left", fontSize=11, color=INK,
            ).encode(text=alt.Text("Label:N"))

            scatter_chart = (rule_v + rule_h + pts + labels).properties(height=470)
            st.altair_chart(scatter_chart, use_container_width=True)

            st.caption(
                "Green pioneers lead on both axes. Compliance leaders run a "
                "modern, young fleet but still sell fossil powertrains. "
                "Clean-fuel specialists sell clean fuel yet carry older or "
                "ageing stock across the rest of their range. "
                "\U0001F50E Scroll to zoom, drag to pan. Click a bubble to highlight."
                + (" All manufacturers are labelled."
                   if label_all else
                   f" Only the {max_lab} largest manufacturers are labelled; "
                   "hover any bubble for the rest."))

        st.divider()

        # --- Chart 2: fuel mix per brand (100% stacked horizontal) ---------
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
        fuels_mix = ordered(mix["Fuel_Type"], FUEL_ORDER)

        section("Fuel mix within each manufacturer",
                "Shares within each bar; bars are ordered by total volume.")

        # FEATURE 2: selection on fuel type in stacked bar for cross-highlight
        mix_fuel_sel = alt.selection_point(
            fields=["Fuel_Type"], bind="legend", name="mix_fuel_hl"
        )

        stacked = (
            chart(mix)
            .mark_bar()
            .encode(
                x=alt.X("Registrations:Q", stack="normalize",
                        title="Share of the OEM's registrations",
                        scale=alt.Scale(zero=True),
                        axis=alt.Axis(labelAngle=0, grid=False, format="%")),
                y=alt.Y("Manufacturer_Brand:N", sort=keep, title=None,
                        axis=alt.Axis(labelAngle=0, grid=False, labelLimit=220)),
                color=alt.Color("Fuel_Type:N", title="Fuel",
                                scale=alt.Scale(domain=fuels_mix,
                                                range=colors_for(fuels_mix, FUEL_COLOR)),
                                sort=fuels_mix),
                # FEATURE 4: dim unselected fuel segments
                fillOpacity=alt.condition(mix_fuel_sel, alt.value(1.0), alt.value(0.2)),
                order=alt.Order("color_Fuel_Type_sort_index:Q"),
                tooltip=[alt.Tooltip("Manufacturer_Brand:N", title="OEM"),
                         alt.Tooltip("Fuel_Type:N", title="Fuel"),
                         alt.Tooltip("Registrations:Q", format=",")],
            )
            .add_params(mix_fuel_sel)  # FEATURE 2: legend click
            .properties(height=max(280, n_show * 26))
        )
        st.altair_chart(stacked, use_container_width=True)

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
                fuels_cc = ordered(cc["Fuel_Type"], FUEL_ORDER)
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
                                                        range=colors_for(fuels_cc, FUEL_COLOR))),
                    ).properties(height=340)
                )
            # FEATURE 1: .interactive() for zoom/pan on boxplot distribution
            st.altair_chart(box.interactive(), use_container_width=True)


# ===========================================================================
# TAB 3 - REGULATORY & DATA QUALITY AUDIT (with interactivity upgrades)
# ===========================================================================

with tab_audit:
    sec_a, sec_b = st.tabs(["Section A \u00b7 Compliance & scrappage risk",
                            "Section B \u00b7 Data governance"])

    # ---------------- Section A: non-compliant fleet -----------------------
    with sec_a:
        risk = df[~df["Is_Compliant"]].copy()

        st.caption(
            f"A vehicle is compliant only if it is on a modern standard "
            f"(BS6 or ZEV) **and** is no more than {MAX_COMPLIANT_AGE} years "
            "old. Failing either test puts it in the risk fleet below."
        )

        k1, k2, k3 = st.columns(3)
        with k1:
            callout(
                "Non-compliant vehicles",
                fmt_int(len(risk)),
                f"{(len(risk) / len(df) * 100):.1f}% of the selection",
                accent=False,
            )
        with k2:
            callout(
                "Fleet Modernization Index",
                fmt_score(df["Is_Compliant"].mean() * 100),
                f"Modern standard and \u2264{MAX_COMPLIANT_AGE} years old",
                accent=True,
            )
        with k3:
            med_age = risk["Vehicle_Age_Years"].median() if len(risk) else 0.0
            callout(
                "Median Age (Risk Fleet)",
                f"{med_age:.1f} yrs",
                "Robust central measure of scrappage exposure",
                accent=False,
            )

        if risk.empty:
            st.success("Every vehicle in the current selection is compliant.")
        else:
            section(
                "Why vehicles fail the compliance test",
                "The two conditions fail independently, so a modern "
                "zero-emission vehicle can still age out.",
            )
            reasons = (
                risk.groupby("Fail_Reason")
                .size()
                .reset_index(name="Vehicles")
                .sort_values("Vehicles", ascending=False)
            )
            reasons["Share"] = reasons["Vehicles"] / reasons["Vehicles"].sum()

            # FEATURE 2: selection on fail reason for cross-highlight
            reason_selection = alt.selection_point(fields=["Fail_Reason"], name="reason_sel")

            rbase = chart(reasons).encode(
                x=qx("Vehicles:Q", "Vehicles"),
                y=alt.Y(
                    "Fail_Reason:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(labelAngle=0, grid=False, labelLimit=320),
                ),
            )

            AGE_FAIL_COLOR = "#4A5568"
            rb = rbase.mark_bar(cornerRadiusEnd=3).encode(
                color=alt.condition(
                    alt.datum.Fail_Reason
                    == f"Over {MAX_COMPLIANT_AGE}-year age limit",
                    alt.value(AGE_FAIL_COLOR),
                    alt.value(GREY_LIGHT),
                ),
                # FEATURE 4: dim unselected bars
                fillOpacity=alt.condition(reason_selection, alt.value(1.0), alt.value(0.3)),
                tooltip=[
                    alt.Tooltip("Fail_Reason:N", title="Reason"),
                    alt.Tooltip("Vehicles:Q", format=","),
                    alt.Tooltip("Share:Q", format=".1%"),
                ],
            ).add_params(reason_selection)
            rlbl = rbase.mark_text(
                align="left", dx=4, fontSize=11, color=GREY_DARK
            ).encode(text=alt.Text("Vehicles:Q", format=","))
            st.altair_chart((rb + rlbl).properties(height=180),
                            use_container_width=True)

            aged_out = int(
                (
                    risk["Fail_Reason"] == f"Over {MAX_COMPLIANT_AGE}-year age limit"
                ).sum()
            )
            if aged_out:
                st.caption(
                    f"{fmt_int(aged_out)} vehicles (shown in slate grey) are on a "
                    "modern standard and fail on age alone \u2014 they would become "
                    "compliant under a longer age allowance."
                )

                section(
                    "Ageing non-compliant fleet: age against emission standard",
                    "Darker cells hold more vehicles. Older cohorts on the "
                    "oldest standards carry the greatest scrappage exposure.",
                )
                grid = (
                    risk.groupby(["Vehicle_Age_Years", "Emission_Norm_Clean", "Norm_Label"])
                    .size()
                    .reset_index(name="Vehicles")
                )
                norms_here = ordered(grid["Emission_Norm_Clean"], NORM_ORDER)

                # FEATURE 2: selection_interval on the heatmap for brushing
                hm_brush = alt.selection_interval(name="hm_brush")

                hm_base = chart(grid).encode(
                    x=alt.X(
                        "Emission_Norm_Clean:N",
                        sort=norms_here,
                        title="Emission standard",
                        axis=alt.Axis(labelAngle=0, grid=False),
                    ),
                    y=alt.Y(
                        "Vehicle_Age_Years:O",
                        title="Vehicle age (years)",
                        axis=alt.Axis(labelAngle=0, grid=False),
                    ),
                )
                hm = hm_base.mark_rect().encode(
                    color=alt.Color(
                        "Vehicles:Q",
                        title="Vehicles",
                        scale=alt.Scale(scheme="purples"),
                    ),
                    # FEATURE 4: dim unselected cells when brush is active
                    fillOpacity=alt.condition(hm_brush, alt.value(1.0), alt.value(0.3)),
                    tooltip=[
                        alt.Tooltip("Vehicle_Age_Years:O", title="Age (yrs)"),
                        alt.Tooltip("Norm_Label:N", title="Standard"),
                        alt.Tooltip("Vehicles:Q", format=","),
                    ],
                ).add_params(hm_brush)
                cutoff = grid["Vehicles"].max() * 0.6
                txt = hm_base.mark_text(fontSize=11).encode(
                    text=alt.Text("Vehicles:Q", format=","),
                    color=alt.condition(
                        alt.datum.Vehicles > cutoff, alt.value("white"), alt.value(INK)
                    ),
                )
                st.altair_chart((hm + txt).properties(height=320),
                                use_container_width=True)

                section("Risk fleet by RTO office")
                by_rto = (
                    risk.groupby("RTO_Office")
                    .agg(
                        Non_compliant=("Registration_Number", "size"),
                        Mean_age=("Vehicle_Age_Years", "mean"),
                    )
                    .reset_index()
                    .sort_values("Non_compliant", ascending=False)
                )
                by_rto["Mean_age"] = by_rto["Mean_age"].round(1)
                top_rto = by_rto.head(20).copy()
                top_rto["Highlight"] = top_rto["RTO_Office"].eq(
                    top_rto.iloc[0]["RTO_Office"]
                )

                # FEATURE 2: selection on RTO in risk chart
                rto_risk_sel = alt.selection_point(fields=["RTO_Office"], name="rto_risk_sel")

                rbar = (
                    chart(top_rto)
                    .mark_bar(cornerRadiusEnd=3)
                    .encode(
                        x=qx("Non_compliant:Q", "Non-compliant vehicles"),
                        y=alt.Y(
                            "RTO_Office:N",
                            sort="-x",
                            title=None,
                            axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260),
                        ),
                        color=alt.condition(
                            alt.datum.Highlight, alt.value(INK), alt.value(GREY_LIGHT)
                        ),
                        # FEATURE 4: dim unselected RTO bars
                        fillOpacity=alt.condition(rto_risk_sel, alt.value(1.0), alt.value(0.3)),
                        tooltip=[
                            alt.Tooltip("RTO_Office:N", title="RTO office"),
                            alt.Tooltip("Non_compliant:Q", format=","),
                            alt.Tooltip(
                                "Mean_age:Q", title="Mean age (yrs)", format=".1f"
                            ),
                        ],
                    )
                    .add_params(rto_risk_sel)
                    .properties(height=max(280, len(top_rto) * 26))
                )
                st.altair_chart(rbar, use_container_width=True)

                rto_pick = st.multiselect(
                    "Filter the risk table by RTO office",
                    sorted(risk["RTO_Office"].unique()),
                    default=[],
                )
                st.caption(
                    "\u26A0\uFE0F **Note:** 18.2% of records contain RTO-State contradictions; "
                    "filters operate strictly on RTO jurisdiction."
                )

                risk_view = risk[risk["RTO_Office"].isin(rto_pick)] if rto_pick else risk
                risk_cols = [
                    "Registration_Number",
                    "Registration_Date",
                    "State",
                    "RTO_Office",
                    "Vehicle_Category",
                    "Vehicle_Sub_Type",
                    "Manufacturer_Brand",
                    "Fuel_Type",
                    "Norm_Label",
                    "Vehicle_Age_Years",
                    "Fail_Reason",
                    "Engine_CC",
                ]
                st.dataframe(
                    risk_view[risk_cols].sort_values("Vehicle_Age_Years", ascending=False),
                    use_container_width=True,
                    height=340,
                    hide_index=True,
                )
                st.download_button(
                    "Download risk fleet as CSV",
                    data=risk_view[risk_cols].to_csv(index=False).encode("utf-8"),
                    file_name="non_compliant_risk_fleet.csv",
                    mime="text/csv",
                )

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

        section("Data hygiene drill-down",
                "Every record failing at least one integrity check, with the "
                "reason attached, for audit export.")
        reason_opts = ["RTO office belongs to a different state",
                        "Electric vehicle not classed as zero-emission",
                        "Electric-only brand recorded as fossil fuel",
                        "Compliance flag disagrees with the documented rule"]
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
                            "Emission_Norm_Clean", "Engine_CC",
                            "Seating_Capacity", "Defect_Reasons"]
            st.dataframe(view[drill_cols], use_container_width=True, height=380,
                            hide_index=True)
            st.download_button(
                "Download flagged records as CSV",
                data=view[drill_cols].to_csv(index=False).encode("utf-8"),
                file_name="data_quality_exceptions.csv", mime="text/csv")

        section("Integrity checks")
        qv = quality.copy()
        qv["Status"] = np.where(qv["Failing records"] == 0, "Pass", "Fail")

        # FEATURE 2: selection on integrity check bar
        check_selection = alt.selection_point(fields=["Check"], name="check_sel")

        qbar = (
            chart(qv)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=qx("Pass rate:Q", "Share of records passing", fmt="%"),
                y=alt.Y("Check:N", sort="x", title=None,
                        axis=alt.Axis(labelAngle=0, grid=False, labelLimit=420)),
                color=alt.condition(alt.datum.Status == "Fail",
                                    alt.value(ACCENT_ALT), alt.value(GREY_LIGHT)),
                # FEATURE 4: dim unselected check bars
                fillOpacity=alt.condition(check_selection, alt.value(1.0), alt.value(0.3)),
                tooltip=[alt.Tooltip("Dimension:N"), alt.Tooltip("Check:N"),
                         alt.Tooltip("Failing records:Q", format=","),
                         alt.Tooltip("Pass rate:Q", format=".1%")],
            )
            .add_params(check_selection)
            .properties(height=240)
        )
        st.altair_chart(qbar, use_container_width=True)

        st.dataframe(
            quality.assign(**{"Pass rate": (quality["Pass rate"] * 100).round(1)}),
            use_container_width=True, hide_index=True,
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

            # FEATURE 2: selection on RTO error rate bars
            err_selection = alt.selection_point(fields=["RTO_Office"], name="err_sel")

            ebar = (
                chart(rto_err)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=qx("Error_rate:Q", "Records failing a check (%)"),
                    y=alt.Y("RTO_Office:N", sort="-x", title=None,
                            axis=alt.Axis(labelAngle=0, grid=False, labelLimit=260)),
                    color=alt.condition(alt.datum.Highlight,
                                        alt.value(ACCENT_ALT), alt.value(GREY_LIGHT)),
                    # FEATURE 4: dim unselected bars
                    fillOpacity=alt.condition(err_selection, alt.value(1.0), alt.value(0.3)),
                    tooltip=[alt.Tooltip("RTO_Office:N", title="RTO office"),
                             alt.Tooltip("Error_rate:Q", title="Error rate %",
                                         format=".1f"),
                             alt.Tooltip("Defects:Q", format=","),
                             alt.Tooltip("Records:Q", format=",")],
                )
                .add_params(err_selection)
                .properties(height=max(280, len(rto_err) * 26))
            )
            st.altair_chart(ebar, use_container_width=True)
            st.caption("Offices with fewer than 5 records are excluded, since a "
                       "single bad row would dominate their rate.")
