"""
theme.py - Plug-and-Play Global Theme Engine for VAHAN RTO Analytics
Injects styling directly into Streamlit and Altair without breaking existing code.
"""

import altair as alt
import streamlit as st

# ==============================================================================
# 1. GLOBAL CONSTANTS (Exported for direct import if needed)
# ==============================================================================
ACCENT = "#1F6FB2"       # Hero Blue: Zero-Emission / Electric
ACCENT_ALT = "#E08214"   # Safety Orange: Hybrid / Transition Accent
GREY_DARK = "#54595F"    # Diesel
GREY_MID = "#8C9196"     # Petrol
GREY_LIGHT = "#C9CDD1"   # Borders / Dividers
GREY_FAINT = "#F8F9FA"   # Gestalt Container Background
INK = "#25292E"          # Soft Charcoal Typography
CANVAS = "#FFFFFF"       # Main Canvas

CLEAN_FUELS = ["Electric", "CNG", "Hybrid"]
FUEL_ORDER = ["Petrol", "Diesel", "CNG", "Hybrid", "Electric"]
UNKNOWN_COLORS = [
    "#2B5C8F",  # 1. Deep Slate Blue (High prominence)
    "#E08214",  # 2. Warm Safety Orange (Secondary highlight)
    "#3B7A57",  # 3. Muted Sage Green (Eco/Clean neutral)
    "#705A8C",  # 4. Muted Plum/Indigo (Distinct categorical)
    "#4A8B9C",  # 5. Muted Cyan/Teal (Transition tone)
    "#54595F",  # 6. Dark Charcoal (GREY_DARK)
    "#8C9196",  # 7. Mid Grey (GREY_MID)
    "#9CC3DE",  # 8. Soft Sky Blue
]
FUEL_COLOR = {
    "Petrol": GREY_MID,
    "Diesel": GREY_DARK,
    "CNG": "#9CC3DE",
    "Hybrid": ACCENT_ALT,
    "Electric": ACCENT
}

# ==============================================================================
# 2. ALTAIR GLOBAL THEME ENGINE
# ==============================================================================
def _vahan_altair_theme():
    return {
        "config": {
            "background": CANVAS,
            "view": {"stroke": "transparent"},  # Removes heavy outer borders
            "axis": {
                "domainColor": GREY_LIGHT,
                "gridColor": "#E8EAEC",
                "gridDash": [3, 3],
                "labelColor": INK,
                "labelFontSize": 11,
                "titleColor": INK,
                "titleFontSize": 12,
                "titleFontWeight": "bold",
            },
            "legend": {
                "labelColor": INK,
                "titleColor": INK,
                "titleFontWeight": "bold",
                "orient": "top"
            },
            "header": {
                "labelColor": INK,
                "titleColor": INK,
            }
        }
    }

# Register Altair theme globally on import
alt.themes.register("vahan_clean_theme", _vahan_altair_theme)
alt.themes.enable("vahan_clean_theme")

# Easy Altair Color Encoder
def fuel_color_scale():
    """Drop-in helper for Altair color encodings.
    Usage in chart: color=fuel_color_scale()
    """
    return alt.Color(
        "Fuel_Type:N",
        scale=alt.Scale(
            domain=FUEL_ORDER,
            range=[FUEL_COLOR[f] for f in FUEL_ORDER]
        ),
        legend=alt.Legend(title="Powertrain Type")
    )

# ==============================================================================
# 3. STREAMLIT GLOBAL CSS INJECTOR
# ==============================================================================
def apply_dashboard_theme():
    """Call this ONE time right after st.set_page_config() in app.py.
    It automatically styles your existing 2,000 lines of code.
    """
    css = f"""
    <style>
        /* Main background & base text color */
        .stApp {{
            background-color: {CANVAS};
            color: {INK};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        
        /* Headers & Typography */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
            color: {INK} !important;
        }}
        
        /* Gestalt Enclosure: Automatic styling for st.container(border=True) and Expander cards */
        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            background-color: {GREY_FAINT} !important;
            border: 1px solid {GREY_LIGHT} !important;
            border-radius: 8px !important;
            padding: 12px !important;
        }}
        
        /* Callout / Metric Cards Styling */
        [data-testid="stMetric"] {{
            background-color: {GREY_FAINT};
            border: 1px solid {GREY_LIGHT};
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 4px solid {ACCENT} !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {ACCENT} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: {INK} !important;
            font-size: 0.85rem !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Clean Tab Navigation Styling */
        button[data-baseweb="tab"] {{
            color: {GREY_DARK} !important;
            font-weight: 600 !important;
        }}
        
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {ACCENT} !important;
            border-bottom-color: {ACCENT} !important;
        }}
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background-color: {GREY_FAINT} !important;
            border-right: 1px solid {GREY_LIGHT} !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)