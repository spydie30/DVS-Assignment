"""
Palette for the VAHAN RTO dashboard.

Deliberately colours only - no CSS injection and no Altair theme override, so
the app inherits Streamlit's default theme (light or dark, following the user's
setting) and Streamlit's built-in chart theme. These constants are used only
for explicit data encodings: accent vs. muted, and fuel-type identity.
"""

# ---------------------------------------------------------------------------
# Semantic colours - mid-tone so they read on a light or a dark background
# ---------------------------------------------------------------------------

ACCENT = "#3B82F6"       # primary accent: highlighted / clean-fuel
ACCENT_ALT = "#F59E0B"   # secondary accent: contrast / negative

INK = "#334155"          # in-chart text and strokes
GREY_DARK = "#64748B"
GREY_MID = "#94A3B8"
GREY_LIGHT = "#CBD5E1"   # muted / de-emphasised marks
GREY_FAINT = "#E2E8F0"

# Fallback colours for values with no explicit palette mapping.
UNKNOWN_COLORS = ["#94A3B8", "#A78BFA", "#5EEAD4", "#FDA4AF", "#BEF264"]

# ---------------------------------------------------------------------------
# Fuel-type vocabulary
# ---------------------------------------------------------------------------

# Clean powertrains present in this dataset (drives CFAR).
CLEAN_FUELS = ["Electric", "CNG", "Hybrid"]

# Dirtiest to cleanest, so stacked bars read in a meaningful order.
FUEL_ORDER = ["Diesel", "Petrol", "CNG", "Hybrid", "Electric"]

FUEL_COLOR = {
    "Diesel": "#64748B",
    "Petrol": "#94A3B8",
    "CNG": "#38BDF8",
    "Hybrid": "#F59E0B",
    "Electric": "#3B82F6",
}
