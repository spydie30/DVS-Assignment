# TASK: Refactor and Update Existing Streamlit Dashboard for VAHAN RTO Analytics

## 1. Context & Dataset Schema
We are updating our existing Streamlit application to provide advanced executive analytics, compliance tracking, and data governance on Indian VAHAN RTO vehicle registration data.

The dataset contains the following 17 columns:
- Identification & Time: `Registration_Number`, `Registration_Date`, `Registration_Year`
- Geographic: `State`, `RTO_Office`
- Vehicle Specs: `Vehicle_Category`, `Vehicle_Sub_Type`, `Manufacturer_Brand`, `Engine_CC`, `Seating_Capacity`, `Vehicle_Age_Years`
- Fuel & Emission: `Fuel_Type`, `Emission_Norm`
- Data Quality & Compliance Flags: `is_clean` (Boolean), `is_compliant` (Boolean)
- Advanced Metrics: `CFAR` (Numeric/Float), `FMI` (Numeric/Float)

---

## 2. Core Business Logic & Metric Definitions

Implement/Ensure the following calculation functions exist in python:

1. Data Partitioning:
   - `analytics_df = df[df['is_clean'] == True]` (Use for Tabs 1 & 2)
   - `governance_df = df[df['is_clean'] == False]` (Use for Tab 3)

2. Clean Fuel Adoption Rate (CFAR):
   - Definition: Percentage of vehicles powered by Electric, CNG, Hybrid, or Petrol/CNG relative to total registrations.
   - Formula: `CFAR (%) = (Count of Clean Fuel Vehicles / Total Vehicles) * 100`
   - Clean Fuels list: `['ELECTRIC', 'ELECTRIC(BOV)', 'CNG', 'PETROL/CNG', 'HYBRID', 'HYBRID ELECTRIC', 'STRONG HYBRID', 'LPG']`

3. Fuel Mix Index (FMI):
   - Definition: Diversification score based on Herfindahl-Hirschman Diversity Index across powertrain fuel types.
   - Formula: `FMI = 1 - sum((Market_Share_of_Fuel_i)^2)`
   - Score Range: 0.0 (Monopolistic / Single Fuel) to ~0.75+ (Highly Balanced Mix)

---

## 3. Sidebar Controls & Dynamic Filters

Add/Update a multi-level sidebar filter that dynamically affects the dashboard:
1. Date Range / Year Selector: `Registration_Year` slider or multiselect.
2. Location Selector: `State` dropdown (updates available `RTO_Office` options dynamically).
3. Classification Filters: `Vehicle_Category` and `Vehicle_Sub_Type` multiselect.
4. OEM Selector: `Manufacturer_Brand` multiselect.
5. Global Data Quality Toggle: "Include Unclean Rows in Analytics" checkbox (default = False).

---

## 4. Streamlit UI Architecture & Tab Structure

Organize the dashboard layout into 3 distinct tabs using `st.tabs(["📊 Macro Fuel Transition", "🏎️ OEM & Powertrain Strategy", "🚨 Regulatory & Data Quality Audit"])`:

### TAB 1: 📊 Macro Fuel Transition (CFAR & Market Shifts)
- Top KPI Cards (`st.metric`):
  1. Total Retails Volume (with MoM / YoY % change delta if date filtered)
  2. National / State `CFAR` (%)
  3. Average Fleet `FMI` Score
  4. Non-Compliant Fleet Share (`is_compliant == False` %)
- Chart 1 (Main): Plotly 100% Stacked Area / Line Chart showing `Fuel_Type` market share trajectory over `Registration_Year` / `Registration_Date`.
- Chart 2: Plotly Choropleth or Grouped Bar Chart ranking `State` or `RTO_Office` by `CFAR` score (Highlights clean fuel adoption hotspots).
- Chart 3: Waterfall or Donut chart showing the net basis point (bps) market share shift across fuels.

### TAB 2: 🏎️ OEM & Powertrain Strategy
- Top KPI Cards: Top Performing OEM by Volume, Highest CFAR OEM, Most Diversified OEM (Highest FMI).
- Chart 1: Interactive Scatter Plot
  - X-Axis: `CFAR` (% Clean Fuel Share)
  - Y-Axis: `FMI` (Fuel Mix Index)
  - Size: Total Registration Volume
  - Color: `Manufacturer_Brand`
  - Annotations: Four quadrants ("Green Pioneers", "Diversified Leaders", "ICE Dependent", "Niche Players").
- Chart 2: 100% Stacked Horizontal Bar Chart comparing fuel mix across `Manufacturer_Brand` (Maruti Suzuki, Tata Motors, Mahindra, Toyota, JSW MG, etc.).
- Chart 3: Boxplot / Violin plot showing `Engine_CC` distribution across `Vehicle_Sub_Type` vs `Fuel_Type`.

### TAB 3: 🚨 Regulatory & Data Quality Audit
- Section A: Compliance & Scrappage Risk (`is_compliant == False`)
  - Heatmap Matrix: `Vehicle_Age_Years` vs `Emission_Norm` (BS-III, BS-IV, BS-VI) highlighting aging vehicles subject to scrappage laws.
  - Expired / Risk Fleet Table: Filterable list showing non-compliant vehicles by `RTO_Office`.
- Section B: Data Governance (`is_clean == False`)
  - KPI: Data Cleanliness Score (`(Clean Rows / Total Rows) * 100`).
  - Bar Chart: Ranking `RTO_Office` locations by percentage of erroneous data records.
  - Data Hygiene Drill-down: `st.dataframe` showing raw error rows (e.g., Hatchbacks with `Seating_Capacity > 5`, EVs with `Engine_CC > 0`) for audit export.

---

## 5. Technical Requirements & Streamlit Code Guidelines

1. Performance Optimization:
   - Wrap heavy data transformation routines in `@st.cache_data`.
   - Use PyArrow engine for fast pandas DataFrame operations.
2. Plotly Visualizations:
   - Use `plotly.express` or `plotly.graph_objects` with a clean modern theme (`template="plotly_white"`).
   - Ensure all charts render responsively using `use_container_width=True`.
3. Error Handling:
   - Add graceful empty-state handling (`st.warning("No data available for the selected filters.")`) if sidebar filters return 0 rows.
4. Export Feature:
   - Include a CSV download button (`st.download_button`) on tables so business users can export the filtered view.

Please generate the updated Streamlit app script (`app.py` or modular layout structure) incorporating these specifications.