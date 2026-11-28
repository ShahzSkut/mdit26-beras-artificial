# MDIT 2026: Python AI and Time-Series Causal Analysis Plan

**Project:** `mdit26-housing-bmci` — Malaysian Housing Affordability, Construction Cost Drivers (BMCI/BCI), Monetary Policy (OPR), Construction Activity (QCS), and Residential Overhang / Supply-Demand Dynamics  
**Submission Target:** November 29, 2026

## 1. Objective & Core Analytical Question

How do construction input cost pressures—specifically **Building Materials Cost Index (BMCI)** and **Building Materials Unit Price Index** (`steel` vs. `cement` and other materials), **fuel/diesel adjustments** (`fuelprice.csv`), and **USD/MYR exchange rates** (`exchange-rates.csv`), alongside **construction demand** (DOSM QCS work done) and **interest rates** (`bnm_opr_decisions.csv`)—transmit over time?

Why does cost push **not** map 1-to-1 into higher housing prices or faster sales clearance due to **residential overhang, stock buffers, and affordability constraints**?

## 2. Dataset Inventory & Role Mapping

### A. Primary Monthly Time-Series Drivers (Cost & Macro)

- `bci_building_peninsular.csv` — Monthly Building Materials Cost Index (Peninsular Malaysia) by building category and region.
- `bci_materials_peninsular.csv` — Monthly Building Materials Unit Price Index (Peninsular Malaysia) by material (e.g., steel vs. cement/sand/aggregates) and region.
- `bci_civil_peninsular.csv` — Monthly Civil Engineering Materials Prices (Peninsular Malaysia) for infrastructure/civil comparison.
- `fuelprice.csv` — Fuel/diesel price adjustments (tracks transport and energy-linked cost shifts).
- `exchange-rates.csv` — Exchange rate series (USD/MYR), capturing imported steel, equipment, and commodity FX sensitivity.
- `bnm_opr_decisions.csv` — Bank Negara Malaysia MPC OPR decisions (expanded to a continuous monthly/quarterly policy rate series and policy-change indicator).

### B. Quarterly, Semi-Annual & Structural Context

- `Statistical Tables QCS 2026Q3.xlsx` — DOSM Quarterly Construction Statistics (value of work done, project counts, residential vs. non-residential/civil sub-sectors).
- `Jadual Penerbitan MHPI Q2 2026P.xlsx` & `Jadual Penerbitan MHPI Q3 2026P.xlsx` (plus `MHPI Summary.csv`, `MHPI Summary(1).csv`, `MEDIAN PRICE TREND.csv`) — Malaysian House Price Index and median price trends.
- **NAPIC / JPPH Overhang, Stock & Absorption CSVs:** `napic_existing_stock_q3_2026.csv`, `napic_stock_by_state_stage_q3_2026.csv`, `napic_unsold_analytics_summary.csv`, `napic_newly_launched_absorption.csv`, `napic_residential_prices_by_state.csv`, `napic_table3_transactions_by_price_range.csv`, `napic_quarterly_value_by_sector_price_range.csv`, `napic_status_table_kl_sample.csv`.
- **OpenDOSM & KRI Structural Context:** `household_vs_livingquarter.csv`, `hh_income.csv`, `hh_inequality.csv`, `hies_state.csv`, `wages_real_state.csv`, and the KRI vacancy/overhang brief.

> **Note on Coal & Energy:** If a dedicated global thermal coal or electricity index CSV is not yet in the folder, explicitly note in the pipeline that `fuelprice.csv` (diesel/RON95/fuel adjustments) and `exchange-rates.csv` act as observed domestic/FX cost channels, while global coal/energy prices are flagged as a secondary external covariate or omitted confounder to add later if available.

## 3. Step-by-Step Python Modeling Workflow

### Step 1: Ingestion, Date Standardization & Frequency Alignment (`src/01_align_data.py`)

1. Parse timestamps to `YYYY-MM` (monthly) and `YYYY-Qn` (quarterly) without inventing artificial intra-quarter movements.
2. Handle OpenDOSM index rebasing by computing **log differences** ($\Delta \ln X_t = \ln X_t - \ln X_{t-1}$) or **YoY percentage changes** within consistent base/series segments rather than splicing raw index levels across incompatible base eras.
3. Expand `bnm_opr_decisions.csv` (event/decision dates) into a clean continuous **monthly/quarterly OPR level** (`opr_pct`) and **OPR change** (`d_opr_bps`) series.
4. Build two clean analytical panels in `data/processed/`:
   - **Monthly Cost & Macro Panel (`panel_monthly.csv`):** BMCI building/material sub-series (steel vs. cement/aggregates), fuel/diesel, USD/MYR, and OPR.
   - **Quarterly Transmission & Overhang Panel (`panel_quarterly.csv`):** Quarterly-aggregated BMCI/material growth, QCS construction work done, MHPI/price indicators, OPR, and NAPIC unsold/overhang/stock indicators.

### Step 2: Stationarity, Lag Structure & Causal / Econometric Diagnostics (`src/02_stationarity_lags.py`)

1. Run **Augmented Dickey-Fuller (ADF)** and **KPSS** tests on levels vs. first/log differences to avoid spurious regressions on trending series.
2. Compute **cross-correlation functions (CCF)** across lags ($k = 0, 1, 2, \dots, 12$ months or $1 \dots 6$ quarters) for:
   - Fuel/diesel & USD/MYR -> Steel vs. Cement unit price indices -> Composite BMCI.
   - BMCI & OPR -> Construction work done (QCS) -> Housing prices (MHPI) & overhang/absorption.
3. Estimate **lagged / distributed-lag / ARDL / VAR** models with **Newey-West (HAC) standard errors** and run **Granger causality** checks (treating Granger causality strictly as *temporal predictive precedence*, not structural proof).
4. Compute **Local Projections (Jordà LP)** or Impulse Response summaries to show cumulative multi-period responses (such as how quickly steel reacts vs. cement or construction activity).

### Step 3: Machine Learning / AI Feature Importance, Regime & Scenario Modeling (`src/03_ml_scenarios.py`)

1. Use **time-series cross-validation** (`TimeSeriesSplit` or expanding/rolling window—**never** random k-fold splits on time series) to prevent data leakage.
2. Compare a regularized linear baseline (**Ridge / Lasso**) against non-linear AI models (**Random Forest / Gradient Boosting**) to test whether non-linearities or threshold effects matter for BMCI growth and construction demand/overhang dynamics.
3. Extract **permutation / model feature importance** (comparing steel vs. cement, fuel/diesel, USD/MYR, OPR, and lag terms) and run **transparent scenario stress tests** (fuel/FX/material shocks) paired with **overhang & affordability frictions** (NAPIC unsold units, absorption rates, median multiple / household income context).

### Step 4: Diagnostics, Overhang Synthesis & Reporting Outputs (`src/04_summary_tables.py`)

1. Export clean CSV/JSON summary tables (`outputs/stationarity_results.csv`, `outputs/lag_correlations.csv`, `outputs/model_metrics.csv`, `outputs/scenario_attribution.csv`) for your report and Power BI dashboard.
2. Document every data limitation, frequency mismatch, clipped PDF export, and observational caveat clearly.

## 4. Why Observational Data Requires Caution on True Cause and Effect

In observational macroeconomic and housing data, strong correlations or significant lag coefficients do **not** automatically prove true structural causality due to five core risks:

1. **Omitted Confounders:** Global steel/commodity cycles, international shipping/freight rates, domestic supply-chain bottlenecks, labor availability, and fiscal/housing policies (such as stamp-duty exemptions or targeted subsidies) can move input costs, construction activity, and housing demand simultaneously.
2. **Simultaneity & Endogeneity (Reverse Causality / Anticipation):** Construction demand and developer expectations can influence ordering/pricing decisions, while contractors and buyers often react *in anticipation* of fuel adjustments, exchange-rate swings, or OPR moves rather than strictly afterward.
3. **Non-Stationarity, Rebasing & Structural Breaks:** Price indices and macro series often trend or experience regime shifts (e.g., COVID-19 disruptions, fuel subsidy/float changes, OPR hiking/easing cycles, or OpenDOSM base-year changes). Unadjusted trends or unmodeled breaks can produce misleading relationships.
4. **Aggregation & Frequency Mismatch:** Mixing monthly cost series (`bci_building_peninsular.csv`, `fuelprice.csv`, `exchange-rates.csv`) with quarterly activity/price series (`QCS`, `MHPI`) or semi-annual/snapshot NAPIC tables (`H1/H2`, `Q3 2026` snapshots) can mask intra-period timing or average out sharp spikes.
5. **Overhang, Stock Buffers & Affordability Frictions:** Even when BMCI or steel/cement costs rise, developers cannot always pass costs straight into higher transaction prices or faster sales because:
   - High existing/unsold inventory and overhang (completed/under-construction unsold units in specific property types and price brackets) create supply buffers and pricing pressure.
   - Household income and affordability constraints (elevated median multiples and borrowing costs) cap buyer willingness/ability to pay, squeezing developer margins, delaying launches/completions, or slowing absorption instead of immediately raising sold prices or sales velocity.

## 5. Immediate Next Action

Proceed to **Step 1 (`src/01_align_data.py`)**: inspect the exact schemas and columns of your core CSV and Excel files in the workspace and build a clean, defensive alignment script that outputs `data/processed/panel_monthly.csv` and `data/processed/panel_quarterly.csv` without guessing missing columns.