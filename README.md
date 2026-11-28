# MDIT 2026 - Housing Cost, BMCI Drivers and Overhang Analytics

A data-driven analytics project exploring Malaysian residential cost and
affordability pressures, Building Materials Cost Index (BMCI) movements, and
housing supply-demand mismatch through official public data.

The project combines data preparation, time-series modelling, econometric
checks, and machine-learning scenario analysis for use in Power BI reporting.

## Project structure

```text
.
├── data/
│   ├── raw/          # Original OpenDOSM, BNM, and official source files
│   ├── processed/    # Frequency-aligned and cleaned analytical datasets
│   └── reference/    # Supporting reference data and metadata
├── dashboard/        # Power BI reports and exported visual assets
├── docs/             # Causal analysis plans, notes, and report drafts
├── scripts/
│   ├── data_cleaning.py
│   └── simulation_model.py
└── outputs/
    └── model_ready/  # CSV exports for Power BI and reporting
```

## Analytical workflow

1. Place unmodified official datasets in `data/raw/`.
2. Align frequencies and clean source data under `data/processed/`.
3. Test stationarity and identify relevant time-series lags.
4. Run econometric and causal checks.
5. Build scenario models and export validated results to
   `outputs/model_ready/`.
6. Connect a Power BI report in `dashboard/` to the model-ready CSV files.

## Getting started

1. Add the required public data files to `data/raw/` without modifying them.
2. Use `scripts/data_cleaning.py` to prepare the analytical datasets.
3. Use `scripts/simulation_model.py` for scenario and housing-model analysis.
4. Open or connect the Power BI report in `dashboard/` to the generated outputs.

Source datasets, model outputs, and report assets are project-specific and are
not included in this repository until supplied. Keep source files unchanged
and use the processed and model-ready directories for derived data.
