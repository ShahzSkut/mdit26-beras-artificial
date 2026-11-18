# Zakat Economic Simulator

A data-driven decision-support and micro-simulation dashboard exploring how zakat
can support sustainable economic growth for B40 households.

## Project structure

```text
.
├── data/
│   ├── raw/          # Original datasets (for example, OpenDOSM HIES and PLI)
│   ├── processed/    # Cleaned datasets for simulation and Power BI
│   └── reference/    # Zakat expenditure benchmarks and asnaf breakdowns
├── dashboard/        # Power BI dashboard
├── docs/             # Project proposal, diagrams, and reports
└── scripts/
    ├── data_cleaning.py
    └── simulation_model.py
```

## Getting started

1. Put original datasets in `data/raw/` and keep them unchanged.
2. Put reference materials in `data/reference/`.
3. Use `scripts/data_cleaning.py` to prepare datasets for analysis and save
   generated files under `data/processed/`.
4. Use `scripts/simulation_model.py` for the household income-growth
   simulation.
5. Open the Power BI report in `dashboard/` to explore the results.

The source datasets, simulation inputs, and report assets are project-specific
and are not included in this repository. Add them to their corresponding
directories when available.
