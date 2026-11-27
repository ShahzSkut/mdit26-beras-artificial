# Data handover: Malaysian housing affordability model

Problem statement: *From Fuel Shocks to Foundation Stones: quantifying how energy and material cost inflation transmit to Malaysian housing affordability.*

Everything here is produced by `scripts/process_data.py` from the raw files in `data/`. Re-run it with `python scripts/process_data.py` (pandas, numpy and openpyxl only). Every cleaning decision is logged in `reports/data_quality_log.md`, and `reports/output_profile.csv` gives a column-level profile (nulls, min/max) of every output.

## Which file should I start with?

| Goal | File (in `data/processed/model_ready/`) | Grain | Span | Rows |
|---|---|---|---|---|
| Cost-shock transmission, national time series | `quarterly_master.csv` | quarter | 2010-Q1 to 2026-Q3 | 67 |
| Panel model, most observations | `state_quarterly_panel.csv` | state x quarter | 2010-Q1 to 2026-Q1 | 910 |
| High-frequency drivers (fuel, FX, OPR, BMCI, CPI, PPI) | `monthly_master.csv` | month | 2010-01 to 2026-10 | 202 |
| Affordability and income vs price growth | `annual_national.csv`, `annual_state_panel.csv` | year | 2010 to 2025 | 16 / 240 |
| B40 / M40 / T20 affordability | `affordability_by_income_group.csv` | geo x year x group | 2019, 2022, 2024 | 60 |

Long, tidy tables in `tidy/` are suited to Power BI relationships. One-off context tables are in `reference/`.

## Column naming conventions

- `fuel_*` is pump price in RM/litre. `fuel_*_mom_pct` and `*_yoy_pct` are % changes of the monthly mean.
- `fx_myr_per_*` is MYR per 1 unit of foreign currency (per 100 for JPY and HKD).
- `bnm_opr_pct` is the Overnight Policy Rate in %.
- `bmci_*` is the Building Material Cost Index by material (national simple mean of 6 regions; no regional weights were provided).
  - `bmci_central_*` is the Selangor / KL / Melaka / N. Sembilan region.
  - `bmci_bld_<building>_{excl,incl}_steel` is the whole-building cost index.
  - **`bmci_main` is the designated main BMCI series** (national RC 2-4 storey flat-roof building, including steel). In `state_quarterly_panel.csv` it is the state's regional version. Reasons: complete 2010-2026 history with no gaps, represents low-rise landed housing, and includes steel. The other columns are sensitivity checks (`..._excl_steel`, `bmci_central_*`, and the material indices `bmci_cement` / `bmci_steel` that only start 2019-12).
- `price_*` is civil-engineering material prices in RM. Units are cement per 50 kg bag, steel and aggregates per tonne, and BRC per kg. `price_central_*` is the Central region only.
- `cpi_*` is the mean of 4 states (Melaka, N. Sembilan, Selangor, KL).
- `ppi_*` is the aggregate Producer Price Index.
- `hpi_<type>_{index,price_rm,yoy_pct,qoq_pct}` is the NAPIC Malaysian House Price Index (2010 = 100).
  - Types are `all_house`, `terraced`, `high_rise`, `semi_detached` and `detached`.
- `income_median_est` is monthly household median income in RM, with provenance in `income_median_source`.
- `median_multiple_*` is price / (12 x monthly income).

## Read before modelling

1. **The usable window is short.**
   - Fuel starts 2017-03 and FX 2015-01.
   - Material-level BMCI cement and steel start 2019-12, and ready-mix concrete starts 2023-01.
   - With the core drivers as YoY changes, the national quarterly model has 34 complete quarters (2018-Q2 to 2026-Q3), or 30 with four-quarter lags (2019-Q2 to 2026-Q3).
   - Use `state_quarterly_panel.csv` for a larger N (about 500 complete state-quarters on the core columns).
   - Cement and steel back to 2008 are available as `price_cement` and `price_steel_*`.
2. **Income is annual and partly estimated.** HIES exists only for survey years. Values in between are log-linear interpolations, and 2025-26 are extrapolated using the 2022-24 growth rate. Filter on `income_median_source == "survey"` if you need only real observations.
3. **Affordability price level.** The MHPI "price" is an index-scaled level, higher than the median transaction price.
   - `median_multiple_mhpi` is about 5.7-6.9x nationally.
   - `median_multiple_txn` (annual_national, 2022+) uses NAPIC median transaction prices and gives 4.3x for 2022, matching the KRI paper.
   - Use the MHPI-based ratio for trends and the transaction-based one for levels.
4. **Fuel structural breaks.**
   - Headline `ron95` is stable at 2.05 until 2025, then jumps after BUDI95 (2025-09-30).
   - Citizens' effective price is `fuel_ron95_effective` (BUDI95 RM1.99 from Sept 2025).
   - Diesel has SKDS and BUDI programme prices (`fuel_diesel_skds`, `fuel_diesel_budi`).
   - Consider regime dummies.
5. **Partial and preliminary data.** 2026-Q3 all-house HPI is preliminary (`hpi_is_preliminary`). The modelling window ends 2026-10-01 (the partial Nov-2026 month was removed). By-type and state HPI stop at 2026-Q1.
6. **Vintage difference.** All-house 2026-Q1 is 235.9 in `quarterly_master` (newer xlsx) but 235.2 in the by-type/state CSV.
7. **Rows that are structurally missing, not errors.**
   - High-rise HPI does not exist for 7 states, and detached does not exist for Perlis.
   - Sabah and Sarawak have no BMCI, so the national mean is used (`bmci_is_national_fallback = True`).
   - CPI covers only 4 states, with no national CPI.
8. **Real vs nominal.** `wage_*_real_rm` in the annual panel is in constant prices. Prices and household income are nominal.
9. **Sabah and Sarawak ratios look high** (about 8x, versus 4-6x elsewhere). This comes straight from the source MHPI prices and HIES income, so please sanity-check it before relying on it.

## Gaps that cannot be filled from the collected files

- **Overhang time series.** None was collected. The PDF gives only 2022 national overhang and 2020 vacancy (`reference/overhang_vacancy_facts_kri_2023.csv`). Census-based vacancy proxy is in `households_vs_living_quarters_census.csv`.
- **Electricity tariff and logistics cost.** Not in the files. Only pump fuel is available.
- **Carbon and green building benchmarks.** Only national GHG totals by source are available (`ghg_emissions_annual.csv`, up to 2023). No cement or steel emission-intensity benchmarks and no Green Building Index data.
- **B40/M40 time series.** Only three survey points (2019, 2022, 2024), and only for Malaysia plus 4 states.
- **Wage data.** Formal-sector monthly wages exist for 4 states only (2022-01 to 2026-06).

## Files not used

- `napic_kuala_lumpur_residential_property_price_range_status_sample.csv` is a 17-row PDF sample with mostly zeros.
- `DATASETS/BMCI_RC_2_4_Storey...csv` is a subset of the full building index file.
- `fx` blocks 2 and 3 (THB, PHP, TWD, VND, KHR, NZD and similar) were stacked under the first block in the same file and were discarded. They are not MYR per USD, and USD there is ~12.
- The CSV exports of the HPI qoq/yoy growth tables are redundant with the state x type file.
