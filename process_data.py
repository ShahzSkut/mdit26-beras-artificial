#!/usr/bin/env python3
"""
process_data.py  --  Housing_Affordability_Project data pipeline
=================================================================
Reads the raw collected files under  data/  and writes cleaned, harmonised,
model-ready tables to  data/processed/  plus QA logs to  reports/.

Run from anywhere:
    python scripts/process_data.py

Only needs pandas, numpy and openpyxl.

Output layout
    data/processed/model_ready/   monthly_master, quarterly_master, state_quarterly_panel,
                                  annual_national, annual_state_panel
    data/processed/tidy/          long/tidy tables (good for Power BI relationships)
    data/processed/reference/     one-off context tables (overhang, stock, price distribution ...)
    reports/data_quality_log.md   every cleaning decision / issue found
    reports/output_profile.csv    column-level profile of every output file
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DS = DATA / "DATASETS"
HF = DATA / "High-Frequency Macro & Input Cost Shocks"
MO = DATA / "Monthly Cost & Material Shocks"
QU = DATA / "Quarterly Construction Demand, Prices & Overhang"
ST = DATA / "Structural Demand, Income & Overhang Context"
OUT = DATA / "processed"
REP = ROOT / "reports"
for d in (OUT / "model_ready", OUT / "tidy", OUT / "reference", REP):
    d.mkdir(parents=True, exist_ok=True)

LAST_MONTH = "2026-10-01"             # modelling window end (Nov-2026 is a partial month)
LOG: list[tuple[str, str]] = []          # (area, message)
OUTPUTS: list[Path] = []


def log(area: str, msg: str) -> None:
    LOG.append((area, msg))
    print(f"[{area}] {msg}")


def save(df: pd.DataFrame, sub: str, name: str) -> pd.DataFrame:
    p = OUT / sub / name
    df.to_csv(p, index=False, float_format="%.6g", date_format="%Y-%m-%d")
    OUTPUTS.append(p)
    print(f"   -> {sub}/{name}  {df.shape}")
    return df


# --------------------------------------------------------------------------- helpers
def read_u16(path: Path, **kw) -> pd.DataFrame:
    """Several exports are UTF-16 tab-separated files (Tableau/Excel 'Unicode text')."""
    return pd.read_csv(path, encoding="utf-16", sep="\t", **kw)


def to_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    s = s.replace({"-": np.nan, "": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})
    return pd.to_numeric(s, errors="coerce")


STATE_MAP = {
    "W.P. Kuala Lumpur": "Kuala Lumpur", "WP Kuala Lumpur": "Kuala Lumpur",
    "W.P. Labuan": "Labuan", "WP Labuan": "Labuan",
    "W.P. Putrajaya": "Putrajaya", "WP Putrajaya": "Putrajaya",
    "Penang": "Pulau Pinang", "Pulau Penang": "Pulau Pinang",
}


def std_state(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.replace(STATE_MAP)


def qstart(label: pd.Series) -> pd.Series:
    """'2010 Q1' -> Timestamp 2010-01-01"""
    y = label.str.extract(r"(\d{4})\s*-?\s*Q([1-4])")
    return pd.to_datetime(dict(year=y[0].astype(int), month=(y[1].astype(int) - 1) * 3 + 1, day=1))


def qlabel(ts: pd.Series) -> pd.Series:
    return ts.dt.year.astype(str) + "-Q" + ts.dt.quarter.astype(str)


def qmean(monthly: pd.DataFrame, min_months: int = 3) -> pd.DataFrame:
    """Quarterly mean of a monthly frame; NaN unless the quarter has all `min_months` months."""
    g = monthly.resample("QS")
    return g.mean().where(g.count() >= min_months)


# =========================================================================== 1. FUEL
def build_fuel() -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_csv(DS / "fuelPriceClean.csv")
    f["date"] = pd.to_datetime(f["date"], dayfirst=True)
    f = f.sort_values("date").drop(columns="series_type").reset_index(drop=True)
    cols = ["ron95", "ron97", "diesel", "diesel_eastmsia", "ron95_budi95", "ron95_skps",
            "diesel_skds", "diesel_budi"]
    # Effective RON95 for a Malaysian citizen: BUDI95 targeted price once it exists, else the headline price
    f["ron95_effective"] = f["ron95_budi95"].fillna(f["ron95"])
    for c in ("ron95", "ron97", "diesel", "ron95_effective"):
        f[f"{c}_wk_change"] = f[c].diff()
    save(f, "tidy", "fuel_weekly_clean.csv")
    log("fuel", f"weekly series {f.date.min().date()}..{f.date.max().date()} ({len(f)} rows); "
                "fuelPriceClean.csv verified identical to the raw weekly 'level' rows.")
    log("fuel", "BUDI95 (from 2025-09-30), SKDS diesel (from 2024-06), diesel_budi (from 2026-07) are "
                "targeted/subsidised programme prices - structural breaks, NOT comparable to headline ron95/diesel. "
                "Use ron95_effective only with that caveat.")

    # Pump prices are administered step prices: a price holds until the next announcement. Forward-fill to a daily
    # grid and average -> time-weighted monthly mean (also fills months with no weekly observation, e.g. 2018-09).
    daily = f.set_index("date")[cols + ["ron95_effective"]].resample("D").ffill()
    m = daily.resample("MS").mean()
    m["fuel_n_weeks"] = f.set_index("date")["ron95"].resample("MS").count()
    log("fuel", "monthly values = time-weighted mean of the daily forward-filled weekly price (price persists until the next "
                "weekly observation). Sept-2018 has no weekly observation (gap 2018-08-30 -> 2018-10-01): filled by this carry-forward; "
                "fuel_n_weeks shows the true number of observations (0 for that month). Subsidised-programme columns stay NaN "
                "before their start dates.")
    for c in ("ron95", "ron97", "diesel", "ron95_effective"):
        m[f"{c}_mom_pct"] = m[c].pct_change() * 100
        m[f"{c}_yoy_pct"] = m[c].pct_change(12) * 100
    m = m.add_prefix("fuel_").rename(columns={"fuel_fuel_n_weeks": "fuel_n_weeks"})
    m.index.name = "month"
    return f, m


# =========================================================================== 2. FX
def build_fx() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(HF / "daily_foreign_currency_exchange_rates.csv", dtype=str)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    blank = raw.index[raw.iloc[:, 1:].isna().all(axis=1)]
    first_block = raw.iloc[: blank[0]].copy() if len(blank) else raw.copy()
    log("fx", f"file contains {len(blank)+1} stacked blocks (blank row + new header between them: THB/PHP/TWD.., "
              f"VND/KHR/NZD..). Only block 1 (rows 1-{len(first_block)}, USD..HKD100 = MYR per unit) kept; "
              "other blocks use different currencies/units and reuse the same column names.")
    fx = first_block
    fx["date"] = pd.to_datetime(fx["date"], format="%d %b %Y")
    num_cols = [c for c in fx.columns if c != "date"]
    for c in num_cols:
        fx[c] = pd.to_numeric(fx[c], errors="coerce")
    dup = fx["date"].duplicated(keep="last").sum()
    fx = fx.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if dup:
        log("fx", f"dropped {dup} duplicated date row(s) (kept last).")
    assert fx["USD"].between(3, 5.5).all(), "USD/MYR outside plausible range - block parsing failed"
    ren = {"USD": "myr_per_usd", "GBP": "myr_per_gbp", "EUR": "myr_per_eur", "JPY100": "myr_per_100jpy",
           "CHF": "myr_per_chf", "AUD": "myr_per_aud", "CAD": "myr_per_cad", "SGD": "myr_per_sgd",
           "HKD100": "myr_per_100hkd"}
    fx = fx.rename(columns=ren)
    save(fx, "tidy", "fx_daily_clean.csv")
    cols = list(ren.values())
    m = fx.set_index("date")[cols].resample("MS").mean().add_prefix("fx_")
    m["fx_myr_per_usd_eom"] = fx.set_index("date")["myr_per_usd"].resample("MS").last()
    m["fx_myr_per_usd_mom_pct"] = m["fx_myr_per_usd"].pct_change() * 100
    m["fx_myr_per_usd_yoy_pct"] = m["fx_myr_per_usd"].pct_change(12) * 100
    m.index.name = "month"
    return fx, m


# =========================================================================== 3. OPR
def build_opr() -> pd.DataFrame:
    o = pd.read_csv(HF / "bnm_overnight_policy_rate_decisions.csv", parse_dates=["decision_date", "month_start"])
    g = o.groupby("month_start").agg(opr_pct=("opr_level_pct", "last"), opr_change_pct=("change_opr_pct", "sum"))
    idx = pd.date_range(g.index.min(), LAST_MONTH, freq="MS")
    m = g.reindex(idx)
    m["opr_pct"] = m["opr_pct"].ffill()
    m["opr_change_pct"] = m["opr_change_pct"].fillna(0.0)
    m.index.name = "month"
    log("opr", "OPR decisions forward-filled to a monthly step series (level persists until next MPC decision).")
    return m.add_prefix("bnm_").rename(columns={"bnm_opr_pct": "bnm_opr_pct", "bnm_opr_change_pct": "bnm_opr_change_pct"})


# =========================================================================== 4. BUILDING COST INDICES
CENTRAL = "Selangor, Kuala Lumpur, Melaka & Negeri Sembilan"
OVERLAP = "Selangor & Kuala Lumpur"     # overlaps CENTRAL -> excluded from national averages
CORE_MATERIALS = ["cement", "steel", "sand", "aggregates", "ready_mix_concrete", "timber", "bricks_wall"]


def build_bmci():
    mat = pd.read_csv(MO / "monthly_building_cost_index_by_material_and_region_peninsular_malaysia.csv.csv",
                      parse_dates=["date"])
    bld = pd.read_csv(MO / "monthly_building_cost_index_by_building_and_region_peninsular_malaysia.csv",
                      parse_dates=["date"])
    assert not mat.duplicated(["date", "material", "region"]).any()
    assert not bld.duplicated(["date", "building", "region"]).any()
    mat["region_overlaps_central"] = mat["region"].eq(OVERLAP)
    save(mat.rename(columns={"date": "month", "index": "bmci_index"}), "tidy",
         "bmci_material_monthly_long.csv")
    save(bld.rename(columns={"date": "month"}), "tidy", "bmci_building_monthly_long.csv")
    log("bmci", f"region '{OVERLAP}' exists only in the material file and overlaps '{CENTRAL}' -> "
                "excluded from the national (unweighted 6-region) mean to avoid double counting. "
                "No regional weights are provided, so national = simple mean of regions.")
    log("bmci", "material-level 'cement' and 'steel' indices only exist from 2019-12 and 'ready_mix_concrete' from 2023-01 "
                "(source limitation). For longer cement/steel history use the civil-engineering price columns "
                "(price_cement, price_steel_*; from 2008) or the building-level indices (from 2008-07).")
    log("bmci", "index_incl_steel is blank before 2008-12 and for a few later months; left as NaN (not imputed).")
    log("bmci", "DATASETS/BMCI_RC_2_4_Storey_Selangor_KL_Melaka_N9.csv is a subset of the building file "
                "(same 220 rows) and is not used separately.")

    m6 = mat[mat.region != OVERLAP]
    nat_mat = m6.groupby(["date", "material"])["index"].mean().unstack().add_prefix("bmci_")
    cen_mat = (mat[mat.region == CENTRAL].pivot(index="date", columns="material", values="index")
               [CORE_MATERIALS].add_prefix("bmci_central_"))
    nat_b_ex = bld.groupby(["date", "building"])["index_excl_steel"].mean().unstack().add_prefix("bmci_bld_").add_suffix("_excl_steel")
    nat_b_in = bld.groupby(["date", "building"])["index_incl_steel"].mean().unstack().add_prefix("bmci_bld_").add_suffix("_incl_steel")
    cen_b = bld[(bld.region == CENTRAL) & (bld.building == "rc_2_4_storey_flat_roof")].set_index("date")[
        ["index_excl_steel", "index_incl_steel"]].add_prefix("bmci_central_rc_2_4_flat_")
    monthly = pd.concat([nat_mat, cen_mat, nat_b_ex, nat_b_in, cen_b], axis=1)
    monthly.index.name = "month"

    # regional monthly frame for the state panel
    reg = mat[CORE_MATERIALS[:5] and mat.material.isin(["cement", "steel", "sand", "aggregates", "ready_mix_concrete"])]
    reg = reg.pivot_table(index=["date", "region"], columns="material", values="index").add_prefix("bmci_region_")
    regb = bld[bld.building == "rc_2_4_storey_flat_roof"].set_index(["date", "region"])[
        ["index_incl_steel", "index_excl_steel"]].add_prefix("bmci_region_rc_2_4_flat_")
    regional = reg.join(regb, how="outer").reset_index()
    return monthly, regional


# =========================================================================== 5. CIVIL ENGINEERING PRICES
CIVIL_KEY = ["cement", "sand_river", "sand_mining", "crusher_run", "granite_20mm", "steel_ht_12mm",
             "steel_ht_16_25mm", "steel_ms_12mm", "brc_a8", "bitumen_60_70", "premix_ac14"]


def build_civil() -> pd.DataFrame:
    c = pd.read_csv(MO / "monthly_civil_engineering_material_prices_peninsular_malaysia.csv.csv", parse_dates=["date"])
    assert not c.duplicated(["date", "item", "region"]).any()
    save(c.rename(columns={"date": "month", "price": "price_rm"}), "tidy", "civil_material_prices_monthly_long.csv")
    log("civil", "prices are quoted for 3 regions (Central/East/North; bitumen = Peninsular). 'National' columns "
                 "= simple mean of the quoted regions; 'central' = Central only. Units differ per item (see dictionary).")
    k = c[c["item"].isin(CIVIL_KEY)]
    nat = k.groupby(["date", "item"])["price"].mean().unstack().add_prefix("price_")
    cen = (c[(c.region == "Central") & c["item"].isin(["cement", "steel_ht_12mm", "sand_river", "crusher_run"])]
           .pivot(index="date", columns="item", values="price").add_prefix("price_central_"))
    out = pd.concat([nat, cen], axis=1)
    out.index.name = "month"
    return out


# =========================================================================== 6. CPI / PPI / WAGES
DIV = {"overall": "overall", "1": "food_non_alcoholic_beverages", "2": "alcohol_tobacco", "3": "clothing_footwear",
       "4": "housing_water_electricity_gas_fuels", "5": "furnishings_household_equipment", "6": "health",
       "7": "transport", "8": "information_communication", "9": "recreation_sport_culture", "10": "education",
       "11": "restaurants_accommodation", "12": "insurance_financial", "13": "personal_care_social_protection_misc"}


def build_cpi() -> pd.DataFrame:
    idx = pd.read_csv(DS / "Monthly CPI by State & Division.csv", dtype={"division": str})
    inf = pd.read_csv(DS / "Monthly CPI Inflation by State & Division.csv", dtype={"division": str})
    for d in (idx, inf):
        d["date"] = pd.to_datetime(d["date"], dayfirst=True)
        d["state"] = std_state(d["state"])
    assert not idx.duplicated(["state", "date", "division"]).any()
    long = idx.merge(inf, on=["state", "date", "division"], how="outer")
    long["division_name"] = long["division"].map(DIV)
    long = long.rename(columns={"date": "month", "index": "cpi_index"}).sort_values(["state", "division", "month"])
    save(long[["state", "month", "division", "division_name", "cpi_index", "inflation_yoy", "inflation_mom"]],
         "tidy", "cpi_state_division_monthly_long.csv")
    log("cpi", "CPI covers only 4 states (Melaka, Negeri Sembilan, Selangor, Kuala Lumpur) - no national CPI in the data. "
               "Master tables use the simple mean of these 4 states as a 'central region' proxy.")
    log("cpi", "division names ASSUMED to follow COICOP-2018 ordering (4 = housing/water/electricity/gas, 7 = transport); "
               "please confirm against the DOSM source.")
    p = idx.pivot_table(index="date", columns="division", values="index", aggfunc="mean")
    out = pd.DataFrame({"cpi_overall": p["overall"], "cpi_housing_utilities": p["4"], "cpi_transport": p["7"]})
    for c in list(out.columns):
        out[f"{c}_yoy_pct"] = out[c].pct_change(12) * 100
    out.index.name = "month"
    return out


def build_ppi() -> pd.DataFrame:
    p = pd.read_csv(DS / "ppi(producerpriceindex).csv", parse_dates=["date"])
    a = p[p.series == "abs"].set_index("date")
    out = pd.DataFrame({
        "ppi_index": a["index"], "ppi_index_sa": a["index_sa"],
        "ppi_yoy_pct": p[p.series == "growth_yoy"].set_index("date")["index"],
        "ppi_mom_pct": p[p.series == "growth_mom"].set_index("date")["index"]})
    out.index.name = "month"
    log("ppi", "PPI file has no sector/commodity label (single aggregate series). SA index only available from 2017-01.")
    return out


def build_formal_wages() -> pd.DataFrame:
    w = pd.read_csv(DS / "formal_wages_state_CLEAN.csv")
    w["date"] = pd.to_datetime(w["date"], dayfirst=True)
    w["state"] = std_state(w["state"])
    save(w.rename(columns={"date": "month"}), "tidy", "formal_wages_state_monthly.csv")
    g = w.groupby("date").agg(formal_wage_median_avg_4states=("wages_median", "mean"),
                              formal_employees_4states=("employees", "sum"))
    g.index.name = "month"
    log("wages", "formal-sector wages are monthly for 4 states only, Jan-2022..Jun-2026. Master column is the mean of the "
                 "4 state medians (not a true pooled median).")
    return g


# =========================================================================== 7. HPI (quarterly)
def parse_change_sheet(path: Path) -> pd.DataFrame:
    ws = openpyxl.load_workbook(path, read_only=True, data_only=True)["CHANGE"]
    rows, year = [], None
    for r in ws.iter_rows(values_only=True):
        v = [c for c in r if c is not None]
        if not v:
            continue
        if isinstance(v[0], (int, float)) and 1990 <= v[0] <= 2100:
            year = int(v[0])
            label, vals = (str(v[1]).strip() if len(v) > 1 else ""), v[2:]
            if "Q1 - Q4" in label:           # annual summary row
                continue
        elif isinstance(v[0], str):
            label, vals = v[0].strip(), v[1:]
        else:
            continue
        m = re.match(r"^Q([1-4])", label)
        if not m or year is None or len(vals) < 3:
            continue
        rows.append(dict(quarter_start=pd.Timestamp(year, (int(m.group(1)) - 1) * 3 + 1, 1),
                         hpi_index=vals[0], hpi_yoy_pct=vals[1], price_rm=vals[2],
                         is_preliminary=label.upper().endswith("P")))
    d = pd.DataFrame(rows)
    for c in ("hpi_index", "hpi_yoy_pct", "price_rm"):
        d[c] = to_num(d[c])
    d = d.drop_duplicates("quarter_start").sort_values("quarter_start").reset_index(drop=True)
    return d


def build_hpi():
    nat = parse_change_sheet(QU / "malaysian_house_price_index_q3_2026p_tables.xlsx")
    log("hpi", f"national All-House MHPI taken from NAPIC xlsx 'CHANGE' sheet (Q3-2026P release): "
               f"{qlabel(nat.quarter_start).iloc[0]}..{qlabel(nat.quarter_start).iloc[-1]} ({len(nat)} quarters). "
               "Q3-2026 is marked PRELIMINARY. Mixed text/number cells in that sheet were coerced to numeric.")
    nat["hpi_qoq_pct"] = nat["hpi_index"].pct_change() * 100

    c = read_u16(QU / "malaysian_house_price_index_quarterly_summary_by_state_and_type_2010_2026.csv.csv", header=None)
    quarters = c.iloc[0, 3:].tolist()
    body = c.iloc[1:].copy()
    body.columns = ["state", "type", "measure"] + quarters
    body[["state", "type"]] = body[["state", "type"]].ffill()
    long = body.melt(id_vars=["state", "type", "measure"], var_name="quarter_raw", value_name="v")
    long["v"] = to_num(long["v"])
    long["state"] = std_state(long["state"])
    long["type"] = long["type"].replace({"High-Rise": "High-rise"})
    mm = {"Index Point": "hpi_index", "Price": "price_rm", "Y-o-Y %": "yoy_pct", "Q-o-Q %": "qoq_pct"}
    long["measure"] = long["measure"].map(mm)
    wide = long.pivot_table(index=["state", "type", "quarter_raw"], columns="measure", values="v").reset_index()
    wide["quarter_start"] = qstart(wide["quarter_raw"])
    wide["quarter"] = qlabel(wide["quarter_start"])
    wide = wide.drop(columns="quarter_raw").sort_values(["state", "type", "quarter_start"])
    wide = wide[["state", "type", "quarter", "quarter_start", "hpi_index", "price_rm", "yoy_pct", "qoq_pct"]]
    save(wide, "tidy", "hpi_state_type_quarterly_long.csv")
    log("hpi", "state x type HPI (2010Q1-2026Q1) parsed from the wide UTF-16 export. High-rise is not reported for "
               "Kedah, Kelantan, Pahang, Perak, Perlis, Sarawak, Terengganu and Detached not for Perlis (absent in source, not missing data).")

    # cross-check national All House: CHANGE sheet vs state/type file
    chk = wide[(wide.state == "Malaysia") & (wide.type == "All House")].merge(nat, on="quarter_start", suffixes=("_a", "_b"))
    diff = (chk["hpi_index_a"] - chk["hpi_index_b"]).abs().max()
    chk["d"] = (chk["hpi_index_a"] - chk["hpi_index_b"]).abs()
    off = chk[chk.d > 0.15]
    log("hpi", f"QA: Malaysia All-House index agrees between the two sources in {len(chk) - len(off)}/{len(chk)} common quarters; "
               f"differences >0.15 pts: {', '.join(f'{q} ({a} vs {b})' for q, a, b in zip(qlabel(off.quarter_start), off.hpi_index_a, off.hpi_index_b)) or 'none'}. "
               "Cause: the xlsx (Q3-2026P release) revised the latest quarter; xlsx is used for all-house, the older CSV for by-type/state, "
               "so 2026-Q1 all-house in quarterly_master (xlsx) can differ slightly from state_quarterly_panel/by-type values (CSV).")

    real = pd.read_csv(QU / "real_residential_property_prices_malaysia_quarterly_index.csv", parse_dates=["observation_date"])
    real = real.rename(columns={"observation_date": "quarter_start", "QMYR628BIS": "real_rppi_bis"})
    return nat, wide, real


# =========================================================================== 8. ANNUAL HPI / INCOME
def build_annual_hpi() -> pd.DataFrame:
    a = read_u16(ST / "annual_malaysian_house_price_index_and_prices_by_state_and_type_2010_2025.csv.csv", header=None)
    types = a.iloc[0, 2:].ffill().tolist()
    meas = a.iloc[1, 2:].tolist()
    body = a.iloc[2:].copy()
    body.columns = ["year", "state"] + [f"{t}|{m}" for t, m in zip(types, meas)]
    long = body.melt(id_vars=["year", "state"], var_name="k", value_name="v")
    long[["type", "measure"]] = long["k"].str.split("|", expand=True)
    long["v"] = to_num(long["v"])
    long["year"] = long["year"].astype(int)
    long["state"] = std_state(long["state"])
    long["measure"] = long["measure"].map({"Index Point": "hpi_index", "Y-o-Y %": "yoy_pct", "Price": "price_rm"})
    long["type"] = long["type"].str.lower().str.replace("-", "_").str.replace(" ", "_")
    w = long.pivot_table(index=["state", "year"], columns=["type", "measure"], values="v")
    w.columns = [f"{t}_{m}" for t, m in w.columns]
    return w.reset_index()


def interp_income(df: pd.DataFrame, geo_col: str, value_col: str, y0=2009, y1=2026) -> pd.DataFrame:
    """Survey-year values -> annual series. Log-linear interpolation between surveys; beyond the last survey the
    CAGR between the last two surveys is carried forward (flagged 'extrapolated')."""
    out = []
    for geo, g in df.dropna(subset=[value_col]).groupby(geo_col):
        s = g.set_index("year")[value_col].astype(float).sort_index()
        if len(s) < 2:
            continue
        s = s[~s.index.duplicated(keep="last")]
        years = pd.Index(range(max(y0, s.index.min()), y1 + 1), name="year")
        ls = np.log(s).reindex(years)
        ls = ls.interpolate(method="index", limit_area="inside")
        last, prev = s.index[-1], s.index[-2]
        cagr = (np.log(s.iloc[-1]) - np.log(s.iloc[-2])) / (last - prev)
        src = pd.Series("interpolated", index=years)
        src[s.index.intersection(years)] = "survey"
        for y in years[years > last]:
            ls[y] = np.log(s.iloc[-1]) + cagr * (y - last)
            src[y] = "extrapolated"
        out.append(pd.DataFrame({geo_col: geo, "year": years, f"{value_col}_est": np.exp(ls.values),
                                 f"{value_col}_source": src.values}))
    return pd.concat(out, ignore_index=True)


def build_income():
    hs = pd.read_csv(DS / "household_Income_state.csv")
    hs["year"] = pd.to_datetime(hs["date"], dayfirst=True).dt.year
    hs["state"] = std_state(hs["state"])
    hm = pd.read_csv(DS / "hh_income.csv")
    hm["year"] = pd.to_datetime(hm["date"]).dt.year
    hm["state"] = "Malaysia"
    # the 'Structural...' folder copy of the national file must match
    hm2 = pd.read_csv(ST / "malaysian_household_mean_and_median_income_1970_2024.csv.csv")
    assert hm[["income_mean", "income_median"]].equals(hm2[["income_mean", "income_median"]]), "national income files differ"
    inc = pd.concat([hs[["state", "year", "income_mean", "income_median"]],
                     hm[["state", "year", "income_mean", "income_median"]]], ignore_index=True)
    save(inc.sort_values(["state", "year"]).rename(columns={"income_mean": "hh_income_mean_rm_month",
                                                              "income_median": "hh_income_median_rm_month"}),
         "tidy", "household_income_survey_years.csv")
    log("income", "household income (DOSM HIES) exists only for survey years (…2012, 2014, 2016, 2019, 2020, 2022, 2024). "
                  "Annual values between surveys are LOG-LINEAR INTERPOLATIONS; 2025-26 are EXTRAPOLATED with the last "
                  "two-survey CAGR. Every row carries *_source = survey | interpolated | extrapolated.")
    log("income", "HIES 2020 (COVID/MCO year) is lower than 2019 for most states - kept as reported, not smoothed.")
    med = interp_income(inc, "state", "income_median")
    mean = interp_income(inc, "state", "income_mean")
    est = med.merge(mean, on=["state", "year"], how="outer")
    return inc, est


# =========================================================================== 9. B40 / M40 / T20
def build_income_groups(annual_hpi: pd.DataFrame) -> pd.DataFrame:
    nat = pd.read_csv(DS / "hies_malaysia_percentile.csv")
    nat["state"] = "Malaysia"
    st = pd.read_csv(DS / "income_percentile_state_clean.csv")
    st["state"] = std_state(st["state"])
    allp = pd.concat([nat, st], ignore_index=True)
    allp["year"] = pd.to_datetime(allp["date"], format="mixed", dayfirst=True).dt.year
    allp = allp[allp.variable == "mean"].dropna(subset=["income"])
    allp["group"] = pd.cut(allp["percentile"], [0, 40, 80, 100], labels=["B40", "M40", "T20"])
    g = allp.groupby(["state", "year", "group"], observed=True)["income"].mean().reset_index(name="group_mean_income_rm_month")
    tot = allp.groupby(["state", "year"])["income"].mean().reset_index(name="group_mean_income_rm_month")
    tot["group"] = "All"
    g = pd.concat([g, tot], ignore_index=True)
    g["group"] = g["group"].astype(str)
    log("income_groups", "B40 = percentiles 1-40, M40 = 41-80, T20 = 81-100 of HIES household income; group value = "
                         "average of the 1%-household percentile means (equal weights). Available for 2019, 2022, 2024: "
                         "Malaysia + 4 states only (Melaka, N.Sembilan, Selangor, KL).")

    px = annual_hpi.copy()
    sel = ["state", "year", "all_house_price_rm", "terraced_price_rm", "high_rise_price_rm"]
    px = px[sel]
    r = g.merge(px, on=["state", "year"], how="left")
    for t in ("all_house", "terraced", "high_rise"):
        r[f"price_to_annual_income_{t}"] = r[f"{t}_price_rm"] / (12 * r["group_mean_income_rm_month"])
    base = r[r.year == 2019].set_index(["state", "group"])
    for c, nm in (("group_mean_income_rm_month", "income_idx_2019"), ("all_house_price_rm", "all_house_price_idx_2019")):
        r[nm] = r.apply(lambda x: x[c] / base[c].get((x["state"], x["group"]), np.nan) * 100
                        if (x["state"], x["group"]) in base.index else np.nan, axis=1)
    r = r.sort_values(["state", "group", "year"]).reset_index(drop=True)
    log("income_groups", "price_to_annual_income = NAPIC MHPI price / (12 x group mean monthly income). MHPI 'price' is an "
                         "index-scaled price level (higher than the median transaction price), so ratios are best read as trends.")
    return r


# =========================================================================== 10. REFERENCE TABLES
def build_reference():
    # --- price distribution by category 2022-2026
    d = read_u16(QU / "msia_residential_property_prices_by_category_2022_2026.csv")
    d = d.drop(columns=[c for c in d.columns if c.startswith("2022")], errors="ignore")
    d.columns = ["category", "min_rm", "p25_rm", "mean_rm", "median_rm", "p75_rm", "max_rm", "sample_size"]
    for c in d.columns[1:]:
        d[c] = to_num(d[c])
    d["is_total"] = d["category"].eq("Grand Total")
    d["period"] = "2022-2026 (pooled)"
    save(d, "reference", "price_distribution_by_category_2022_2026.csv")

    # --- annual mean / median transaction price
    a = read_u16(QU / "Annual_mean_and_median_price_trends_(2022#U20132026)..csv")
    a = a.rename(columns={a.columns[0]: "stat"}).set_index("stat").T
    a.index = a.index.astype(int)
    a = a.apply(to_num).rename(columns={"Mean": "mean_price_rm", "Median": "median_price_rm"})
    a.index.name = "year"
    a["is_partial_year"] = a.index == 2026
    save(a.reset_index(), "reference", "annual_transaction_price_mean_median_2022_2026.csv")
    log("prices", "2026 mean/median transaction price is year-to-date (partial year).")

    # --- price by state/category (NAPIC)
    p = pd.read_csv(QU / "napic_residential_prices_by_state.csv")
    p = p.drop(columns=["maximum_clipped_prefix", "year_filter", "quarter_filter", "source_note"])
    p["state"] = std_state(p["state"])
    save(p, "reference", "napic_price_by_state_category.csv")
    log("prices", "napic_residential_prices_by_state.csv: only Johor, Kedah, Kelantan, Kuala Lumpur, Melaka covered (PDF export); "
                  "'maximum' column was truncated in the source (e.g. '3,6...') so it was dropped.")

    # --- transactions extract -> summary
    t = pd.read_csv(QU / "msia_residential_property_transactions_by_category_and_state.csv")
    t["State"] = std_state(t["State"])
    s = (t.groupby(["Category", "State"])["Nilai Lpr"]
         .agg(n_sampled="size", mean_rm="mean", median_rm="median",
              p25_rm=lambda x: x.quantile(.25), p75_rm=lambda x: x.quantile(.75)).reset_index())
    s.columns = ["category", "state"] + list(s.columns[2:])
    s["low_sample_flag"] = s["n_sampled"] < 30
    save(s, "reference", "transaction_price_summary_category_state.csv")
    log("transactions", f"transactions file is a ~{len(t):,}-row EXTRACT (no dates) of 2022-2026 deals "
                        "(source states 395,011). Summaries by category x state are indicative; rows with n<30 flagged.")

    # --- transaction value tables
    h = pd.read_csv(QU / "half_yearly_property_transaction_values_by_sector_and_price_range.csv",
                    parse_dates=["period_start_date", "period_end_date"])
    h = h.drop(columns=["price_range_raw", "filter_sector", "filter_half_year", "filter_price_range", "source_note"])
    save(h, "reference", "transaction_value_half_yearly_by_sector_price_range.csv")
    q = pd.read_csv(QU / "napic_quarterly_value_by_sector_price_range.csv", parse_dates=["quarter_start_date", "quarter_end_date"])
    q["value_clipped_in_source"] = q["dev_land_clipped_prefix"].notna()
    q = q.drop(columns=["filter_date_part", "filter_year", "filter_half_year", "filter_quarter", "source_note",
                        "dev_land_clipped_prefix"])
    save(q, "reference", "transaction_value_quarterly_by_sector_price_range.csv")
    log("transactions", "quarterly value table: Development Land values and price brackets above RM600k are truncated in the "
                        "source PDF export -> left NaN and flagged value_clipped_in_source; treat as incomplete.")

    # --- stock & supply
    s = pd.read_csv(QU / "napic_residential_property_stock_and_supply_by_state_and_stage_q3_2026.csv.csv")
    types = [c for c in s.columns if c not in ("review_period", "state", "development_stage", "notes", "row_total_units", "source_file")]
    sl = s.melt(id_vars=["review_period", "state", "development_stage", "notes", "row_total_units"], value_vars=types,
                var_name="property_type", value_name="units")
    sl["state"] = std_state(sl["state"])
    save(sl, "reference", "residential_stock_supply_q3_2026_long.csv")
    log("overhang", "stock/supply table covers ONLY Negeri Sembilan and Johor (Q3-2026 dashboard capture). It has existing stock, "
                    "completions, incoming supply, starts, planned supply - but NO unsold/overhang unit counts.")

    # --- overhang facts transcribed from the KRI PDF (Sept 2023) -- source figures are NAPIC 2022 / MyCensus 2020
    facts = [
        ("overhang_units_total", "Malaysia", 2022, 27746, "units", "NAPIC Property Market Report 2022 via KRI Views 19/23"),
        ("overhang_value_rm_billion", "Malaysia", 2022, 18.41, "RM bn", "same"),
        ("overhang_units_rm300k_and_below", "Malaysia", 2022, 6509, "units", "same (23.5% of overhang)"),
        ("overhang_units_rm300k_to_500k", "Malaysia", 2022, 11988, "units", "same (43.2%)"),
        ("overhang_units_rm500k_to_1m", "Malaysia", 2022, 5463, "units", "same (19.7%)"),
        ("overhang_units_above_rm1m", "Malaysia", 2022, 3786, "units", "same (13.6%)"),
        ("transacted_units_rm300k_and_below", "Malaysia", 2022, 135616, "units", "same (55.8% of transactions)"),
        ("transacted_units_rm300k_to_500k", "Malaysia", 2022, 58864, "units", "same"),
        ("transacted_units_rm500k_to_1m", "Malaysia", 2022, 37358, "units", "same"),
        ("transacted_units_above_rm1m", "Malaysia", 2022, 11352, "units", "same"),
        ("transacted_units_total", "Malaysia", 2022, 243190, "units", "same"),
        ("median_multiple_affordability", "Malaysia", 2022, 4.3, "x annual household income", "KRI Views 19/23 ('seriously unaffordable')"),
        ("vacant_living_quarters", "Malaysia", 2020, 1_800_000, "units", "MyCensus 2020 via KRI (19.4% of 9.6m living quarters)"),
        ("vacant_newly_completed_for_sale_or_rent", "Malaysia", 2020, 704935, "units", "MyCensus 2020 via KRI Fig. 6"),
    ]
    f = pd.DataFrame(facts, columns=["indicator", "geography", "year", "value", "unit", "source"])
    save(f, "reference", "overhang_vacancy_facts_kri_2023.csv")
    log("overhang", "the overhang PDF is an opinion piece: only 2022 national overhang/transaction counts and 2020 vacancy were "
                    "transcribed. NO overhang time series exists in the collected data (gap for the model).")

    # --- households vs living quarters (vacancy proxy)
    n = pd.read_csv(ST / "household_vs_livingquarter.csv")
    n["state"] = "Malaysia"
    s4 = pd.read_csv(DS / "hh_lq_state_CLEAN.csv")
    s4["state"] = std_state(s4["state"])
    hl = pd.concat([n, s4[["date", "households", "living_quarters", "state"]]], ignore_index=True)
    hl["year"] = pd.to_datetime(hl["date"], format="mixed", dayfirst=True).dt.year
    hl["surplus_living_quarters"] = hl["living_quarters"] - hl["households"]
    hl["vacancy_proxy_pct"] = hl["surplus_living_quarters"] / hl["living_quarters"] * 100
    save(hl[["state", "year", "households", "living_quarters", "surplus_living_quarters", "vacancy_proxy_pct"]]
         .sort_values(["state", "year"]), "reference", "households_vs_living_quarters_census.csv")
    log("overhang", "vacancy_proxy_pct = (living quarters - households) / living quarters. Census years only; a proxy, "
                    "not the official vacancy rate.")

    # --- internet access (basic amenities)
    i = pd.read_csv(DS / "hh_access_internet(CLEAN).csv")
    i["year"] = pd.to_datetime(i["date"], dayfirst=True).dt.year
    i["state"] = std_state(i["state"])
    save(i[["state", "year", "strata", "proportion"]].rename(columns={"proportion": "hh_internet_access_pct"}),
         "reference", "hh_internet_access_state.csv")

    # --- GHG
    e = pd.read_csv(DS / "ghg_emissions.csv")
    e["year"] = pd.to_datetime(e["date"]).dt.year
    w = e.pivot(index="year", columns="source", values="emissions").add_prefix("ghg_").add_suffix("_ktco2e").reset_index()
    save(w, "reference", "ghg_emissions_annual.csv")
    log("carbon", "GHG: national annual 1990-2023 by source (unit assumed kt CO2e - not stated in file). 'industrial_processes' "
                  "includes cement/steel process emissions but is not split out; LULUCF/net blank for 2022-23. "
                  "NO cement/steel intensity benchmarks or Green Building Index data were provided (gap).")
    return w


# =========================================================================== MASTER TABLES
def main() -> None:
    print("== Fuel / FX / OPR");           fuel_w, fuel_m = build_fuel(); fx_d, fx_m = build_fx(); opr_m = build_opr()
    print("== Cost indices");              bmci_m, bmci_reg = build_bmci(); civil_m = build_civil()
    print("== CPI / PPI / wages");         cpi_m = build_cpi(); ppi_m = build_ppi(); fw_m = build_formal_wages()
    print("== HPI");                       nat, hpi_long, real = build_hpi()
    print("== Income");                    inc, inc_est = build_income()
    ann_hpi = build_annual_hpi()
    print("== Reference tables");          build_reference()

    # ------------------------------------------------------------------ monthly master
    grid = pd.date_range("2010-01-01", LAST_MONTH, freq="MS", name="month")
    mm = pd.DataFrame(index=grid)
    for part in (fuel_m, fx_m, opr_m, bmci_m, civil_m, cpi_m, ppi_m, fw_m):
        mm = mm.join(part, how="left")
    mm = mm.reset_index()
    mm["bmci_main"] = mm["bmci_bld_rc_2_4_storey_flat_roof_incl_steel"]   # designated main BMCI series
    mm.insert(1, "year", mm["month"].dt.year)
    mm.insert(2, "quarter", qlabel(mm["month"]))
    save(mm, "model_ready", "monthly_master.csv")
    first_full = mm.dropna(subset=["fuel_ron95", "fx_myr_per_usd", "bmci_bld_rc_2_4_storey_flat_roof_incl_steel", "cpi_overall", "ppi_index"])["month"].min()
    log("master", f"monthly_master: {len(mm)} months 2010-01..2026-10. Fuel starts 2017-03 and FX 2015-01, so the first month with "
                  f"all core drivers present is {first_full:%Y-%m}. BMCI/civil prices end 2026-10, PPI 2026-09, formal wages 2026-06; "
                  "later months are NaN (not imputed). "
                  "2026-11 (partial month for fuel/FX) was removed: modelling window ends 2026-10-01.")

    # ------------------------------------------------------------------ quarterly master (national)
    key_m = [c for c in mm.columns if c not in ("month", "year", "quarter")]
    qm = qmean(mm.set_index("month")[key_m])
    # OPR: quarter-end level, summed change
    opr_q = pd.DataFrame({"bnm_opr_pct_eoq": opr_m["bnm_opr_pct"].resample("QS").last(),
                          "bnm_opr_change_pct_sum": opr_m["bnm_opr_change_pct"].resample("QS").sum()})
    qm = qm.drop(columns=["bnm_opr_pct", "bnm_opr_change_pct"]).join(opr_q)
    qm.index.name = "quarter_start"
    yoy_cols = ["fuel_ron95", "fuel_ron97", "fuel_diesel", "fuel_ron95_effective", "fx_myr_per_usd", "bmci_cement", "bmci_steel",
                "bmci_sand", "bmci_aggregates", "bmci_ready_mix_concrete", "price_cement", "price_steel_ht_12mm",
                "price_sand_river", "cpi_overall", "ppi_index", "bmci_bld_rc_2_4_storey_flat_roof_incl_steel"]
    for c in yoy_cols:
        qm[f"{c}_qoq_pct"] = qm[c].pct_change() * 100
        qm[f"{c}_yoy_pct_q"] = qm[c].pct_change(4) * 100
    qm = qm.loc["2010-01-01":"2026-07-01"]

    types = hpi_long[hpi_long.state == "Malaysia"].copy()
    types["type"] = types["type"].str.lower().str.replace("-", "_").str.replace(" ", "_")
    tw = types.pivot_table(index="quarter_start", columns="type", values=["hpi_index", "price_rm", "yoy_pct", "qoq_pct"])
    MS = {"hpi_index": "index", "price_rm": "price_rm", "yoy_pct": "yoy_pct", "qoq_pct": "qoq_pct"}
    tw.columns = [f"hpi_{t}_{MS[m]}" for m, t in tw.columns]
    tw = tw.drop(columns=[c for c in tw.columns if c.startswith("hpi_all_house")])   # all-house comes from CHANGE sheet
    nat_i = nat.set_index("quarter_start").rename(columns={
        "hpi_index": "hpi_all_house_index", "hpi_yoy_pct": "hpi_all_house_yoy_pct", "price_rm": "hpi_all_house_price_rm",
        "hpi_qoq_pct": "hpi_all_house_qoq_pct", "is_preliminary": "hpi_is_preliminary"})
    qmaster = nat_i.join(tw, how="left").join(real.set_index("quarter_start"), how="left").join(qm, how="left")
    qmaster["bmci_main"] = qmaster["bmci_bld_rc_2_4_storey_flat_roof_incl_steel"]
    qmaster["bmci_main_yoy_pct_q"] = qmaster["bmci_main"].pct_change(4) * 100
    qmaster = qmaster.loc["2010-01-01":].reset_index()
    qmaster.insert(1, "quarter", qlabel(qmaster["quarter_start"]))
    save(qmaster, "model_ready", "quarterly_master.csv")
    log("master", "quarterly_master: 2010-Q1..2026-Q3. Monthly drivers are quarter MEANS, only when all 3 months exist "
                  "(so fuel starts 2017-Q2, BMCI/civil end 2026-Q3). By-type HPI ends 2026-Q1 (source file); all-house runs to "
                  "2026-Q3P (xlsx). *_yoy_pct_q / *_qoq_pct are computed from the quarterly means.")

    # ------------------------------------------------------------------ annual state panel
    ann = ann_hpi.merge(inc_est, on=["state", "year"], how="left")
    w = pd.read_csv(ST / "malaysian_real_mean_and_median_wages_by_state_2010_2025.csv.csv")
    w["year"] = pd.to_datetime(w["date"]).dt.year
    w["state"] = std_state(w["state"])
    ann = ann.merge(w[["state", "year", "wages_median_real", "wages_mean_real"]].rename(
        columns={"wages_median_real": "wage_median_real_rm", "wages_mean_real": "wage_mean_real_rm"}), on=["state", "year"], how="left")
    g = pd.read_csv(ST / "malaysian_state_household_income_expenditure_gini_and_poverty_2022_2024.csv.csv")
    g["year"] = pd.to_datetime(g["date"]).dt.year
    g["state"] = std_state(g["state"])
    ann = ann.merge(g[["state", "year", "expenditure_mean", "gini", "poverty"]].rename(
        columns={"expenditure_mean": "hh_expenditure_mean_rm_month", "gini": "gini", "poverty": "poverty_rate_pct"}),
        on=["state", "year"], how="left")
    ann["median_multiple_mhpi"] = ann["all_house_price_rm"] / (12 * ann["income_median_est"])
    ann["terraced_multiple_mhpi"] = ann["terraced_price_rm"] / (12 * ann["income_median_est"])
    ann["high_rise_multiple_mhpi"] = ann["high_rise_price_rm"] / (12 * ann["income_median_est"])
    ann = ann.sort_values(["state", "year"])
    ann["price_yoy_pct_calc"] = ann.groupby("state")["all_house_price_rm"].pct_change() * 100
    ann["income_median_yoy_pct_calc"] = ann.groupby("state")["income_median_est"].pct_change() * 100
    ann = ann[(ann.year >= 2010) & (ann.year <= 2025)]
    save(ann, "model_ready", "annual_state_panel.csv")
    log("wages", "malaysian_real_*_wages are REAL (constant-price) formal-sector wages 2010-2025; prices/household income are "
                 "nominal. Do not divide nominal price by real wage without re-basing.")
    log("income", "gini/poverty/expenditure only exist for 2022 and 2024.")

    # ------------------------------------------------------------------ annual national
    an = ann[ann.state == "Malaysia"].drop(columns="state").copy()
    txn = pd.read_csv(QU / "Annual_mean_and_median_price_trends_(2022#U20132026)..csv", encoding="utf-16", sep="\t")
    txn = txn.rename(columns={txn.columns[0]: "stat"}).set_index("stat").T
    txn.index = txn.index.astype(int)
    txn = txn.apply(to_num)
    an = an.merge(txn.rename(columns={"Median": "txn_median_price_rm", "Mean": "txn_mean_price_rm"}),
                  left_on="year", right_index=True, how="left")
    an["median_multiple_txn"] = an["txn_median_price_rm"] / (12 * an["income_median_est"])
    drivers = mm.set_index("month").resample("YS").mean(numeric_only=True)
    cnt = mm.set_index("month")["fuel_ron95"].resample("YS").count()
    keep = ["fuel_ron95", "fuel_ron97", "fuel_diesel", "fuel_ron95_effective", "fx_myr_per_usd", "bnm_opr_pct", "bmci_cement",
            "bmci_steel", "bmci_sand", "bmci_ready_mix_concrete", "bmci_bld_rc_2_4_storey_flat_roof_incl_steel",
            "price_cement", "price_steel_ht_12mm", "price_sand_river", "cpi_overall", "cpi_transport", "cpi_housing_utilities", "ppi_index"]
    d = drivers[keep].copy()
    d.loc[cnt < 12, [c for c in keep if c.startswith("fuel_")]] = np.nan      # require full year of fuel data
    d["year"] = d.index.year
    d = d.add_suffix("_annual_mean").rename(columns={"year_annual_mean": "year"})
    an = an.merge(d, on="year", how="left").merge(
        pd.read_csv(OUT / "reference" / "ghg_emissions_annual.csv"), on="year", how="left")
    for c in ("income_median_est", "all_house_price_rm", "terraced_price_rm", "high_rise_price_rm"):
        an[f"{c}_idx_2015"] = an[c] / an.loc[an.year == 2015, c].iloc[0] * 100
    save(an, "model_ready", "annual_national.csv")
    log("master", "annual_national: Malaysia, 2010-2025. median_multiple_txn uses NAPIC median transaction price (2022+; reproduces "
                  "KRI's 4.3x for 2022 within rounding) while median_multiple_mhpi uses the MHPI price level (higher).")

    # ------------------------------------------------------------------ B40/M40/T20 affordability
    grp = build_income_groups(ann_hpi)
    save(grp, "model_ready", "affordability_by_income_group.csv")

    # ------------------------------------------------------------------ state-quarter panel
    p = hpi_long[(hpi_long.state != "Malaysia")].copy()
    p["type"] = p["type"].str.lower().str.replace("-", "_").str.replace(" ", "_")
    pw = p.pivot_table(index=["state", "quarter", "quarter_start"], columns="type",
                       values=["hpi_index", "price_rm", "yoy_pct", "qoq_pct"])
    pw.columns = [f"hpi_{t}_{MS[m]}" for m, t in pw.columns]
    pw = pw.reset_index()
    REGION = {"Johor": "Johor", "Kelantan": "Kelantan & Terengganu", "Terengganu": "Kelantan & Terengganu",
              "Pahang": "Pahang", "Pulau Pinang": "Penang, Kedah & Perlis", "Kedah": "Penang, Kedah & Perlis",
              "Perlis": "Penang, Kedah & Perlis", "Perak": "Perak", "Selangor": CENTRAL, "Kuala Lumpur": CENTRAL,
              "Melaka": CENTRAL, "Negeri Sembilan": CENTRAL}
    pw["bmci_region"] = pw["state"].map(REGION).fillna("none (Sabah/Sarawak not covered)")
    # regional quarterly BMCI
    reg = bmci_reg[bmci_reg.region != OVERLAP].set_index(["region", "date"])
    regq = (reg.groupby(level=0).resample("QS", level=1).mean()).reset_index().rename(columns={"region": "bmci_region", "date": "quarter_start"})
    pw = pw.merge(regq, on=["bmci_region", "quarter_start"], how="left")
    # East Malaysia: national mean fallback (flagged)
    natq = qm[[c for c in qm.columns if c in ("bmci_cement", "bmci_steel", "bmci_sand", "bmci_aggregates", "bmci_ready_mix_concrete")]]
    fallback = pw["bmci_region"].str.startswith("none")
    pw["bmci_is_national_fallback"] = fallback
    for c in natq.columns:
        tgt = c.replace("bmci_", "bmci_region_")
        pw.loc[fallback, tgt] = pw.loc[fallback, "quarter_start"].map(natq[c])
    pw.loc[fallback, "bmci_region_rc_2_4_flat_index_incl_steel"] = pw.loc[fallback, "quarter_start"].map(qm["bmci_bld_rc_2_4_storey_flat_roof_incl_steel"])
    pw.loc[fallback, "bmci_region_rc_2_4_flat_index_excl_steel"] = pw.loc[fallback, "quarter_start"].map(qm["bmci_bld_rc_2_4_storey_flat_roof_excl_steel"])
    pw["bmci_main"] = pw["bmci_region_rc_2_4_flat_index_incl_steel"]   # regional version of the main series
    # national drivers
    ndr = ["fuel_ron95", "fuel_ron97", "fuel_diesel", "fuel_ron95_effective", "fx_myr_per_usd", "bnm_opr_pct_eoq", "ppi_index",
           "cpi_overall", "cpi_transport", "price_cement", "price_steel_ht_12mm", "price_sand_river"]
    pw = pw.merge(qm[ndr].add_prefix("nat_").reset_index(), on="quarter_start", how="left")
    # income & affordability
    pw["year"] = pw["quarter_start"].dt.year
    pw = pw.merge(inc_est[["state", "year", "income_median_est", "income_median_source", "income_mean_est"]], on=["state", "year"], how="left")
    pw["median_multiple_mhpi"] = pw["hpi_all_house_price_rm"] / (12 * pw["income_median_est"])
    pw = pw.sort_values(["state", "quarter_start"])
    save(pw, "model_ready", "state_quarterly_panel.csv")
    log("master", f"state_quarterly_panel: {pw.state.nunique()} states x {pw.quarter.nunique()} quarters ({len(pw)} rows), 2010-Q1..2026-Q1. "
                  "Regional BMCI joined via a state->BMCI-region mapping (Kelantan+Terengganu, Penang+Kedah+Perlis, "
                  "Selangor+KL+Melaka+N9 share a region). Sabah/Sarawak have no BMCI -> national mean used (bmci_is_national_fallback=True).")
    log("master", "Income in the panel is the ANNUAL estimate repeated across the 4 quarters of a year (not a quarterly measure).")

    write_reports()


def write_reports() -> None:
    lines = ["# Data quality & decision log", "", "_Auto-generated by `scripts/process_data.py`._", ""]
    cur = None
    for area, msg in LOG:
        if area != cur:
            lines += ["", f"## {area}"]
            cur = area
        lines.append(f"- {msg}")
    (REP / "data_quality_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    prof = []
    for p in OUTPUTS:
        df = pd.read_csv(p, low_memory=False)
        for c in df.columns:
            s = df[c]
            num = pd.to_numeric(s, errors="coerce")
            isnum = num.notna().sum() > 0.9 * s.notna().sum() and s.notna().any()
            prof.append(dict(file=f"{p.parent.name}/{p.name}", column=c, dtype=str(s.dtype), rows=len(df),
                             non_null=int(s.notna().sum()), pct_null=round(100 * s.isna().mean(), 1),
                             min=(num.min() if isnum else None), max=(num.max() if isnum else None),
                             first=str(s.dropna().iloc[0]) if s.notna().any() else None))
    pd.DataFrame(prof).to_csv(REP / "output_profile.csv", index=False)
    print(f"\nDone: {len(OUTPUTS)} files in {OUT}")


if __name__ == "__main__":
    main()
