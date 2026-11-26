"""
MDIT26 - Shahizz Python AI Modeling Track
Step 2: Stationarity (ADF/KPSS), Lag/VAR & Granger Checks, and Time-Series ML/AI Baselines
for BMCI / Building Cost Drivers.
"""

from pathlib import Path
import contextlib
import io
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.api import VAR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


DEFAULT_PANEL_PATH = "data/processed/monthly_master.csv"
DEFAULT_TARGET_COL = "bmci_bld_rc_2_4_storey_flat_roof_incl_steel"
DEFAULT_DRIVERS = [
    "fuel_diesel",
    "fx_myr_per_usd",
    "bnm_opr_pct",
    "cpi_overall",
    "ppi_index",
]


def make_demo_panel(n_periods: int = 84) -> pd.DataFrame:
    """Fallback monthly series so you can verify execution before Danish's export arrives."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2019-01-01", periods=n_periods, freq="MS")

    steel_shock = rng.normal(0.003, 0.012, size=n_periods)
    cement_shock = rng.normal(0.002, 0.007, size=n_periods)
    diesel_shock = rng.normal(0.002, 0.015, size=n_periods)
    fx_shock = rng.normal(0.001, 0.009, size=n_periods)
    demand_shock = rng.normal(0.002, 0.008, size=n_periods)

    df = pd.DataFrame({
        "date": dates,
        "bmci_steel_idx": 100.0 * np.exp(np.cumsum(steel_shock + 0.2 * fx_shock)),
        "bmci_cement_idx": 100.0 * np.exp(np.cumsum(cement_shock + 0.15 * diesel_shock)),
        "fuel_diesel_price": 2.15 * np.exp(np.cumsum(diesel_shock)),
        "usd_myr_rate": 4.15 * np.exp(np.cumsum(fx_shock)),
        "construction_demand_proxy": 100.0 * np.exp(np.cumsum(demand_shock)),
    })

    bmci_growth = (
        0.002
        + 0.35 * pd.Series(steel_shock).shift(1).fillna(0).to_numpy()
        + 0.20 * pd.Series(cement_shock).shift(2).fillna(0).to_numpy()
        + 0.15 * pd.Series(diesel_shock).shift(1).fillna(0).to_numpy()
        + 0.12 * pd.Series(fx_shock).shift(2).fillna(0).to_numpy()
        + 0.10 * pd.Series(demand_shock).shift(1).fillna(0).to_numpy()
        + rng.normal(0, 0.003, size=n_periods)
    )
    df["bmci_total_idx"] = 100.0 * np.exp(np.cumsum(bmci_growth))
    return df


def load_panel(
    csv_path: str = DEFAULT_PANEL_PATH,
    max_date: str = "2026-10-01",
) -> pd.DataFrame:
    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
        date_col = (
            "date" if "date" in df.columns
            else "month" if "month" in df.columns
            else df.columns[0]
        )
        df = df.copy()
        df["date"] = pd.to_datetime(df[date_col])
        df = df.sort_values("date")
        if max_date:
            df = df[df["date"] <= pd.to_datetime(max_date)]
        return df.reset_index(drop=True)
    print(f"[INFO] '{csv_path}' not found yet. Using synthetic fallback for dry-run.")
    return make_demo_panel()


def run_stationarity_checks(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    #Runs ADF and KPSS on levels and first-differences to check stationarity
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < 15:
                continue
            _, adf_p, *_ = adfuller(s, autolag="AIC")
            try:
                _, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
            except Exception:
                kpss_p = np.nan

            s_diff = s.diff().dropna()
            diff_adf_p = adfuller(s_diff, autolag="AIC")[1] if len(s_diff) > 10 else np.nan

            rows.append({
                "series": col,
                "adf_p_level": round(float(adf_p), 4),
                "kpss_p_level": round(float(kpss_p), 4) if pd.notna(kpss_p) else np.nan,
                "adf_p_diff1": round(float(diff_adf_p), 4) if pd.notna(diff_adf_p) else np.nan,
            })
    return pd.DataFrame(rows)


def run_var_and_granger(stationary_df: pd.DataFrame, target_col: str, maxlags: int = 4) -> None:
    #Fits a VAR on stationary growth/diff series and prints pairwise Granger p-values
    clean = stationary_df.dropna()
    print(f"[VAR] rows={len(clean)}, target_col='{target_col}'")
    if len(clean) < maxlags + 15:
        print("[WARN] Too few rows for VAR/Granger check.")
        return

    var_model = VAR(clean)
    lag_sel = var_model.select_order(maxlags=min(maxlags, max(1, len(clean) // 8)))
    best_lag = max(1, int(lag_sel.aic or 1))
    fitted = var_model.fit(best_lag)

    print(f"\n[VAR] Selected lag (AIC): {best_lag}")
    header_lines = str(fitted.summary()).splitlines()[:8]
    print("\n".join(header_lines))

    target_d1 = f"{target_col}_d1"
    if target_d1 not in clean.columns:
        print(f"[WARN] '{target_d1}' not found in differenced columns.")
        return

    print(f"\n[Granger Causality -> '{target_d1}'] (maxlag={best_lag})")
    for col in clean.columns:
        if col == target_d1:
            continue
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            res = grangercausalitytests(clean[[target_d1, col]], maxlag=best_lag)
        p_vals = {
            f"lag_{l}": round(float(res[l][0]["ssr_ftest"][1]), 4)
            for l in range(1, best_lag + 1)
        }
        print(f"  {col} -> {target_d1}: {p_vals}")


def build_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int] = [1, 2, 3]) -> pd.DataFrame:
    #Creates stationary log-return / diff features and lagged predictors without look-ahead
    feat = pd.DataFrame(index=df.index)
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        if (s.dropna() > 0).all():
            base = np.log(s).diff()
        else:
            base = s.diff()
        feat[f"{col}_d1"] = base
        for lag in lags:
            feat[f"{col}_d1_lag{lag}"] = base.shift(lag)
    return feat


def evaluate_time_series_models(feat_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    #Evaluates Ridge and tree/boosting models using expanding TimeSeriesSplit
    target_d1 = f"{target_col}_d1"
    target_next = feat_df[target_d1].shift(-1)
    feature_cols = [c for c in feat_df.columns if not c.startswith(target_d1) or "_lag" in c]
    data = feat_df[feature_cols].copy()
    data["y_next"] = target_next
    data = data.dropna()
    print(f"[ML] rows={len(data)}, target_col='{target_col}'")

    X = data[feature_cols].to_numpy()
    y = data["y_next"].to_numpy()

    tscv = TimeSeriesSplit(n_splits=min(4, max(2, len(data) // 12)))
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        "HistGBM": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=150, random_state=42),
    }

    results = []
    for name, model in models.items():
        rmses, maes = [], []
        for train_idx, test_idx in tscv.split(X):
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict(X[test_idx])
            rmses.append(np.sqrt(mean_squared_error(y[test_idx], preds)))
            maes.append(mean_absolute_error(y[test_idx], preds))
        results.append({
            "model": name,
            "cv_rmse": round(float(np.mean(rmses)), 5),
            "cv_mae": round(float(np.mean(maes)), 5),
        })
    return pd.DataFrame(results)


def main() -> None:
    df = load_panel()
    selected_cols = [
        c for c in [DEFAULT_TARGET_COL, *DEFAULT_DRIVERS]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not selected_cols:
        raise ValueError("The panel contains no numeric target or driver columns.")
    target_col = DEFAULT_TARGET_COL if DEFAULT_TARGET_COL in selected_cols else selected_cols[0]

    print("=== 1. Stationarity Diagnostics ===")
    stat_summary = run_stationarity_checks(df, selected_cols)
    print(stat_summary.to_string(index=False))

    feat_df = build_lag_features(df, selected_cols, lags=[1, 2, 3])
    d1_cols = [f"{c}_d1" for c in selected_cols]

    print("\n=== 2. VAR & Granger Causality on First-Differenced / Growth Series ===")
    run_var_and_granger(feat_df[d1_cols], target_col=target_col, maxlags=3)

    print("\n=== 3. TimeSeriesSplit AI/ML Baselines (Predicting Next-Period BMCI Change) ===")
    ml_summary = evaluate_time_series_models(feat_df, target_col=target_col)
    print(ml_summary.to_string(index=False))

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    stat_summary.to_csv(output_dir / "stationarity_summary.csv", index=False)
    ml_summary.to_csv(output_dir / "ml_model_comparison.csv", index=False)
    print(f"\n[INFO] Summary tables saved to '{output_dir}'.")


if __name__ == "__main__":
    main()