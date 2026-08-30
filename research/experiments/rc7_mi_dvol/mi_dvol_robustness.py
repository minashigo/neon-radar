from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from scipy.stats import spearmanr


def run_ols(x_series, y_series):
    if len(x_series) < 30:
        return np.nan, np.nan
    x_vars = sm.add_constant(x_series)
    try:
        model = sm.OLS(y_series, x_vars).fit()
        return model.params.iloc[1], model.pvalues.iloc[1]
    except Exception:
        return np.nan, np.nan

def phase1_wfa(df, feature):
    print(f"\n--- Phase 1: Walk-Forward Analysis for {feature} ---")
    df['date'] = pd.to_datetime(df['date'])
    start_date = df['date'].min()
    end_date = df['date'].max()

    current_start = start_date
    oos_results = []

    while True:
        is_end = current_start + relativedelta(months=6)
        oos_end = is_end + relativedelta(months=2)
        if oos_end > end_date:
            break

        oos_df = df[(df['date'] >= is_end) & (df['date'] < oos_end)].dropna(subset=[feature, 'fwd_ret_7d'])
        if len(oos_df) >= 10:
            coef, _ = run_ols(oos_df[feature], oos_df['fwd_ret_7d'])
            if not np.isnan(coef):
                oos_results.append(coef)

        current_start += relativedelta(months=2) # Step forward by OOS length

    if oos_results:
        pos_ratio = sum(1 for c in oos_results if c > 0) / len(oos_results)
        print(f"OOS Periods Evaluated: {len(oos_results)}")
        print(f"Percentage Positive Coefs: {pos_ratio*100:.1f}%")
        print(f"Mean Coef: {np.mean(oos_results):.6f}, Std: {np.std(oos_results):.6f}")
    else:
        print("Not enough OOS periods.")
    return oos_results

def phase2_bootstrap(df, feature, iterations=1000):
    print(f"\n--- Phase 2: Bootstrap for {feature} ---")
    clean_df = df.dropna(subset=[feature, 'fwd_ret_7d']).copy()
    coefs = []

    n = len(clean_df)
    for _ in range(iterations):
        sample = clean_df.sample(n=n, replace=True)
        coef, _ = run_ols(sample[feature], sample['fwd_ret_7d'])
        if not np.isnan(coef):
            coefs.append(coef)

    if coefs:
        coefs = np.array(coefs)
        mean_coef = np.mean(coefs)
        median_coef = np.median(coefs)
        ci_lower = np.percentile(coefs, 2.5)
        ci_upper = np.percentile(coefs, 97.5)
        pos_ratio = sum(1 for c in coefs if c > 0) / len(coefs)

        print(f"Iterations: {len(coefs)}")
        print(f"Mean: {mean_coef:.6f}, Median: {median_coef:.6f}")
        print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"Stability (Same Sign as Mean): {pos_ratio*100 if mean_coef > 0 else (1-pos_ratio)*100:.1f}%")
    else:
        print("Bootstrap failed.")

def phase3_regimes(df, feature):
    print(f"\n--- Phase 3: Regime Dependence for {feature} ---")
    clean_df = df.dropna(subset=[feature, 'fwd_ret_7d', 'regime']).copy()

    regimes = clean_df['regime'].unique()
    for regime in regimes:
        regime_df = clean_df[clean_df['regime'] == regime]
        coef, pval = run_ols(regime_df[feature], regime_df['fwd_ret_7d'])
        corr, corr_pval = spearmanr(regime_df[feature], regime_df['fwd_ret_7d'])
        print(f"Regime: {regime:<15} (n={len(regime_df):<4}) | OLS Coef: {coef:.6f} (p={pval:.4f}) | Spearman: {corr:.4f} (p={corr_pval:.4f})")

def main():
    csv_path = Path("C:/Users/orphan/.gemini/antigravity/brain/fb6f9fda-e97a-4713-83ed-9f3f4425bb3d/scratch/mi_features_outcomes.csv")
    df = pd.read_csv(csv_path)

    features = ['dvol_value', 'dvol_z_score_30d', 'dvol_percentile_30d']

    for feature in features:
        print("="*60)
        print(f"EVALUATING REPRESENTATION: {feature}")
        print("="*60)
        phase1_wfa(df, feature)
        phase2_bootstrap(df, feature)
        phase3_regimes(df, feature)

if __name__ == "__main__":
    main()
