"""Script for analyzing extracted MI features and calculating incremental value."""

from pathlib import Path

import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


def run_analysis():
    data_path = Path(r"C:\Users\orphan\.gemini\antigravity\brain\fb6f9fda-e97a-4713-83ed-9f3f4425bb3d\scratch\mi_features_outcomes.csv")
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    df = pd.read_csv(data_path)

    # Drop rows without 7d forward return
    df = df.dropna(subset=['fwd_ret_7d'])

    print("=== Univariate Feature Analysis (Spearman Rank Correlation) ===")

    features = ['fng_value', 'fng_z_score_30d', 'fng_percentile_30d',
                'dvol_value', 'dvol_z_score_30d', 'dvol_percentile_30d']
    outcomes = ['fwd_ret_1d', 'fwd_ret_3d', 'fwd_ret_7d', 'fwd_ret_14d', 'fwd_ret_30d', 'mae_7d']

    print(f"{'Feature':<20} | {'Outcome':<15} | {'Correlation':>12} | {'p-value':>12}")
    print("-" * 65)

    for feature in features:
        if feature not in df.columns or df[feature].isna().all():
            continue

        valid_df = df.dropna(subset=[feature])

        for outcome in outcomes:
            if outcome not in valid_df.columns:
                continue

            clean_df = valid_df.dropna(subset=[outcome])
            if len(clean_df) < 30:
                continue

            corr, pval = stats.spearmanr(clean_df[feature], clean_df[outcome])

            # Print if p-value is somewhat significant or just show all
            if pval < 0.1:
                print(f"{feature:<20} | {outcome:<15} | {corr:>12.4f} | {pval:>12.4f}")

    print("\n=== Incremental Value & Orthogonality Test (OLS) ===")
    print("Testing if MI features add orthogonal information to Baseline Confidence for 7d Forward Return.")

    target = 'fwd_ret_7d'

    for feature in features:
        if feature not in df.columns or df[feature].isna().all():
            continue

        # Prepare data for OLS
        clean_df = df.dropna(subset=[feature, target, 'baseline_confidence']).copy()

        if len(clean_df) < 50:
            continue

        x_vars = clean_df[['baseline_confidence', feature]]
        x_vars = sm.add_constant(x_vars)
        y = clean_df[target]

        model = sm.OLS(y, x_vars).fit()

        print(f"\n--- OLS Results for: {feature} ---")
        coef = model.params[feature]
        p_val = model.pvalues[feature]
        conf_int = model.conf_int().loc[feature]

        print(f"Coefficient: {coef:.6f}")
        print(f"p-value:     {p_val:.6f}")
        print(f"95% CI:      [{conf_int[0]:.6f}, {conf_int[1]:.6f}]")
        print(f"Adj. R-sq:   {model.rsquared_adj:.6f}")

        if p_val < 0.05:
            print(f"YES STATISTICALLY SIGNIFICANT orthogonal edge for {feature}!")
        else:
            print(f"NO significant orthogonal edge for {feature}.")

if __name__ == "__main__":
    run_analysis()
