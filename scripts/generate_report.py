import json

def load_results():
    with open("results/phase1_raw.json", "r", encoding="utf-8") as f:
        return json.load(f)

def generate_markdown(results):
    lines = []
    lines.append("# Neon Radar Phase 1 Validation Analytical Report\n")
    lines.append("## Executive Summary\n")
    lines.append("This report summarizes the Out-of-Sample stability and feature importance across 5 distinct market regimes (6 evaluated periods) on two core timeframes (1D, 4H). The universe includes `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, and `ADAUSDT`.\n")
    
    # Analyze the regimes
    lines.append("## 1. Market Regime Performance (Baseline)\n")
    lines.append("The table below shows the baseline performance of the current trading system (Gross vs Net).\n")
    lines.append("| Period | Timeframe | Trades | Win Rate | Gross PF | Net PF | Gross Exp | Net Exp | Sharpe | p-value |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    
    overall_net_pf = []
    overall_net_exp = []
    overall_sharpe = []
    overall_wr = []
    
    for period, tfs in results.items():
        for tf, data in tfs.items():
            b = data["baseline"]
            tr = b["total_trades"]
            wr = b["win_rate"]
            g_pf = b["gross_profit_factor"]
            n_pf = b["net_profit_factor"]
            g_exp = b["gross_expectancy"]
            n_exp = b["net_expectancy"]
            sr = b["net_sharpe_ratio"]
            pval = b["validation"]["p_value"]
            
            overall_net_pf.append(n_pf)
            overall_net_exp.append(n_exp)
            overall_sharpe.append(sr)
            overall_wr.append(wr)
            
            # Format
            lines.append(f"| {period} | {tf} | {tr} | {wr:.1%} | {g_pf:.2f} | {n_pf:.2f} | {g_exp:.2%} | {n_exp:.2%} | {sr:.2f} | {pval:.3f} |")
    
    lines.append("\n**Key Findings (Performance):**\n")
    avg_pf = sum(overall_net_pf) / len(overall_net_pf)
    avg_wr = sum(overall_wr) / len(overall_wr)
    avg_exp = sum(overall_net_exp) / len(overall_net_exp)
    avg_sharpe = sum(overall_sharpe) / len(overall_sharpe)
    
    lines.append(f"- **Overall Averaged Metrics**: Net PF = {avg_pf:.2f}, WR = {avg_wr:.1%}, Net Exp = {avg_exp:.2%}, Sharpe = {avg_sharpe:.2f}.")
    lines.append("- *Observation*: Add your analytical notes here after seeing the data.\n")
    
    lines.append("## 2. Feature Importance (Ablation Analysis)\n")
    lines.append("This section shows which features contributed most to the strategy's edge. A positive Score means the rule is helpful; a negative score means the rule is actively harming the system.\n")
    
    # Aggregate feature scores
    feature_scores = {}
    for period, tfs in results.items():
        for tf, data in tfs.items():
            for f in data.get("features", []):
                name = f["rule_name"]
                if name not in feature_scores:
                    feature_scores[name] = {"scores": [], "dPF": [], "dExp": [], "dSharpe": [], "dWR": []}
                
                feature_scores[name]["scores"].append(f["feature_score"])
                feature_scores[name]["dPF"].append(f["delta_profit_factor"])
                feature_scores[name]["dExp"].append(f["delta_expectancy"])
                feature_scores[name]["dSharpe"].append(f["delta_sharpe_ratio"])
                feature_scores[name]["dWR"].append(f["delta_win_rate"])
                
    lines.append("| Feature | Avg Score | Avg $\\Delta$PF | Avg $\\Delta$Exp | Avg $\\Delta$Sharpe | Avg $\\Delta$WR |")
    lines.append("|---|---|---|---|---|---|")
    
    # Sort features by avg score
    sorted_features = []
    for name, stats in feature_scores.items():
        avg_score = sum(stats["scores"]) / len(stats["scores"])
        avg_dpf = sum(stats["dPF"]) / len(stats["dPF"])
        avg_dexp = sum(stats["dExp"]) / len(stats["dExp"])
        avg_dsharpe = sum(stats["dSharpe"]) / len(stats["dSharpe"])
        avg_dwr = sum(stats["dWR"]) / len(stats["dWR"])
        sorted_features.append((name, avg_score, avg_dpf, avg_dexp, avg_dsharpe, avg_dwr))
        
    sorted_features.sort(key=lambda x: x[1], reverse=True)
    
    for f in sorted_features:
        lines.append(f"| {f[0]} | {f[1]:+.2f} | {f[2]:+.2f} | {f[3]:+.2%} | {f[4]:+.2f} | {f[5]:+.1%} |")
        
    lines.append("\n**Key Findings (Features):**\n")
    lines.append("- *Observation*: Add your analytical notes here based on feature importance.\n")
    
    lines.append("## 3. Conclusions & Recommendations\n")
    lines.append("1. **Edge Existence**: (Does the strategy have a statistically significant edge Gross of costs?)\n")
    lines.append("2. **Market Regime Robustness**: (Are there specific regimes where it fails?)\n")
    lines.append("3. **Rule Set Refinement**: (Which rules should be dropped or refactored?)\n")
    lines.append("4. **Next Steps**: (Move to Phase 2: Net-of-Costs simulation and true Walk-Forward validation).\n")
    
    with open("results/phase1_analytical_report_draft.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Draft generated at results/phase1_analytical_report_draft.md")

if __name__ == "__main__":
    generate_markdown(load_results())
