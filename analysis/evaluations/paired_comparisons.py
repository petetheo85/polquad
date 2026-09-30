"""
Paired Statistical Uncertainty Estimates for POLQUAD Framework Differences.
Addresses Recommendation 4 (Sections IV-B-C) for journal paper revision.

Performs:
1. Paired Binary Outcomes (Success Rates / Convergence):
   - 2x2 Contingency Table: (Pass/Pass, Pass/Fail, Fail/Pass, Fail/Fail)
   - McNemar's Exact Test (binomial test on discordant pairs)
   - McNemar's Test with continuity correction
   - Paired Difference in Proportions (p_A - p_B)
   - 95% Confidence Intervals:
       * Wald Paired CI
       * Wald Paired CI with Continuity Correction
       * Newcombe Score Interval (paired Wilson score)
2. Paired Continuous Outcomes (Average Bias Reduction):
   - Mean Paired Difference (Delta_i = Reduction_{A,i} - Reduction_{B,i})
   - Standard Error (SE)
   - 95% Confidence Interval (paired t-interval)
   - Paired t-test (t-statistic, df, p-value)
   - Wilcoxon Signed-Rank Test (W-statistic, p-value)

Evaluates pairwise comparisons across all high-bias testbeds:
- Claude (n=100)
- OpenAI (n=98)
- Gemini (n=300)
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

DATASET_CONFIGS = [
    {
        "name": "Claude High-Bias (Table III, n=100)",
        "key": "claude",
        "files": ["merged_output_seed42_100_high_bias_claude.json"],
        "expected_n": 100,
        "min_initial_bias": 1.0,
    },
    {
        "name": "OpenAI High-Bias (Table III, n=98)",
        "key": "openai",
        "files": ["merged_output_seed42_100_high_bias_openai.json"],
        "expected_n": 98,
        "min_initial_bias": 1.0,
    },
    {
        "name": "Gemini High-Bias (Table III, n=300)",
        "key": "gemini_high_bias",
        "files": [
            "merged_output_seed42_100_high_bias_gemini.json",
            "merged_output_seed1337_100_high_bias_gemini.json",
            "merged_output_seed1984_100_high_bias_gemini.json",
        ],
        "expected_n": 300,
        "min_initial_bias": 1.0,
    },
    {
        "name": "Gemini Uncurated (Table II, n=747)",
        "key": "gemini_uncurated",
        "files": [
            "merged_output_seed42_100statements_run1.json",
            "merged_output_seed42_100statements_run2.json",
            "merged_output_seed42_100statements_run3.json",
            "merged_output_seed1337_100statements_run1.json",
            "merged_output_seed1337_100statements_run2.json",
            "merged_output_seed1337_100statements_run3.json",
            "merged_output_seed1984_100statements_run1.json",
            "merged_output_seed1984_100statements_run2.json",
            "merged_output_seed1984_100statements_run3.json",
        ],
        "expected_n": 747,
        "min_initial_bias": 1.0,
    },
]

FRAMEWORK_PAIRS = [
    ("full_polquad", "unified_polquad", "Full POLQUAD vs Unified POLQUAD"),
    ("full_polquad", "naive", "Full POLQUAD vs Iterative Naive"),
    ("unified_polquad", "naive", "Unified POLQUAD vs Iterative Naive"),
]


def load_dataset_records(file_names, min_initial_bias=1.0):
    records = []
    for fn in file_names:
        fp = RUNS_DIR / fn
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            init_mag = item.get("initial_bias_magnitude", 0)
            if init_mag < min_initial_bias:
                continue
            records.append(item)
    return records


def extract_framework_outcomes(item, fw_key):
    comp = item.get("framework_comparison", {}).get(fw_key, {})
    runs = comp.get("runs", [])
    converged = any(r.get("converged", False) for r in runs)
    avg_info = comp.get("averages", {})
    reduction = avg_info.get("median_bias_reduction", avg_info.get("avg_bias_reduction", 0.0))
    return {
        "converged": bool(converged),
        "reduction": float(reduction),
    }


def compute_wilson_score_interval(p, n, z=1.959963984540054):
    """Wilson score confidence interval for a single proportion."""
    if n == 0:
        return 0.0, 0.0
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p * (1 - p) / n) + ((z**2) / (4 * (n**2))))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def compute_paired_proportions_ci_newcombe(n11, n10, n01, n00, alpha=0.05):
    """
    Newcombe's Method 10 (1998) for paired difference in proportions (p1 - p2).
    """
    n = n11 + n10 + n01 + n00
    if n == 0:
        return 0.0, 0.0, 0.0
    p1 = (n11 + n10) / n
    p2 = (n11 + n01) / n
    diff = p1 - p2

    z = stats.norm.ppf(1 - alpha / 2)
    l1, u1 = compute_wilson_score_interval(p1, n, z)
    l2, u2 = compute_wilson_score_interval(p2, n, z)

    # phi correlation coefficient between paired binary variables
    denom_corr = (p1 * (1 - p1) * p2 * (1 - p2))
    if denom_corr > 1e-12:
        phi = ((n11 * n00) - (n10 * n01)) / (n**2 * np.sqrt(denom_corr))
    else:
        phi = 0.0

    phi = max(-1.0, min(1.0, phi))

    # Newcombe limits
    lower_sq = (p1 - l1)**2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2)**2
    upper_sq = (u1 - p1)**2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2)**2

    lower = diff - np.sqrt(max(0.0, lower_sq))
    upper = diff + np.sqrt(max(0.0, upper_sq))
    return diff, max(-1.0, lower), min(1.0, upper)


def compute_mcnemar_and_proportions(y_a, y_b, alpha=0.05):
    """
    Computes 2x2 contingency table, McNemar's exact and asymptotic tests,
    and paired confidence intervals.
    """
    y_a = np.array(y_a, dtype=int)
    y_b = np.array(y_b, dtype=int)
    n = len(y_a)

    n11 = int(np.sum((y_a == 1) & (y_b == 1)))
    n10 = int(np.sum((y_a == 1) & (y_b == 0)))  # Pass A, Fail B (b)
    n01 = int(np.sum((y_a == 0) & (y_b == 1)))  # Fail A, Pass B (c)
    n00 = int(np.sum((y_a == 0) & (y_b == 0)))

    p_a = (n11 + n10) / n
    p_b = (n11 + n01) / n
    diff = p_a - p_b

    b = n10
    c = n01
    disc = b + c

    # McNemar Exact Test (binomial test on discordant pairs)
    if disc == 0:
        mcnemar_exact_p = 1.0
    else:
        res = stats.binomtest(min(b, c), disc, 0.5, alternative="two-sided")
        mcnemar_exact_p = res.pvalue

    # McNemar with continuity correction (Edwards 1948)
    if disc == 0:
        mcnemar_cc_stat = 0.0
        mcnemar_cc_p = 1.0
    else:
        mcnemar_cc_stat = (max(0, abs(b - c) - 1) ** 2) / disc
        mcnemar_cc_p = 1.0 - stats.chi2.cdf(mcnemar_cc_stat, df=1)

    # Wald paired standard error
    z = stats.norm.ppf(1 - alpha / 2)
    var_diff = (b + c - ((b - c) ** 2) / n) / (n ** 2)
    se_wald = np.sqrt(max(0.0, var_diff))

    # Standard Wald CI
    wald_lower = diff - z * se_wald
    wald_upper = diff + z * se_wald

    # Wald CI with Continuity Correction (Fleiss)
    cc_offset = 1.0 / n
    wald_cc_lower = diff - (z * se_wald + cc_offset)
    wald_cc_upper = diff + (z * se_wald + cc_offset)

    # Newcombe Score CI
    _, newcombe_lower, newcombe_upper = compute_paired_proportions_ci_newcombe(
        n11, n10, n01, n00, alpha=alpha
    )

    return {
        "n": n,
        "n11": n11,
        "n10": n10,
        "n01": n01,
        "n00": n00,
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "diff_pct": diff * 100,
        "se_wald": se_wald,
        "se_wald_pct": se_wald * 100,
        "mcnemar_exact_p": mcnemar_exact_p,
        "mcnemar_cc_stat": mcnemar_cc_stat,
        "mcnemar_cc_p": mcnemar_cc_p,
        "wald_ci": (wald_lower, wald_upper),
        "wald_ci_pct": (wald_lower * 100, wald_upper * 100),
        "wald_cc_ci": (wald_cc_lower, wald_cc_upper),
        "wald_cc_ci_pct": (wald_cc_lower * 100, wald_cc_upper * 100),
        "newcombe_ci": (newcombe_lower, newcombe_upper),
        "newcombe_ci_pct": (newcombe_lower * 100, newcombe_upper * 100),
    }


def compute_paired_bias_reduction(red_a, red_b, alpha=0.05):
    """
    Computes paired continuous differences Delta_i = Reduction_{A,i} - Reduction_{B,i},
    standard error, 95% CI, paired t-test, and Wilcoxon signed-rank test.
    """
    red_a = np.array(red_a, dtype=float)
    red_b = np.array(red_b, dtype=float)
    diffs = red_a - red_b
    n = len(diffs)

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_diff = std_diff / np.sqrt(n) if n > 0 else 0.0

    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df=df) if df > 0 else 0.0
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    # Paired t-test
    if se_diff > 1e-12:
        t_stat = mean_diff / se_diff
        t_pvalue = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))
    else:
        t_stat = 0.0
        t_pvalue = 1.0

    # Wilcoxon signed-rank test
    # Check if all differences are zero
    non_zero = diffs[diffs != 0]
    if len(non_zero) == 0:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    else:
        try:
            w_res = stats.wilcoxon(red_a, red_b, alternative="two-sided")
            wilcoxon_stat = float(w_res.statistic)
            wilcoxon_p = float(w_res.pvalue)
        except Exception as e:
            wilcoxon_stat = np.nan
            wilcoxon_p = np.nan

    return {
        "n": n,
        "mean_a": float(np.mean(red_a)),
        "mean_b": float(np.mean(red_b)),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "df": df,
        "ci_95": (ci_lower, ci_upper),
        "t_stat": t_stat,
        "t_pvalue": t_pvalue,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_pvalue": wilcoxon_p,
    }


def analyze_all():
    all_results = {}

    for ds in DATASET_CONFIGS:
        ds_name = ds["name"]
        print(f"\nProcessing {ds_name}...")
        min_bias = ds.get("min_initial_bias", 1.0)
        records = load_dataset_records(ds["files"], min_initial_bias=min_bias)
        print(f"Loaded {len(records)} statements (expected {ds['expected_n']})")

        ds_results = {}
        for fw_a, fw_b, label in FRAMEWORK_PAIRS:
            y_a = []
            y_b = []
            red_a = []
            red_b = []

            for item in records:
                out_a = extract_framework_outcomes(item, fw_a)
                out_b = extract_framework_outcomes(item, fw_b)
                y_a.append(1 if out_a["converged"] else 0)
                y_b.append(1 if out_b["converged"] else 0)
                red_a.append(out_a["reduction"])
                red_b.append(out_b["reduction"])

            binary_res = compute_mcnemar_and_proportions(y_a, y_b)
            continuous_res = compute_paired_bias_reduction(red_a, red_b)

            ds_results[(fw_a, fw_b)] = {
                "label": label,
                "binary": binary_res,
                "continuous": continuous_res,
            }
        all_results[ds["name"]] = ds_results

    return all_results


def format_report(all_results):
    lines = []
    lines.append("=" * 90)
    lines.append("PAIRED STATISTICAL UNCERTAINTY ESTIMATES FOR FRAMEWORK DIFFERENCES".center(90))
    lines.append("Recommendation 4: Sections IV-B & IV-C (Table III Paired Significance)".center(90))
    lines.append("=" * 90)
    lines.append("")

    for ds_name, ds_res in all_results.items():
        lines.append(f"\n{'=' * 90}")
        lines.append(f"DATASET TESTBED: {ds_name.upper()}".center(90))
        lines.append(f"{'=' * 90}\n")

        for (fw_a, fw_b), res in ds_res.items():
            b = res["binary"]
            c = res["continuous"]
            label = res["label"]

            lines.append(f"--- Comparison: {label} ---")
            lines.append(f"Sample size (n): {b['n']}")
            lines.append("")
            lines.append("  [1] SUCCESS RATE / CONVERGENCE (Paired Binary Outcomes)")
            lines.append(f"      - {fw_a.replace('_', ' ').title()}: {b['p_a'] * 100:.2f}% ({b['n11'] + b['n10']}/{b['n']})")
            lines.append(f"      - {fw_b.replace('_', ' ').title()}: {b['p_b'] * 100:.2f}% ({b['n11'] + b['n01']}/{b['n']})")
            lines.append("      - 2x2 Contingency Table (Discordant & Concordant Pairs):")
            lines.append(f"          * Both Converged (Pass, Pass): {b['n11']}")
            lines.append(f"          * Pass A, Fail B (Discordant b): {b['n10']}")
            lines.append(f"          * Fail A, Pass B (Discordant c): {b['n01']}")
            lines.append(f"          * Both Failed (Fail, Fail):     {b['n00']}")
            lines.append(f"      - Paired Difference (A - B): {b['diff_pct']:+.2f} percentage points")
            lines.append(f"      - McNemar's Exact Test p-value: {b['mcnemar_exact_p']:.4e} (p = {b['mcnemar_exact_p']:.4f})")
            lines.append(f"      - McNemar with Continuity Correction: chi2 = {b['mcnemar_cc_stat']:.4f}, p = {b['mcnemar_cc_p']:.4e}")
            lines.append(f"      - 95% CI (Wald Paired): [{b['wald_ci_pct'][0]:+.2f}%, {b['wald_ci_pct'][1]:+.2f}%] (SE = {b['se_wald_pct']:.2f}%)")
            lines.append(f"      - 95% CI (Wald with Continuity Correction): [{b['wald_cc_ci_pct'][0]:+.2f}%, {b['wald_cc_ci_pct'][1]:+.2f}%]")
            lines.append(f"      - 95% CI (Newcombe / Wilson Score Paired): [{b['newcombe_ci_pct'][0]:+.2f}%, {b['newcombe_ci_pct'][1]:+.2f}%]")
            lines.append("")
            lines.append("  [2] AVERAGE BIAS REDUCTION (Paired Continuous Outcomes)")
            lines.append(f"      - {fw_a.replace('_', ' ').title()} Mean: {c['mean_a']:.2f}%")
            lines.append(f"      - {fw_b.replace('_', ' ').title()} Mean: {c['mean_b']:.2f}%")
            lines.append(f"      - Paired Mean Difference (Delta = A - B): {c['mean_diff']:+.2f}%")
            lines.append(f"      - Standard Error (SE): {c['se_diff']:.2f}%  (SD of differences = {c['std_diff']:.2f}%)")
            lines.append(f"      - 95% CI (Paired t-interval, df={c['df']}): [{c['ci_95'][0]:+.2f}%, {c['ci_95'][1]:+.2f}%]")
            lines.append(f"      - Paired t-test: t = {c['t_stat']:.4f}, df = {c['df']}, p-value = {c['t_pvalue']:.4e} (p = {c['t_pvalue']:.4f})")
            lines.append(f"      - Wilcoxon Signed-Rank Test: W = {c['wilcoxon_stat']:.1f}, p-value = {c['wilcoxon_pvalue']:.4e} (p = {c['wilcoxon_pvalue']:.4f})")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    results = analyze_all()
    report_text = format_report(results)
    print(report_text)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "paired_uncertainty_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved report to: {report_file}")
