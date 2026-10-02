"""
Independent Blinded Evaluation & Semantic Similarity Analysis.
Evaluates generated debiased statements across Full POLQUAD, Unified, and Naive.
Computes:
1. Gemini Embedding Cosine Similarity (gemini-embedding-001)
2. Lexical ROUGE-L & Token Jaccard Similarity
3. Blinded LLM Assessment (1-5 Meaning Preservation, 1-5 Neutrality)
4. Paired Statistical Tests & Correlation Analysis (Pearson & Spearman)
"""

import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import scipy.stats as stats

# Suppress Pydantic warning from google-genai on Python 3.14
warnings.filterwarnings("ignore", message=".*is not a Python type.*")
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polquad.utils.gemini_client import GeminiClient
from polquad.agents.evaluator_agent import EvaluatorAgent

INPUT_DEBIASED_FILE = PROJECT_ROOT / "outputs" / "runs" / "debiased_statements_seed42_100_gemini.json"
OUTPUT_EVAL_FILE = PROJECT_ROOT / "outputs" / "runs" / "blinded_evaluation_results_gemini.json"
OUTPUT_REPORT_FILE = PROJECT_ROOT / "outputs" / "reports" / "blinded_semantic_evaluation_report.txt"


def cosine_similarity(v1: list, v2: list) -> float:
    """Calculates cosine similarity between two numeric vectors."""
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_rouge_l_f1(reference: str, candidate: str) -> float:
    """Computes token-level ROUGE-L F1 using Longest Common Subsequence."""
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    m, n = len(ref_tokens), len(cand_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    prec = lcs / n
    rec = lcs / m
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def compute_token_jaccard(s1: str, s2: str) -> float:
    """Computes word-level Jaccard similarity coefficient."""
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 and not w2:
        return 1.0
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    return len(intersection) / len(union) if union else 0.0


def load_checkpoint(filepath: Path) -> dict:
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {item["index"]: item for item in data}
        except Exception:
            pass
    return {}


def save_checkpoint(results_dict: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = filepath.with_suffix(".tmp")
    data_list = sorted(results_dict.values(), key=lambda x: x["index"])
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2)
    temp_path.replace(filepath)


def run_blinded_evaluation():
    if not INPUT_DEBIASED_FILE.exists():
        print(f"Error: Required debiased input file not found: {INPUT_DEBIASED_FILE}")
        print("Please run scripts/generate_debiased_statements.py first.")
        return

    with open(INPUT_DEBIASED_FILE, "r", encoding="utf-8") as f:
        statements = json.load(f)

    print(f"Loaded {len(statements)} debiased statements for evaluation.")

    client = GeminiClient()
    evaluator = EvaluatorAgent(client)

    completed = load_checkpoint(OUTPUT_EVAL_FILE)
    total = len(statements)

    print(f"\nStarting Blinded Evaluation & Similarity Analysis across {total} statements...\n")

    for idx, item in enumerate(statements, start=1):
        s_idx = item["index"]
        orig_text = item["original_statement"]

        if s_idx in completed:
            print(f"[{idx}/{total}] Skipping Statement {s_idx} (already evaluated).")
            continue

        print(f"[{idx}/{total}] Evaluating Statement {s_idx}...")

        framework_data = item.get("frameworks", {})
        candidates = {
            fw: fw_info.get("final_statement", orig_text)
            for fw, fw_info in framework_data.items()
        }

        # Embeddings & Semantic Similarity
        try:
            orig_emb = client.embed_text(orig_text)
        except Exception as e:
            print(f"   Warning: Embedding failed for original statement ({e})")
            orig_emb = None

        similarity_metrics = {}
        for fw, cand_text in candidates.items():
            if orig_emb:
                try:
                    cand_emb = client.embed_text(cand_text)
                    emb_sim = cosine_similarity(orig_emb, cand_emb)
                except Exception as e:
                    emb_sim = 0.0
            else:
                emb_sim = 0.0

            rouge_f1 = compute_rouge_l_f1(orig_text, cand_text)
            jaccard = compute_token_jaccard(orig_text, cand_text)

            similarity_metrics[fw] = {
                "embedding_cosine_similarity": emb_sim,
                "rouge_l_f1": rouge_f1,
                "token_jaccard": jaccard
            }

        # Independent Blinded Assessment
        try:
            blind_eval = evaluator.evaluate_blinded_candidates(
                original_statement=orig_text,
                candidates=candidates,
                random_seed=s_idx  # Deterministic seed per statement
            )
            unblinded = blind_eval["unblinded_evaluations"]
        except Exception as e:
            print(f"   Warning: Blinded evaluation failed ({e})")
            unblinded = {
                fw: {"meaning_preservation": 0.0, "neutrality": 0.0, "rationale": str(e)}
                for fw in candidates.keys()
            }
            blind_eval = {"blinding_map": {}}

        # Compile record
        eval_record = {
            "index": s_idx,
            "original_statement": orig_text,
            "initial_bias_magnitude": item.get("initial_bias_magnitude", 0.0),
            "blinding_map": blind_eval.get("blinding_map", {}),
            "evaluations": {}
        }

        for fw in candidates.keys():
            fw_info = framework_data.get(fw, {})
            unb = unblinded.get(fw, {})
            sim = similarity_metrics.get(fw, {})

            eval_record["evaluations"][fw] = {
                "final_statement": candidates[fw],
                "bias_reduction": fw_info.get("bias_reduction", 0.0),
                "converged": fw_info.get("converged", False),
                "num_iterations": fw_info.get("num_iterations", 0),
                "meaning_preservation": unb.get("meaning_preservation", 0.0),
                "neutrality": unb.get("neutrality", 0.0),
                "rationale": unb.get("rationale", ""),
                "embedding_cosine_similarity": sim.get("embedding_cosine_similarity", 0.0),
                "rouge_l_f1": sim.get("rouge_l_f1", 0.0),
                "token_jaccard": sim.get("token_jaccard", 0.0)
            }

            print(f"   ↳ {fw:<16}: Meaning={unb.get('meaning_preservation', 0.0):.1f} | Neut={unb.get('neutrality', 0.0):.1f} | CosSim={sim.get('embedding_cosine_similarity', 0.0):.3f}")

        completed[s_idx] = eval_record
        save_checkpoint(completed, OUTPUT_EVAL_FILE)

    print(f"\nAll evaluations complete! Results saved to {OUTPUT_EVAL_FILE}")

    # Generate Statistical Report
    generate_report(completed)


def generate_report(results_dict: dict):
    records = list(results_dict.values())
    n = len(records)
    if n == 0:
        return

    frameworks = ["naive", "unified_polquad", "full_polquad"]
    fw_labels = {
        "naive": "Iterative Naive",
        "unified_polquad": "Unified POLQUAD",
        "full_polquad": "Full POLQUAD"
    }

    metrics = [
        ("meaning_preservation", "Meaning Preservation (1-5)"),
        ("neutrality", "Perceived Neutrality (1-5)"),
        ("embedding_cosine_similarity", "Gemini Embedding Cosine Sim"),
        ("rouge_l_f1", "Lexical ROUGE-L F1"),
        ("token_jaccard", "Token Jaccard Overlap")
    ]

    # Aggregate metric values
    data = defaultdict(lambda: defaultdict(list))
    for r in records:
        for fw in frameworks:
            eval_data = r["evaluations"].get(fw, {})
            for m_key, _ in metrics:
                data[fw][m_key].append(eval_data.get(m_key, 0.0))
            data[fw]["bias_reduction"].append(eval_data.get("bias_reduction", 0.0))

    lines = []
    lines.append("=" * 86)
    lines.append("INDEPENDENT BLINDED ASSESSMENT OF NEUTRALITY & MEANING PRESERVATION".center(86))
    lines.append("Testing Semantic Fidelity and Neutrality of Framework Rewrites (n = 100)".center(86))
    lines.append("=" * 86)
    lines.append("")

    # Summary Table
    lines.append("### Table 1: Comparative Evaluation Metrics (Mean ± Standard Error)")
    lines.append("")
    header = f"| {'Metric':<32} | {fw_labels['naive']:<16} | {fw_labels['unified_polquad']:<18} | {fw_labels['full_polquad']:<16} |"
    lines.append(header)
    lines.append("|" + "-" * 34 + "|" + "-" * 18 + "|" + "-" * 20 + "|" + "-" * 18 + "|")

    for m_key, m_label in metrics:
        row = f"| {m_label:<32} "
        for fw in frameworks:
            vals = data[fw][m_key]
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            row += f"| {mean:.3f} ± {se:.3f}   "
        row += "|"
        lines.append(row)

    lines.append("")
    lines.append("=" * 86)
    lines.append("### Table 2: Paired Statistical Significance Tests (Full POLQUAD vs. Baselines)")
    lines.append("=" * 86)
    lines.append("")

    pairs = [
        ("full_polquad", "unified_polquad", "Full POLQUAD vs Unified POLQUAD"),
        ("full_polquad", "naive", "Full POLQUAD vs Iterative Naive")
    ]

    for fw_a, fw_b, pair_label in pairs:
        lines.append(f"\n--- Comparison: {pair_label} ---")
        for m_key, m_label in metrics:
            a = np.array(data[fw_a][m_key])
            b = np.array(data[fw_b][m_key])
            diff = a - b
            mean_diff = np.mean(diff)
            se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))

            # Paired t-test
            t_res = stats.ttest_rel(a, b)
            # Wilcoxon signed-rank test
            try:
                w_res = stats.wilcoxon(a, b, alternative='two-sided')
                w_stat, w_p = w_res.statistic, w_res.pvalue
            except Exception:
                w_stat, w_p = np.nan, np.nan

            lines.append(f"  • {m_label:<32}: Diff = {mean_diff:+.3f} (SE = {se_diff:.3f}), Paired t = {t_res.statistic:.3f} (p = {t_res.pvalue:.4e}), Wilcoxon p = {w_p:.4e}")

    lines.append("")
    lines.append("=" * 86)
    lines.append("### Table 3: Correlation Analysis (Movement Toward Center vs. Neutrality & Fidelity)")
    lines.append("Testing whether coordinate movement toward (0,0) represents a genuine, faithful rewrite")
    lines.append("=" * 86)
    lines.append("")

    for fw in frameworks:
        lines.append(f"\nFramework: {fw_labels[fw]}")
        red = np.array(data[fw]["bias_reduction"])
        for m_key, m_label in [("neutrality", "Perceived Neutrality"), ("meaning_preservation", "Meaning Preservation"), ("embedding_cosine_similarity", "Embedding Cosine Sim")]:
            metric_vals = np.array(data[fw][m_key])
            p_corr, p_val = stats.pearsonr(red, metric_vals)
            s_corr, s_val = stats.spearmanr(red, metric_vals)
            lines.append(f"  • Bias Reduction vs {m_label:<24}: Pearson r = {p_corr:+.3f} (p = {p_val:.4e}) | Spearman rho = {s_corr:+.3f} (p = {s_val:.4e})")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    OUTPUT_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport written to: {OUTPUT_REPORT_FILE}")


if __name__ == "__main__":
    run_blinded_evaluation()
