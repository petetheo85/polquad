"""
POLQUAD Benchmark Statement Debiasing Pipeline.
Generates debiased rewrites across Full POLQUAD, Unified POLQUAD, and Iterative Naive
for the canonical 100 high-bias statements (Seed 42, Gemini).
Saves output with complete iteration histories to a dedicated, isolated JSON file.
"""

import json
import sys
import warnings
from pathlib import Path

# Suppress Pydantic warning from google-genai on Python 3.14
warnings.filterwarnings("ignore", message=".*is not a Python type.*")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from polquad.utils.gemini_client import GeminiClient
from polquad.utils.bias_calculator import BiasCalculator
from polquad.frameworks.polquad import PolquadFramework
from polquad.frameworks.unified import UnifiedPolquadFramework
from polquad.frameworks.naive import NaiveFramework

INPUT_BENCHMARK_FILE = PROJECT_ROOT / "outputs" / "runs" / "merged_output_seed42_100_high_bias_gemini.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "runs" / "debiased_statements_seed42_100_gemini.json"

CONFIG = {
    "max_iters": 3,
    "bias_threshold": 2.5,
    "bias_calc_runs": 1,
    "verbose": False
}


def load_input_statements(filepath: Path) -> list:
    """Reads benchmark statements in read-only mode."""
    print(f"Loading input statements from: {filepath.name}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} statements.")
    return data


def load_checkpoint(filepath: Path) -> dict:
    """Loads existing progress if present."""
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                records = json.load(f)
            print(f"Found existing checkpoint with {len(records)} statements processed.")
            return {r["index"]: r for r in records}
        except Exception as e:
            print(f"Warning: could not read checkpoint ({e}), starting fresh.")
    return {}


def save_results(results_dict: dict, filepath: Path):
    """Atomically writes results to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = filepath.with_suffix(".tmp")
    data_list = sorted(results_dict.values(), key=lambda x: x["index"])
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2)
    temp_path.replace(filepath)


def extract_final_statement(history: dict, fallback: str) -> str:
    """Extracts the final moderated statement text from moderation history."""
    if not history:
        return fallback
    numeric_keys = [int(k) for k in history.keys()]
    max_key = max(numeric_keys)
    last_entry = history.get(max_key) or history.get(str(max_key)) or {}
    return last_entry.get("moderated_statement", fallback)


def run_pipeline(limit: int = 100):
    client = GeminiClient()
    bias_calculator = BiasCalculator(client, CONFIG)

    frameworks = {
        "naive": NaiveFramework(config=CONFIG, client=client, bias_calculator=bias_calculator),
        "unified_polquad": UnifiedPolquadFramework(config=CONFIG, client=client, bias_calculator=bias_calculator),
        "full_polquad": PolquadFramework(config=CONFIG, client=client, bias_calculator=bias_calculator),
    }

    input_data = load_input_statements(INPUT_BENCHMARK_FILE)
    completed_records = load_checkpoint(OUTPUT_FILE)

    statements_to_process = input_data[:limit]
    total = len(statements_to_process)

    print(f"\nStarting debiasing run on {total} statements (max_iters={CONFIG['max_iters']}, threshold={CONFIG['bias_threshold']})...\n")

    for idx, item in enumerate(statements_to_process, start=1):
        statement_idx = item["index"]
        original_statement = item["original_statement"]

        if statement_idx in completed_records:
            print(f"[{idx}/{total}] Skipping Statement {statement_idx} (already completed).")
            continue

        print(f"[{idx}/{total}] Processing Statement {statement_idx}: \"{original_statement[:60]}...\"")

        coords, mag = bias_calculator.calculate_bias(original_statement)
        row = {"text": original_statement}

        record = {
            "index": statement_idx,
            "original_statement": original_statement,
            "initial_bias_coords": coords,
            "initial_bias_magnitude": mag,
            "frameworks": {}
        }

        for fw_name in ["naive", "unified_polquad", "full_polquad"]:
            fw = frameworks[fw_name]
            try:
                res = fw.run_analysis(index=statement_idx, row=row, initial_bias_coords=coords, initial_bias_mag=mag)
                fw_data = res.get("frameworks", {}).get(fw_name, {})
                history = fw_data.get("moderation_history", {})
                final_text = extract_final_statement(history, original_statement)

                record["frameworks"][fw_name] = {
                    "final_statement": final_text,
                    "final_bias_magnitude": fw_data.get("final_bias_magnitude", mag),
                    "bias_reduction": fw_data.get("bias_reduction", 0.0),
                    "num_iterations": fw_data.get("num_iterations", 0),
                    "converged": fw_data.get("converged", False),
                    "moderation_history": history
                }
                print(f"   ↳ {fw_name:<16}: Red={fw_data.get('bias_reduction', 0.0):.1f}% | Iters={fw_data.get('num_iterations', 0)} | Conv={fw_data.get('converged', False)}")
            except Exception as e:
                print(f"   ↳ {fw_name:<16}: ERROR ({e})")
                record["frameworks"][fw_name] = {
                    "error": str(e),
                    "final_statement": original_statement,
                    "final_bias_magnitude": mag,
                    "bias_reduction": 0.0,
                    "num_iterations": 0,
                    "converged": False
                }

        completed_records[statement_idx] = record
        save_results(completed_records, OUTPUT_FILE)

    print(f"\nDebiasing generation complete! Saved {len(completed_records)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_pipeline()
