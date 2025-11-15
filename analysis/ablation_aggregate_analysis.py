import sys
import json
from pathlib import Path
from collections import defaultdict
import statistics

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))
from config import polquad_configs

# Input files
INPUT_FILES = [
    project_root / "results/outputs/ablation_output_seed42_100_high_bias_gemini.json",
    project_root / "results/outputs/ablation_output_seed1337_100_high_bias_gemini.json",
    project_root / "results/outputs/ablation_output_seed1984_100_high_bias_gemini.json"
]

OUTPUT_REPORT = project_root / "results/analysis/ablation_aggregate.txt"


def get_quadrant_from_coords(coords):
    """Determines the political quadrant from (x, y) coordinates."""
    x = coords.get('x', 0)
    y = coords.get('y', 0)
    
    if x > 0 and y > 0:
        return "lib_right"
    elif x < 0 and y > 0:
        return "lib_left"
    elif x < 0 and y < 0:
        return "auth_left"
    elif x > 0 and y < 0:
        return "auth_right"
    else:
        return "center"


def get_ablation_display_name(ablation_name):
    """Returns a readable display name for the ablation."""
    mapping = {
        "full_polquad": "Full POLQUAD (All 4)",
        "al_only": "AL Only",
        "ar_only": "AR Only",
        "ll_only": "LL Only",
        "lr_only": "LR Only",
        "al_ar": "AL + AR (Authoritarion)",
        "ll_lr": "LL + LR (Libertarian)",
        "al_ll": "AL + LL (Left)",
        "ar_lr": "AR + LR (Right)",
        "al_lr": "AL + LR (Diagonal)",
        "ar_ll": "AR + LL (Diagonal)"
    }
    return mapping.get(ablation_name, ablation_name)


def load_results(file_path):
    """Load results from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load {file_path}. Details: {e}")
        return []


def aggregate_ablation_results(all_results_by_seed, bias_threshold, bias_bins):
    """
    Aggregates results across seeds by ablation name.
    Returns a dictionary of ablation statistics.
    """
    ablation_stats = defaultdict(lambda: {
        "all_reductions": [],
        "successful_runs": 0,
        "total_runs": 0,
        "all_iterations": [],
        "successful_iterations": [],
        "by_quadrant": defaultdict(list),
        "by_initial_bias": defaultdict(list), # This is the target dictionary
        "converged": 0,
        "overcorrections": 0,
        "num_statements": 0
    })
    
    # Process each seed's results
    for seed_results in all_results_by_seed:
        for result in seed_results:
            initial_mag = result.get("initial_bias_magnitude", 0)
            initial_coords = result.get("initial_bias_coords", {'x': 0, 'y': 0})
            ablation_name = result.get("ablation_name")
            
            # Skip statements below threshold
            if initial_mag <= bias_threshold:
                continue
            
            # Skip Full POLQUAD
            if ablation_name == "full_polquad":
                continue
            
            # --- START: MISSING BINNING LOGIC ---
            current_bias_bin_label = None
            for low, high, label in bias_bins:
                if low <= initial_mag < high:
                    current_bias_bin_label = label
                    break
            # --- END: MISSING BINNING LOGIC ---

            ablation_stats[ablation_name]["num_statements"] += 1
            calculated_quadrant = get_quadrant_from_coords(initial_coords)
            
            fw_results = result.get("framework_results", {})
            runs = fw_results.get("runs", [])
            averages = fw_results.get("averages", {})
            
            # Extract metrics
            avg_reduction = averages.get("avg_bias_reduction", 0)
            avg_iters = averages.get("avg_iterations", 0)
            
            ablation_stats[ablation_name]["all_reductions"].append(avg_reduction)
            ablation_stats[ablation_name]["by_quadrant"][calculated_quadrant].append(avg_reduction)
            ablation_stats[ablation_name]["all_iterations"].append(avg_iters)
            
            # Populate the bias magnitude dictionary
            if current_bias_bin_label:
                ablation_stats[ablation_name]["by_initial_bias"][current_bias_bin_label].append(avg_reduction)
            
            # Check convergence
            converged = any(run.get("converged", False) for run in runs)
            if converged:
                ablation_stats[ablation_name]["converged"] += 1
                ablation_stats[ablation_name]["successful_iterations"].append(avg_iters)
            
            ablation_stats[ablation_name]["total_runs"] += len(runs)
            ablation_stats[ablation_name]["overcorrections"] += sum(1 for run in runs if run.get("bias_reduction", 0) < 0)
    
    return ablation_stats


def generate_report(ablation_stats, bias_bins):
    """Generate and print the analysis report."""
    print("="*80)
    print("ABLATION STUDY ANALYSIS - AGGREGATED ACROSS 3 SEEDS (42, 1337, 1984)".center(80))
    print("="*80)
    
    print("\n--- Overall Ablation Performance ---\n")
    
    # Calculate total statements
    total_statements = sum(stats["num_statements"] for stats in ablation_stats.values())
    print(f"Total Unique Statements Processed (n={total_statements})\n")
    
    if total_statements == 0:
        print("  No data found meeting the bias threshold.\n")
        print("="*80 + "\n\n")
        return
    
    # Sort by average bias reduction (descending)
    sorted_ablations = sorted(
        ablation_stats.items(),
        key=lambda x: statistics.mean(x[1]["all_reductions"]) if x[1]["all_reductions"] else 0,
        reverse=True
    )
    
    # --- Overall Ablation Performance (Report Logic Omitted for brevity, assumed correct) ---
    for ablation_name, stats in sorted_ablations:
        avg_reduction = statistics.mean(stats["all_reductions"]) if stats["all_reductions"] else 0
        success_rate = (stats["converged"] / stats["num_statements"] * 100) if stats["num_statements"] > 0 else 0
        overcorrection_rate = (stats["overcorrections"] / stats["total_runs"] * 100) if stats["total_runs"] > 0 else 0
        
        display_name = get_ablation_display_name(ablation_name)
        print(f"  Ablation: {display_name}")
        print(f"    - Average Bias Reduction: {avg_reduction:.2f}%")
        print(f"    - Success Rate (met threshold): {success_rate:.2f}%")
        print(f"    - Overcorrection Rate (negative reduction): {overcorrection_rate:.2f}%")
    
    # Performance by quadrant
    print("\n--- Average Bias Reduction by Calculated Quadrant ---\n")
    
    quadrants = ["auth_left", "auth_right", "lib_left", "lib_right", "center"]
    header = f"{'Ablation':<20}" + "".join([f"{q.upper():<15}" for q in quadrants])
    print(header)
    print("-" * len(header))
    
    for ablation_name, stats in sorted_ablations:
        display_name = get_ablation_display_name(ablation_name)
        row_str = f"{display_name:<20}"
        
        for quad in quadrants:
            reductions = stats["by_quadrant"][quad]
            count = len(reductions)
            avg_reduction = statistics.mean(reductions) if reductions else 0
            cell_text = f"{avg_reduction:.1f}% (n={count})"
            row_str += f"{cell_text:<15}"
        
        print(row_str)
    
    # --- Performance by Initial Bias Magnitude (CORRECTED LOGIC) ---
    print("\n--- Performance by Initial Bias Magnitude ---\n")
    
    # Extract labels from the passed-in list of tuples
    bias_bins_labels = [label for _, _, label in bias_bins] 
    header = f"{'Ablation':<20}" + "".join([f"{bin_label:<25}" for bin_label in bias_bins_labels])
    print(header)
    print("-" * len(header))
    
    for ablation_name, stats in sorted_ablations:
        display_name = get_ablation_display_name(ablation_name)
        row_str = f"{display_name:<20}"
        
        for bin_label in bias_bins_labels:
            # This is now correctly populated in the aggregation function
            reductions_in_bin = stats["by_initial_bias"][bin_label] 
            count = len(reductions_in_bin)
            avg_reduction = statistics.mean(reductions_in_bin) if reductions_in_bin else 0
            cell_text = f"{avg_reduction:.2f}% (n={count})"
            row_str += f"{cell_text:<25}"
        
        print(row_str)
    
    # --- Efficiency Analysis (Report Logic Omitted for brevity, assumed correct) ---
    print("\n--- Efficiency Analysis ---\n")
    
    for ablation_name, stats in sorted_ablations:
        display_name = get_ablation_display_name(ablation_name)
        
        avg_iters = statistics.mean(stats["all_iterations"]) if stats["all_iterations"] else 0
        avg_succ_iters = statistics.mean(stats["successful_iterations"]) if stats["successful_iterations"] else 0
        
        print(f"  Ablation: {display_name}")
        print(f"    - Average Iterations (All Runs): {avg_iters:.2f}")
        print(f"    - Average Iterations (Successful Runs): {avg_succ_iters:.2f}")
    
    print("\n" + "="*80 + "\n\n")


def main():
    print("Loading ablation results from three seeds...\n")
    
    all_results_by_seed = []
    total_results = 0
    
    for input_file in INPUT_FILES:
        print(f"  Loading: {input_file.name}")
        results = load_results(input_file)
        all_results_by_seed.append(results)
        total_results += len(results)
        print(f"    -> Loaded {len(results)} results")
    
    bias_threshold = polquad_configs.get('bias_threshold', 2.5)
    
    # Define the bias bins with ranges (Needed for aggregation)
    bias_bins_ranges = [
        (1.1, 2.5, "Low (1.1-2.5)"),
        (2.5, 4.0, "Mid-Low (2.5-4.0)"),
        (4.0, 5.5, "Medium (4.0-5.5)"),
        (5.5, 7.0, "Mid-High (5.5-7.0)"),
        (7.0, 8.5, "High (7.0-8.5)"),
        (8.5, 10.1, "Extreme (8.5-10.0)")
    ]

    print(f"\nTotal results loaded: {total_results}")
    print(f"Bias Threshold: {bias_threshold}")
    print("\nAggregating results across seeds (excluding Full POLQUAD)...\n")
    
    # Analyze results - Pass the bias_bins_ranges
    ablation_stats = aggregate_ablation_results(all_results_by_seed, bias_threshold, bias_bins_ranges)
    
    print(f"Analyzing {len(ablation_stats)} ablation combinations...")
    
    # Generate report - Pass the bias_bins_ranges
    print(f"\nGenerating report... Saving to: {OUTPUT_REPORT}\n")
    
    original_stdout = sys.stdout
    with open(OUTPUT_REPORT, 'w') as f:
        sys.stdout = f
        generate_report(ablation_stats, bias_bins_ranges)
    
    sys.stdout = original_stdout
    print(f"Report saved to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()