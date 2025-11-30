import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
from polquad.config import polquad_configs

INPUTFILE = project_root / "outputs/runs/ablation_output.json"

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


def categorize_ablation(ablation_name):
    """Categorizes ablation by type: single, pair, or full."""
    if ablation_name == "full_polquad":
        return "full"
    elif ablation_name in ["al_ar", "ll_lr", "al_ll", "ar_lr", "al_lr", "ar_ll"]:
        return "pair"
    else:
        return "single"


def get_ablation_display_name(ablation_name):
    """Returns a readable display name for the ablation."""
    mapping = {
        "full_polquad": "Full POLQUAD (All 4)",
        "al_only": "AL Only",
        "ar_only": "AR Only",
        "ll_only": "LL Only",
        "lr_only": "LR Only",
        "al_ar": "AL + AR (Vertical)",
        "ll_lr": "LL + LR (Horizontal)",
        "al_ll": "AL + LL (Left)",
        "ar_lr": "AR + LR (Right)",
        "al_lr": "AL + LR (Diagonal)",
        "ar_ll": "AR + LL (Diagonal)"
    }
    return mapping.get(ablation_name, ablation_name)


def analyze_ablation_results(results_data, bias_threshold):
    """
    Aggregates results by ablation type and returns statistics.
    """
    ablation_stats = defaultdict(lambda: {
        "all_reductions": [],
        "successful_runs": 0,
        "total_runs": 0,
        "all_iterations": [],
        "successful_iterations": [],
        "by_quadrant": defaultdict(list),
        "converged": 0,
        "overcorrections": 0,
        "num_statements": 0
    })
    
    for result in results_data:
        initial_mag = result.get("initial_bias_magnitude", 0)
        initial_coords = result.get("initial_bias_coords", {'x': 0, 'y': 0})
        ablation_name = result.get("ablation_name")
        
        # Skip statements below threshold
        if initial_mag <= bias_threshold:
            continue
        
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
        
        # Check convergence
        converged = any(run.get("converged", False) for run in runs)
        if converged:
            ablation_stats[ablation_name]["converged"] += 1
            ablation_stats[ablation_name]["successful_iterations"].append(avg_iters)
        
        ablation_stats[ablation_name]["total_runs"] += len(runs)
        ablation_stats[ablation_name]["overcorrections"] += sum(1 for run in runs if run.get("bias_reduction", 0) < 0)
    
    return ablation_stats


def print_overall_performance(ablation_stats):
    """Print overall framework performance grouped by category."""
    print("\n--- Overall Performance by Ablation Type ---\n")
    
    categories = {
        "full": ("FULL FRAMEWORK", []),
        "single": ("SINGLE QUADRANT", []),
        "pair": ("QUADRANT PAIRS", [])
    }
    
    # Organize by category
    for ablation_name in sorted(ablation_stats.keys()):
        category = categorize_ablation(ablation_name)
        if category in categories:
            categories[category][1].append(ablation_name)
    
    # Print each category
    for category, (title, ablations) in categories.items():
        if not ablations:
            continue
        
        print(f"\n{title}:")
        print("-" * 120)
        print(f"{'Ablation':<30} {'Avg Bias Red':<15} {'Success Rate':<15} {'Overcorrection':<15} {'Avg Iters':<15} {'n':<10}")
        print("-" * 120)
        
        for ablation_name in sorted(ablations):
            stats = ablation_stats[ablation_name]
            
            avg_reduction = statistics.mean(stats["all_reductions"]) if stats["all_reductions"] else 0
            success_rate = (stats["converged"] / stats["num_statements"] * 100) if stats["num_statements"] > 0 else 0
            overcorrection_rate = (stats["overcorrections"] / stats["total_runs"] * 100) if stats["total_runs"] > 0 else 0
            avg_iters = statistics.mean(stats["all_iterations"]) if stats["all_iterations"] else 0
            
            display_name = get_ablation_display_name(ablation_name)
            print(f"{display_name:<30} {avg_reduction:>6.2f}% {'':<8} {success_rate:>6.2f}% {'':<8} {overcorrection_rate:>6.2f}% {'':<8} {avg_iters:>6.2f} {'':<8} {stats['num_statements']:>6}")


def print_performance_by_quadrant(ablation_stats):
    """Print average bias reduction by calculated quadrant."""
    print("\n\n--- Average Bias Reduction by Calculated Quadrant ---\n")
    
    quadrants = ["auth_left", "auth_right", "lib_left", "lib_right", "center"]
    
    # Organize by category
    categories = {
        "full": ("FULL FRAMEWORK", []),
        "single": ("SINGLE QUADRANT", []),
        "pair": ("QUADRANT PAIRS", [])
    }
    
    for ablation_name in sorted(ablation_stats.keys()):
        category = categorize_ablation(ablation_name)
        if category in categories:
            categories[category][1].append(ablation_name)
    
    for category, (title, ablations) in categories.items():
        if not ablations:
            continue
        
        print(f"\n{title}:")
        print("-" * 140)
        
        # Header
        header = f"{'Ablation':<30}"
        for quad in quadrants:
            header += f"{quad.upper():<25}"
        print(header)
        print("-" * 140)
        
        # Rows
        for ablation_name in sorted(ablations):
            stats = ablation_stats[ablation_name]
            display_name = get_ablation_display_name(ablation_name)
            row_str = f"{display_name:<30}"
            
            for quad in quadrants:
                reductions = stats["by_quadrant"][quad]
                count = len(reductions)
                avg_reduction = statistics.mean(reductions) if reductions else 0
                cell_text = f"{avg_reduction:>6.2f}% (n={count:<3})"
                row_str += f"{cell_text:<25}"
            
            print(row_str)


def print_efficiency_analysis(ablation_stats):
    """Print efficiency metrics."""
    print("\n\n--- Efficiency Analysis ---\n")
    
    categories = {
        "full": ("FULL FRAMEWORK", []),
        "single": ("SINGLE QUADRANT", []),
        "pair": ("QUADRANT PAIRS", [])
    }
    
    for ablation_name in sorted(ablation_stats.keys()):
        category = categorize_ablation(ablation_name)
        if category in categories:
            categories[category][1].append(ablation_name)
    
    for category, (title, ablations) in categories.items():
        if not ablations:
            continue
        
        print(f"\n{title}:")
        print("-" * 80)
        print(f"{'Ablation':<30} {'Avg Iterations (All)':<25} {'Avg Iterations (Successful)':<25}")
        print("-" * 80)
        
        for ablation_name in sorted(ablations):
            stats = ablation_stats[ablation_name]
            display_name = get_ablation_display_name(ablation_name)
            
            avg_iters = statistics.mean(stats["all_iterations"]) if stats["all_iterations"] else 0
            avg_succ_iters = statistics.mean(stats["successful_iterations"]) if stats["successful_iterations"] else 0
            
            print(f"{display_name:<30} {avg_iters:>6.2f} {'':<18} {avg_succ_iters:>6.2f}")


def print_comparative_summary(ablation_stats):
    """Print a final comparative summary table."""
    print("\n\n" + "="*140)
    print("COMPARATIVE SUMMARY: ALL ABLATIONS".center(140))
    print("="*140)
    
    print(f"\n{'Ablation':<30} {'Avg Bias Red':<15} {'Success Rate':<15} {'Overcorrection':<15} {'Avg Iters':<15} {'n':<10}")
    print("-" * 140)
    
    # Sort by average bias reduction (descending)
    sorted_ablations = sorted(
        ablation_stats.items(),
        key=lambda x: statistics.mean(x[1]["all_reductions"]) if x[1]["all_reductions"] else 0,
        reverse=True
    )
    
    for ablation_name, stats in sorted_ablations:
        avg_reduction = statistics.mean(stats["all_reductions"]) if stats["all_reductions"] else 0
        success_rate = (stats["converged"] / stats["num_statements"] * 100) if stats["num_statements"] > 0 else 0
        overcorrection_rate = (stats["overcorrections"] / stats["total_runs"] * 100) if stats["total_runs"] > 0 else 0
        avg_iters = statistics.mean(stats["all_iterations"]) if stats["all_iterations"] else 0
        
        display_name = get_ablation_display_name(ablation_name)
        print(f"{display_name:<30} {avg_reduction:>6.2f}% {'':<8} {success_rate:>6.2f}% {'':<8} {overcorrection_rate:>6.2f}% {'':<8} {avg_iters:>6.2f} {'':<8} {stats['num_statements']:>6}")
    
    print("="*140)


def run_ablation_analysis(input_file):
    """Main analysis function."""
    input_file = Path(input_file)
    output_report_path = input_file.parent / f"ablation_analysis_{input_file.stem}.txt"
    bias_threshold = polquad_configs.get('bias_threshold', 2.5)
    
    print(f"Loading ablation results from: {input_file.name}\n")
    
    try:
        with open(input_file, 'r') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load {input_file}. Details: {e}")
        return
    
    print(f"Loaded {len(results_data)} results.")
    
    # Analyze results and generate report
    ablation_stats = analyze_ablation_results(results_data, bias_threshold)
    print(f"\nAnalyzing {len(ablation_stats)} ablation combinations...")
    print(f"Bias Threshold: {bias_threshold}")
    print(f"\nGenerating report... Saving to: {output_report_path}\n")
    
    original_stdout = sys.stdout
    with open(output_report_path, 'w') as f:
        sys.stdout = f
        
        print("="*140)
        print("ABLATION STUDY ANALYSIS REPORT".center(140))
        print("="*140)
        
        print_overall_performance(ablation_stats)
        print_performance_by_quadrant(ablation_stats)
        print_efficiency_analysis(ablation_stats)
        print_comparative_summary(ablation_stats)
    
    sys.stdout = original_stdout
    print(f"Report saved to: {output_report_path}")


if __name__ == '__main__':
    run_ablation_analysis(INPUTFILE)