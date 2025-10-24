import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics
import re # Import regex for flexible filename parsing

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))
from config import polquad_configs

# --- Keep your get_quadrant_from_coords function ---
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

def parse_filename(filename_stem):
    """
    Parses a filename stem to extract experimental conditions.
    Example: 'framework_output_seed42_100_high_bias_gemini'
    Returns: (dataset_type, llm)
    """
    # Default values
    llm = 'gemini' # You mentioned most original runs were gemini
    dataset_type = 'original' # Assume 'original' unless 'high_bias' is found
    
    if 'high_bias' in filename_stem:
        dataset_type = 'high_bias'
    
    if 'openai' in filename_stem:
        llm = 'openai'
    elif 'claude' in filename_stem:
        llm = 'claude'
    elif 'gemini' in filename_stem:
        llm = 'gemini'
        
    return f"{dataset_type}_{llm}"

def generate_report_for_group(data_items, group_title, bias_threshold, bias_bins):
    """
    Analyzes a list of data items for a specific group and prints the report.
    This function will contain the core logic from your old 'analyze_results'.
    """
    print("="*80)
    print(f"ANALYSIS FOR GROUP: {group_title.upper()}".center(80))
    print("="*80)

    # Aggregate the Data (for this group)
    framework_stats = {
        name: {
            "all_reductions": [],
            "successful_runs": 0,
            "overcorrections": 0,
            "total_runs": 0,
            "all_iterations": [],
            "successful_iterations": [],
            "by_quadrant": defaultdict(list),
            "by_initial_bias": defaultdict(list)
        }
        for name in ["full_polquad", "unified_polquad", "naive"]
    }
    
    total_statements = 0

    for item in data_items:
        initial_mag = item.get("initial_bias_magnitude", 0)
        initial_coords = item.get("initial_bias_coords", {'x': 0, 'y': 0})

        if initial_mag < bias_threshold:
            continue
        
        total_statements += 1
            
        # Determine calculated quadrant and bias bin
        calculated_quadrant = get_quadrant_from_coords(initial_coords)
        current_bias_bin_label = None
        for low, high, label in bias_bins:
            if low <= initial_mag < high:
                current_bias_bin_label = label
                break

        for name, stats in framework_stats.items():
            comp = item.get("framework_comparison", {}).get(name)
            if not comp:
                continue

            # Use median if available, fallback to avg
            median_reduction = comp["averages"].get("median_bias_reduction", comp["averages"].get("avg_bias_reduction"))
            
            stats["all_reductions"].append(median_reduction)
            stats["by_quadrant"][calculated_quadrant].append(median_reduction)
            
            if current_bias_bin_label:
                stats["by_initial_bias"][current_bias_bin_label].append(median_reduction)

            avg_iters = comp["averages"].get("avg_iterations", 0)
            stats["all_iterations"].append(avg_iters)

            was_successful = any(run.get("converged", False) for run in comp["runs"])
            if was_successful:
                stats["successful_runs"] += 1
                stats["successful_iterations"].append(avg_iters)
            
            stats["total_runs"] += len(comp["runs"])
            stats["overcorrections"] += sum(1 for run in comp["runs"] if run.get("bias_reduction", 0) < 0)

    # --- Overall Performance ---
    print("\n--- Overall Framework Performance ---\n")
    print(f"Total Unique Statements Processed (n={total_statements})\n")
    if total_statements == 0:
        print("  No data found for this group meeting the bias threshold.\n")
        print("="*80 + "\n\n")
        return # Skip rest of report if no data

    for name, stats in framework_stats.items():
        avg_reduction = statistics.mean(stats["all_reductions"]) if stats["all_reductions"] else 0
        success_rate = (stats["successful_runs"] / total_statements) * 100 if total_statements > 0 else 0
        overcorrection_rate = (stats["overcorrections"] / stats["total_runs"]) * 100 if stats["total_runs"] > 0 else 0
        
        print(f"  Framework: {name.upper().replace('_', ' ')}")
        print(f"    - Average Bias Reduction: {avg_reduction:.2f}%")
        print(f"    - Success Rate (met threshold): {success_rate:.2f}%")
        print(f"    - Overcorrection Rate (negative reduction): {overcorrection_rate:.2f}%")

    # --- Performance by Quadrant ---
    print("\n--- Average Bias Reduction by Calculated Quadrant ---\n")
    quadrants = ["auth_left", "auth_right", "lib_left", "lib_right", "center"]
    header = f"{'Framework':<20}" + "".join([f"{q.upper():<15}" for q in quadrants])
    print(header)
    print("-" * len(header))
    for name, stats in framework_stats.items():
        row_str = f"{name.upper().replace('_', ' '):<20}"
        for quad in quadrants:
            reductions = stats["by_quadrant"][quad]
            count = len(reductions)
            avg_reduction = statistics.mean(reductions) if reductions else 0
            cell_text = f"{avg_reduction:.1f}% (n={count})"
            row_str += f"{cell_text:<15}"
        print(row_str)

    # --- Performance by Initial Bias Magnitude ---
    print("\n--- Performance by Initial Bias Magnitude ---\n")
    bin_labels = [label for _, _, label in bias_bins]
    header = f"{'Framework':<20}" + "".join([f"{label:<25}" for label in bin_labels])
    print(header)
    print("-" * len(header))
    for name, stats in framework_stats.items():
        row_str = f"{name.upper().replace('_', ' '):<20}"
        for label in bin_labels:
            reductions_in_bin = stats["by_initial_bias"][label]
            count = len(reductions_in_bin)
            avg_reduction = statistics.mean(reductions_in_bin) if reductions_in_bin else 0
            cell_text = f"{avg_reduction:.2f}% (n={count})"
            row_str += f"{cell_text:<25}"
        print(row_str)

    # --- Efficiency Analysis ---
    print("\n--- Efficiency Analysis ---\n")
    for name, stats in framework_stats.items():
        avg_iters = statistics.mean(stats["all_iterations"]) if stats["all_iterations"] else 0
        avg_succ_iters = statistics.mean(stats["successful_iterations"]) if stats["successful_iterations"] else 0
        print(f"  Framework: {name.upper().replace('_', ' ')}")
        print(f"    - Average Iterations (All Runs): {avg_iters:.2f}")
        print(f"    - Average Iterations (Successful Runs): {avg_succ_iters:.2f}")

    print("\n" + "="*80 + "\n\n")


def run_aggregate_analysis():
    """
    Loads ALL framework outputs from '/results/outputs/old',
    groups them by experimental condition, and saves one
    consolidated report.
    """
    data_dir = project_root / "results/outputs/old"
    output_report_path = project_root / "results/analysis/analysis_report_aggregate.txt"
    bias_threshold = polquad_configs.get('bias_threshold', 2.5)
    
    # Define your standard 6 bins here
    bias_bins = [
        (1.1, 2.5, "Low (1.1-2.5)"),
        (2.5, 4.0, "Mid-Low (2.5-4.0)"),
        (4.0, 5.5, "Medium (4.0-5.5)"),
        (5.5, 7.0, "Mid-High (5.5-7.0)"),
        (7.0, 8.5, "High (7.0-8.5)"),
        (8.5, 10.1, "Extreme (8.5-10.0)")
    ]

    # 1. Discover files and group data
    data_groups = defaultdict(list)
    print(f"Scanning for JSON files in: {data_dir}\n")
    
    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        print("ERROR: No .json files found in that directory.")
        return

    for file_path in json_files:
        print(f"  Loading: {file_path.name}")
        group_key = parse_filename(file_path.stem)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                data_groups[group_key].extend(data)
        except Exception as e:
            print(f"    WARNING: Could not load or parse {file_path.name}. Error: {e}")

    print(f"\nFound {len(data_groups)} experimental groups: {list(data_groups.keys())}")

    # 2. Generate consolidated report
    print(f"\nGenerating aggregate report... Saving to: {output_report_path}")
    original_stdout = sys.stdout
    with open(output_report_path, 'w') as f:
        sys.stdout = f  # Redirect stdout to the report file
        
        # Sort groups for a consistent report order
        sorted_groups = sorted(data_groups.items(), key=lambda item: item[0])
        
        for group_name, items_list in sorted_groups:
            generate_report_for_group(items_list, group_name, bias_threshold, bias_bins)

    sys.stdout = original_stdout # Reset stdout
    print("\nAggregate analysis complete.")


if __name__ == '__main__':
    run_aggregate_analysis()