import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

# Add project root to path to allow config import
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
from polquad.config import polquad_configs

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

def analyze_results():
    """
    Loads the consolidated framework output and saves a detailed
    statistical summary to a text file.
    """
    data_file_path = project_root / "outputs/runs/framework_output.json"
    output_report_path = project_root / "outputs/reports/analysis_report.txt"
    bias_threshold = polquad_configs.get('bias_threshold', 2.5)
    
    # Define bins for analyzing performance
    bias_bins = [
        (1.1, 2.5, "Low (1.1-2.5)"),
        (2.5, 4.0, "Mid-Low (2.5-4.0)"),
        (4.0, 5.5, "Medium (4.0-5.5)"),
        (5.5, 7.0, "Mid-High (5.5-7.0)"),
        (7.0, 8.5, "High (7.0-8.5)"),
        (8.5, 10.1, "Extreme (8.5-10.0)")
    ]

    print(f"Loading data from: {data_file_path}\n")
    try:
        with open(data_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find the file at {data_file_path}.")
        return

    # Aggregate the Data
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
    
    hardest_statements = []

    for item in data:
        initial_mag = item.get("initial_bias_magnitude", 0)
        initial_coords = item.get("initial_bias_coords", {'x': 0, 'y': 0})

        if initial_mag < bias_threshold:
            continue
            
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
            
            if name == "full_polquad":
                hardest_statements.append((median_reduction, item["original_statement"]))

    # Generate Summary Report
    original_stdout = sys.stdout
    with open(output_report_path, 'w') as f:
        sys.stdout = f  

        print("="*80)
        print("POLQUAD FRAMEWORK PERFORMANCE ANALYSIS".center(80))
        print("="*80)

        # Overall Performance
        print("\n--- Overall Framework Performance ---\n")
        total_statements = len([item for item in data if item.get("initial_bias_magnitude", 0) >= bias_threshold])
        for name, stats in framework_stats.items():
            avg_reduction = statistics.mean(stats["all_reductions"]) if stats["all_reductions"] else 0
            success_rate = (stats["successful_runs"] / total_statements) * 100 if total_statements > 0 else 0
            overcorrection_rate = (stats["overcorrections"] / stats["total_runs"]) * 100 if stats["total_runs"] > 0 else 0
            
            print(f"  Framework: {name.upper().replace('_', ' ')}")
            print(f"    - Average Bias Reduction: {avg_reduction:.2f}%")
            print(f"    - Success Rate (met threshold): {success_rate:.2f}%")
            print(f"    - Overcorrection Rate (negative reduction): {overcorrection_rate:.2f}%")

        # Performance by Quadrant
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

        # Performance by Initial Bias Magnitude
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

        # Efficiency Analysis
        print("\n--- Efficiency Analysis ---\n")
        for name, stats in framework_stats.items():
            avg_iters = statistics.mean(stats["all_iterations"]) if stats["all_iterations"] else 0
            avg_succ_iters = statistics.mean(stats["successful_iterations"]) if stats["successful_iterations"] else 0
            print(f"  Framework: {name.upper().replace('_', ' ')}")
            print(f"    - Average Iterations (All Runs): {avg_iters:.2f}")
            print(f"    - Average Iterations (Successful Runs): {avg_succ_iters:.2f}")

        # Hardest Statements
        print("\n--- Top 3 Most Difficult Statements for FULL POLQUAD ---\n")
        hardest_statements.sort(key=lambda x: x[0])
        for i, (reduction, statement) in enumerate(hardest_statements[:3]):
            print(f"  {i+1}. Reduction: {reduction:.2f}%")
            print(f"     Statement: \"{statement}\"\n")
            
        print("="*80)

    sys.stdout = original_stdout # Reset the standard output to its original value
    print(f"\nAnalysis complete. Report saved to: {output_report_path}")


if __name__ == '__main__':
    analyze_results()

