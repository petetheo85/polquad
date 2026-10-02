import json
import sys
from pathlib import Path
from collections import defaultdict
import statistics

# --- Environment Setup and Paths ---
# Assume project_root is the directory containing the 'results' and 'analysis' folders.
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

# Define locations based on your clarification
OUTPUT_REPORT_PATH = project_root / "outputs/reports/no_iteration_analysis_report.txt"
DATA_INPUT_DIR = project_root / "outputs/runs"

# --- Configuration (Based on paper standards) ---
class Configs:
    # Threshold for a statement to be processed (Bias Magnitude > 2.5)
    polquad_configs = {'bias_threshold': 2.5} 
polquad_configs = Configs.polquad_configs

# --- Utility Functions ---

def get_quadrant_from_coords(coords):
    """Determines the political quadrant from (x, y) coordinates."""
    x = coords.get('x', 0)
    y = coords.get('y', 0)
    
    if x > 0 and y > 0: return "LIB_RIGHT"
    elif x < 0 and y > 0: return "LIB_LEFT"
    elif x < 0 and y < 0: return "AUTH_LEFT"
    elif x > 0 and y < 0: return "AUTH_RIGHT"
    else: return "CENTER"

def get_bias_bin(magnitude, bias_bins):
    """Maps bias magnitude to a predefined bin label."""
    for low, high, label in bias_bins:
        if low <= magnitude < high:
            return label
    return None

def generate_report_for_no_iteration(data_items, group_title, bias_threshold, bias_bins):
    """
    Analyzes a list of aggregated data items by iterating over ALL runs, 
    as each represents an independent non-iterative attempt.
    """
    print("="*80)
    print(f"ANALYSIS FOR GROUP: {group_title.upper()} (NON-ITERATIVE, FIRST ATTEMPT)".center(80))
    print("="*80)

    framework_stats = {
        name: {
            "all_reductions": [],
            "successful_runs": 0,
            "overcorrections": 0,
            "total_attempts": 0,
            "by_quadrant": defaultdict(list),
            "by_initial_bias": defaultdict(list)
        }
        for name in ["full_polquad", "unified_polquad", "naive"]
    }
    
    total_statements_read = 0

    for item in data_items:
        initial_mag = item.get("initial_bias_magnitude", 0)
        initial_coords = item.get("initial_bias_coords", {'x': 0, 'y': 0})

        if initial_mag < bias_threshold:
            continue
        
        total_statements_read += 1
            
        # Determine calculated quadrant and bias bin
        calculated_quadrant = get_quadrant_from_coords(initial_coords)
        current_bias_bin_label = get_bias_bin(initial_mag, bias_bins)

        for name, stats in framework_stats.items():
            comp = item.get("framework_comparison", {}).get(name)
            if not comp or not comp.get("runs"):
                continue

            # --- CRITICAL: LOOP OVER ALL RUNS (3 independent attempts) ---
            for run in comp["runs"]:
                reduction = run.get("bias_reduction", 0)
                
                # Record metrics for this single attempt
                stats["all_reductions"].append(reduction)
                stats["by_quadrant"][calculated_quadrant].append(reduction)
                
                if current_bias_bin_label:
                    stats["by_initial_bias"][current_bias_bin_label].append(reduction)

                if run.get("converged", False):
                    stats["successful_runs"] += 1
                
                stats["total_attempts"] += 1 
                if reduction < 0:
                     stats["overcorrections"] += 1


    # --- Overall Performance ---
    print("\n--- Overall Framework Performance (First Attempt Only) ---\n")
    
    total_attempts_ref = next(iter(framework_stats.values()))["total_attempts"] 

    print(f"Total Aggregated Attempts Processed (N={total_attempts_ref})\n")
    
    if total_attempts_ref == 0:
        print("  No data found for this group meeting the bias threshold.\n")
        return

    for name, stats in framework_stats.items():
        avg_reduction = statistics.mean(stats["all_reductions"]) if stats["all_reductions"] else 0
        success_rate = (stats["successful_runs"] / total_attempts_ref) * 100 if total_attempts_ref > 0 else 0
        overcorrection_rate = (stats["overcorrections"] / total_attempts_ref) * 100 if total_attempts_ref > 0 else 0
        
        print(f"  Framework: {name.upper().replace('_', ' ')}")
        print(f"    - Average Bias Reduction: {avg_reduction:.2f}%")
        print(f"    - Success Rate (met threshold): {success_rate:.2f}%")
        print(f"    - Overcorrection Rate (negative reduction): {overcorrection_rate:.2f}%")
    
    # --- Performance by Quadrant ---
    print("\n--- Average Bias Reduction by Calculated Quadrant ---\n")
    quadrants = ["AUTH_LEFT", "AUTH_RIGHT", "LIB_LEFT", "LIB_RIGHT", "CENTER"]
    header = f"{'Framework':<20}" + "".join([f"{q:<15}" for q in quadrants])
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

    print("\n" + "="*80 + "\n\n")

def run_no_iteration_analysis(file_names, output_path):
    """
    Loads and aggregates data from the specified no_iteration files.
    """
    bias_bins = [
        (1.1, 2.5, "Low (1.1-2.5)"), (2.5, 4.0, "Mid-Low (2.5-4.0)"),
        (4.0, 5.5, "Medium (4.0-5.5)"), (5.5, 7.0, "Mid-High (5.5-7.0)"),
        (7.0, 8.5, "High (7.0-8.5)"), (8.5, 10.1, "Extreme (8.5-10.0)")
    ]
    
    all_data = []
    
    # --- FILE LOADING AND AGGREGATION ---
    print(f"Scanning directory: {DATA_INPUT_DIR}")
    
    # Find all specified JSON files in the input directory
    for name in file_names:
        file_path = DATA_INPUT_DIR / name
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
            print(f"  Successfully loaded and aggregated data from: {name}")
        except FileNotFoundError:
            print(f"  ERROR: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"  An error occurred loading {name}: {e}. Skipping.")

    # --- Generate Report ---
    
    # Ensure the analysis directory exists before writing
    output_path.parent.mkdir(parents=True, exist_ok=True) 

    original_stdout = sys.stdout
    
    try:
        with open(output_path, 'w') as f:
            sys.stdout = f  # Redirect stdout to the report file
            print(f"--- Report for Non-Iterative Analysis (Gemini High Bias, Aggregated) ---\n")
            generate_report_for_no_iteration(all_data, "HIGH_BIAS_GEMINI_FIRST_ATTEMPT", polquad_configs.get('bias_threshold', 2.5), bias_bins)
    finally:
        sys.stdout = original_stdout # Reset stdout
    
    print(f"\nAnalysis complete. Report saved to: {output_path}")


if __name__ == '__main__':
    no_iteration_files = [
        "framework_output_seed42_no_iterations.json",
        "framework_output_seed1337_no_iterations.json",
        "framework_output_seed1984_no_iterations.json"
    ]
    
    # This will run the analysis and save the report to project_root/analysis/
    run_no_iteration_analysis(no_iteration_files, OUTPUT_REPORT_PATH)