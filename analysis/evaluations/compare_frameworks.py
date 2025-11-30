import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

unified_json_path = project_root / "outputs" / "runs" / "unified_polquad_results.json"
full_json_path = project_root / "outputs" / "runs" / "full_polquad_results.json"
output_path = project_root / "outputs" / "reports" / "framework_comparison_output.txt"

def calculate_metrics(data, framework_name):
    """Calculates performance metrics per quadrant for a single framework's data."""
    results = defaultdict(lambda: {
        'total_final_bias': 0.0,
        'total_bias_reduction': 0.0,
        'total_iters': 0,
        'converged_count': 0,
        'total_count': 0,
        'initial_bias_sum': 0.0,
    })

    for statement in data:
        # Check if the statement ran successfully
        if 'error' in statement:
            continue
        
        # Determine the quadrant (grouping key)
        true_label = statement.get('true_label', 'UNKNOWN')
        
        # Extract metrics, assuming the nested structure is consistent
        framework_data = statement['frameworks']['full_polquad']
        
        final_mag = framework_data.get('final_bias_magnitude', 0.0)
        bias_reduction = framework_data.get('bias_reduction', 0.0)
        num_iters = framework_data.get('num_iterations', 0)
        converged = framework_data.get('converged', False)
        initial_mag = statement.get('initial_bias_magnitude', 0.0)

        # Update Overall metrics
        for key in [true_label, 'OVERALL']:
            results[key]['total_final_bias'] += final_mag
            results[key]['total_bias_reduction'] += bias_reduction
            results[key]['total_iters'] += num_iters
            results[key]['total_count'] += 1
            results[key]['initial_bias_sum'] += initial_mag
            if converged:
                results[key]['converged_count'] += 1

    # Finalize calculations (averages, rates)
    final_metrics = {}
    for quadrant, metrics in results.items():
        if metrics['total_count'] > 0:
            final_metrics[quadrant] = {
                'Framework': framework_name,
                'Avg Final Bias': metrics['total_final_bias'] / metrics['total_count'],
                'Avg Bias Reduction': metrics['total_bias_reduction'] / metrics['total_count'],
                'Avg Iters': metrics['total_iters'] / metrics['total_count'],
                'Convergence Rate': (metrics['converged_count'] / metrics['total_count']) * 100,
                'Count': metrics['total_count']
            }
    return final_metrics

def generate_summary_table(full_metrics, unified_metrics, output_filename=output_path):
    """Formats the calculated metrics into a comparative table and writes to a file."""
    quadrant_map = {
        'auth_left': 'AUTHORITARIAN LEFT',
        'auth_right': 'AUTHORITARIAN RIGHT',
        'lib_left': 'LIBERTARIAN LEFT',
        'lib_right': 'LIBERTARIAN RIGHT',
        'OVERALL': 'OVERALL PERFORMANCE'
    }
    
    # Define the desired order of quadrants
    quadrant_order = ['auth_left', 'auth_right', 'lib_left', 'lib_right', 'OVERALL']
    
    markdown_output = []
    
    markdown_output.append("========================================================\n")
    markdown_output.append("FRAMEWORK PERFORMANCE SUMMARY\n")
    markdown_output.append("========================================================\n")

    for key in quadrant_order:
        quadrant_name = quadrant_map.get(key)
        
        if key not in full_metrics or key not in unified_metrics:
            continue

        markdown_output.append(f"\n## QUADRANT: {quadrant_name}\n")
        
        data = [
            full_metrics[key],
            unified_metrics[key]
        ]
        
        # Create and sort DataFrame
        df = pd.DataFrame(data).sort_values(
            ['Convergence Rate', 'Avg Final Bias'], 
            ascending=[False, True]
        ).reset_index(drop=True)
        
        # Format columns
        df['Rank'] = df.index + 1
        df['Avg Final Bias'] = df['Avg Final Bias'].round(2)
        df['Avg Bias Reduction'] = df['Avg Bias Reduction'].round(1).astype(str) + '%'
        df['Avg Iters'] = df['Avg Iters'].round(1)
        df['Convergence Rate'] = df['Convergence Rate'].round(1).astype(str) + '%'
        
        # Select and rename columns for final display
        df = df[['Rank', 'Framework', 'Avg Final Bias', 'Avg Bias Reduction', 'Avg Iters', 'Convergence Rate', 'Count']]
        df.rename(columns={'Avg Final Bias': 'Avg Final Bias (Mag)',
                           'Avg Bias Reduction': 'Avg Bias Reduction (%)',
                           'Avg Iters': 'Avg Iters',
                           'Convergence Rate': 'Conv. Rate (%)'}, inplace=True)
        
        # Convert DataFrame to Markdown table and append
        markdown_output.append(df.to_markdown(index=False))
        markdown_output.append("\n***\n")

    # Write all output to the specified text file
    with open(output_filename, 'w') as f:
        f.write("\n".join(markdown_output))
    
    print(f"✅ Summary successfully written to {output_filename}")


def main():
    # Load data from the provided JSON files
    try:
        with open(full_json_path, 'r') as f:
            full_data = json.load(f)
        with open(unified_json_path, 'r') as f:
            unified_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: One of the input files was not found: {e}. Please ensure 'full_polquad_results.json' and 'unified_polquad_results.json' are in the correct directory.")
        return

    # Calculate metrics for each framework
    full_metrics = calculate_metrics(full_data, 'FULL_POLQUAD')
    unified_metrics = calculate_metrics(unified_data, 'UNIFIED_POLQUAD')

    # Generate the formatted comparison table and save to file
    generate_summary_table(full_metrics, unified_metrics)

if __name__ == "__main__":
    main()