import json
import pandas as pd
import numpy as np
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent

def analyze_bias_correlation(file_path, framework_name):
    """
    Loads data, calculates correlation between initial magnitude and bias reduction,
    and compares performance on High vs. Low bias statements.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 1. Prepare data for DataFrame
    statements = []
    for item in data:
        if 'error' in item:
            continue
        
        # Ensure we can safely access nested data
        try:
            initial_mag = item['initial_bias_magnitude']
            bias_reduction = item['frameworks']['full_polquad']['bias_reduction']
            final_mag = item['frameworks']['full_polquad']['final_bias_magnitude']
            
            statements.append({
                'initial_mag': initial_mag,
                'bias_reduction': bias_reduction,
                'final_mag': final_mag
            })
        except KeyError:
            # Skip items with missing expected keys (e.g., failed to run)
            continue

    if not statements:
        return {
            'framework': framework_name,
            'correlation': np.nan,
            'high_bias_avg_reduction': np.nan,
            'low_bias_avg_reduction': np.nan,
            'high_bias_count': 0,
            'low_bias_count': 0
        }

    df = pd.DataFrame(statements)

    # 2. Calculate Correlation
    # We correlate initial_mag with bias_reduction.
    # A POSITIVE correlation means HIGHER initial bias leads to HIGHER reduction. (Good)
    # A NEGATIVE correlation means HIGHER initial bias leads to LOWER reduction. (Bad)
    correlation = df['initial_mag'].corr(df['bias_reduction'])

    # 3. High vs. Low Bias Comparison
    # Define the threshold as the median initial magnitude
    median_mag = df['initial_mag'].median()
    
    # Exclude statements with 0 initial bias (index 21 in your data) which skew the reduction %
    df_clean = df[df['initial_mag'] > 0] 

    high_bias = df_clean[df_clean['initial_mag'] >= median_mag]
    low_bias = df_clean[df_clean['initial_mag'] < median_mag]
    
    # Calculate average bias reduction for each group
    high_avg_reduction = high_bias['bias_reduction'].mean()
    low_avg_reduction = low_bias['bias_reduction'].mean()

    return {
        'framework': framework_name,
        'correlation': correlation,
        'high_bias_avg_reduction': high_avg_reduction,
        'low_bias_avg_reduction': low_avg_reduction,
        'high_bias_count': len(high_bias),
        'low_bias_count': len(low_bias)
    }

# --- Execution ---

# Define file paths
unified_json_path = project_root / "results" / "unified_polquad_results.json"
full_json_path = project_root / "results" / "full_polquad_results.json"
output_path = project_root / "results" / "framework_comparison_output.txt"

# Run analysis for both frameworks
full_analysis = analyze_bias_correlation(full_json_path, 'FULL_POLQUAD')
unified_analysis = analyze_bias_correlation(unified_json_path, 'UNIFIED_POLQUAD')

# --- Output Results ---
print("=========================================================")
print("      INITIAL BIAS MAGNITUDE VS. BIAS REDUCTION ANALYSIS")
print("=========================================================")

def print_results(analysis):
    print(f"\n--- Framework: {analysis['framework']} ---")
    
    # Correlation Interpretation
    corr = analysis['correlation']
    if np.isnan(corr):
        print("Not enough data to calculate metrics.")
        return

    trend = "a POSITIVE correlation" if corr > 0 else "a NEGATIVE correlation"
    performance = "better" if corr > 0 else "worse"
    
    print(f"Correlation (Initial Mag vs. Bias Reduction): {corr:.3f}")
    print(f"This indicates {trend}. Higher initial bias leads to {performance} bias reduction.")
    
    print("\nHigh vs. Low Bias Performance (based on median initial magnitude):")
    
    high_red = analysis['high_bias_avg_reduction']
    low_red = analysis['low_bias_avg_reduction']
    diff = high_red - low_red
    
    # High vs. Low Bias Reduction Comparison
    if diff > 0:
        conclusion = f"Higher bias statements perform BETTER by {diff:.1f} percentage points."
    elif diff < 0:
        conclusion = f"Lower bias statements perform BETTER by {abs(diff):.1f} percentage points."
    else:
        conclusion = "High and Low bias statements perform equally."

    print(f"  - High Bias Statements (N={analysis['high_bias_count']}): Avg Reduction {high_red:.1f}%")
    print(f"  - Low Bias Statements (N={analysis['low_bias_count']}): Avg Reduction {low_red:.1f}%")
    print(f"  Conclusion: {conclusion}")

print_results(full_analysis)
print_results(unified_analysis)