import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use('seaborn-v0_8-paper')

def create_output_directory():
    """Ensures the 'graphs' directory exists."""
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    print("Graphs will be saved to the 'graphs/' directory.")

def plot_simple_bar(data, title, filename, ylabel, color_map=None):
    """
    Generates a simple bar chart, sorted descendingly.
    Accepts a color_map to assign specific colors to bars.
    """
    # Create a pandas Series from the data
    s = pd.Series(data)
    # Sort by value in descending order for better visualization
    s = s.sort_values(ascending=True)
    
    # Assign colors if a map is provided
    colors = None
    if color_map:
        # Create a list of colors based on the sorted index
        try:
            colors = [color_map[framework] for framework in s.index]
        except KeyError:
            print("Warning: Not all frameworks in data have a defined color. Using default colors.")
            colors = None
            
    ax = s.plot(kind='bar', figsize=(10, 6), width=0.7, color=colors)
    ax.set_axisbelow(True)
    
    # Set titles and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('Framework', fontsize=14)
    
    # Format X-axis labels
    ax.set_xticklabels(s.index, rotation=0, fontsize=11)
    
    # Add percentage labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_grouped_bar(data, title, filename, ylabel, color_map=None):
    """
    Generates a grouped bar chart.
    Accepts a color_map to assign specific colors to bars.
    """
    df = pd.DataFrame(data)
    
    # Re-order columns for a logical flow
    ordered_columns = ['Naive', 'Unified POLQUAD', 'Full POLQUAD']
    df = df[ordered_columns]
    
    # Get colors from map, if provided
    colors = None
    if color_map:
        try:
            colors = [color_map[col] for col in ordered_columns]
        except KeyError:
            print("Warning: Not all frameworks in data have a defined color. Using default colors.")
            colors = None

    # Pass colors to the plot
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.7, color=colors)
    ax.set_axisbelow(True)
    
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlabel('LLM Backend (Dataset)', fontsize=16)
    ax.set_xticklabels(df.index, rotation=0, fontsize=14)
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}\%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9)
    
    ax.legend(title='Framework', loc='lower right', framealpha=0.8, fontsize='large', title_fontsize='13')
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=-1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def main():
    """
    Main function to define data and call plotting functions.
    Data is manually extracted from 'analysis_report_aggregate_merged.txt'.
    """
    create_output_directory()

    # Define color map to ensure consistency between graphs
    # Based on the default 'C' color cycle used by matplotlib
    # C0=blue, C1=orange, C2=green
    # This matches the order in plot_grouped_bar: ['Iterative Naive', 'Unified POLQUAD', 'Full POLQUAD']
    framework_colors = {
        'Naive': 'lightsteelblue',
        'Unified POLQUAD': 'cornflowerblue',
        'Full POLQUAD': 'royalblue'
    }

    # --- Data for Graph 1: Original Gemini Bias Reduction ---
    # Source: ORIGINAL_GEMINI section, "Average Bias Reduction"
    original_gemini_reduction_data = {
        'Naive': 63.13,
        'Unified POLQUAD': 64.25,
        'Full POLQUAD': 67.34
        
    }
    plot_simple_bar(original_gemini_reduction_data,
                    'Overall Bias Reduction (Uncurated Dataset, n=747)',
                    'graphs/original_gemini_reduction.png',
                    'Average Bias Reduction (%)',
                    color_map=framework_colors) # Pass the color map here

    # --- Data for Graph 2: High-Bias Average Bias Reduction ---
    # Source: HIGH_BIAS_* sections, "Average Bias Reduction"
    high_bias_reduction_data = {
        'Naive': {
            'Claude (n=100)': 65.20,
            'OpenAI (n=98)': 32.67,
            'Gemini (n=300)': 68.28
        },
        'Unified POLQUAD': {
            'Claude (n=100)': 73.30,
            'OpenAI (n=98)': 41.93,
            'Gemini (n=300)': 71.59
        },
        'Full POLQUAD': {
            'Claude (n=100)': 76.45,
            'OpenAI (n=98)': 76.66,
            'Gemini (n=300)': 77.75
        }
    }
    plot_grouped_bar(high_bias_reduction_data,
                     'Average Bias Reduction on High-Bias Dataset',
                     'graphs/high_bias_reduction.png',
                     'Average Bias Reduction (%)',
                     color_map=framework_colors)
    
    # --- Data for Graph 3: High-Bias Success Rate ---
    # Source: HIGH_BIAS_* sections, "Success Rate (met threshold)"
    high_bias_success_rate_data = {
        'Naive': {
            'Claude (n=100)': 39.00,
            'OpenAI (n=98)': 30.61,
            'Gemini (n=300)': 62.00
        },
        'Unified POLQUAD': {
            'Claude (n=100)': 59.00,
            'OpenAI (n=98)': 43.88,
            'Gemini (n=300)': 69.00
        },
        'Full POLQUAD': {
            'Claude (n=100)': 60.00,
            'OpenAI (n=98)': 76.53,
            'Gemini (n=300)': 80.00
        }
    }
    plot_grouped_bar(high_bias_success_rate_data,
                     'Framework Success Rate on High-Bias Dataset',
                     'graphs/high_bias_success_rate.png',
                     'Success Rate (%)',
                     color_map=framework_colors)

if __name__ == '__main__':
    main()

print("New graphs with consistent colors generated successfully.")