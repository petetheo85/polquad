import pandas as pd
import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-paper')

def create_output_directory():
    """Ensures the 'graphs' directory exists."""
    if not os.path.exists('outputs/figures'):
        os.makedirs('outputs/figures', exist_ok=True)
    print("Graphs will be saved to the 'outputs/figures/' directory.")

def plot_success_rate(data, title, filename):
    """
    Generates a grouped bar chart for Success Rate.
    This chart is your "money chart" showing performance collapse.
    """
    df = pd.DataFrame(data)
    
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.7)
    
    # Set titles and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('LLM Backend (Dataset)', fontsize=12)
    
    # Format X-axis labels
    ax.set_xticklabels(df.index, rotation=0, fontsize=11)
    
    # Add percentage labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9)
    
    # Position legend
    ax.legend(title='Framework', bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make room for legend
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_efficiency(data, title, filename):
    """
    Generates a grouped bar chart for Efficiency (Avg. Successful Iterations).
    This chart shows WHY your framework is better (more efficient).
    """
    df = pd.DataFrame(data)
    
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.7)
    
    # Set titles and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Average Iterations per Success', fontsize=12)
    ax.set_xlabel('LLM Backend (Dataset)', fontsize=12)
    
    # Format X-axis labels
    ax.set_xticklabels(df.index, rotation=0, fontsize=11)
    
    # Add value labels on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=9)
    
    # Position legend
    ax.legend(title='Framework', bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Make room for legend
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

def plot_reduction_by_bias(data, title, filename):
    """
    Generates a line graph showing performance as problem difficulty increases.
    This uses the HIGH_BIAS_OPENAI data.
    """
    df = pd.DataFrame(data)
    # Shorten labels for the x-axis
    df.index = [
        'Low\n(1.1-2.5)\n(n=1)', 
        'Mid-Low\n(2.5-4.0)\n(n=3)', 
        'Medium\n(4.0-5.5)\n(n=35)', 
        'Mid-High\n(5.5-7.0)\n(n=4)', 
        'High\n(7.0-8.5)\n(n=32)', 
        'Extreme\n(8.5-10.0)\n(n=13)'
    ]

    ax = df.plot(kind='line', marker='o', figsize=(12, 7), lw=2.5)

    # Set titles and labels
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Average Bias Reduction (%)', fontsize=12)
    ax.set_xlabel('Initial Bias Magnitude (Bin)', fontsize=12)
    
    # Format X-axis labels
    plt.xticks(fontsize=10)

    # Add grid and legend
    ax.grid(linestyle='--', alpha=0.7)
    ax.legend(title='Framework', fontsize=11)
    
    # Adjust layout and save
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

    # --- Data for Graph 1: Success Rate ---
    # Source: HIGH_BIAS_* sections, "Success Rate (met threshold)"
    success_rate_data = {
        'Iterative Naive': {
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
    plot_success_rate(success_rate_data, 
                      'Framework Success Rate on High-Bias Dataset', 
                      'outputs/figures/high_bias_success_rate.png')

    # --- Data for Graph 2: Efficiency ---
    # Source: HIGH_BIAS_* sections, "Average Iterations (Successful Runs)"
    efficiency_data = {
        'Iterative Naive': {
            'Claude (n=100)': 2.38,
            'OpenAI (n=98)': 2.17,
            'Gemini (n=300)': 1.83
        },
        'Unified POLQUAD': {
            'Claude (n=100)': 2.10,
            'OpenAI (n=98)': 2.33,
            'Gemini (n=300)': 1.96
        },
        'Full POLQUAD': {
            'Claude (n=100)': 2.13,
            'OpenAI (n=98)': 1.75,
            'Gemini (n=300)': 1.84
        }
    }
    plot_efficiency(efficiency_data, 
                    'Framework Efficiency on High-Bias Dataset (Avg. Iterations per Success)', 
                    'outputs/figures/fig2_efficiency.png')

    # --- Data for Graph 3: Reduction vs. Bias ---
    # Source: HIGH_BIAS_OPENAI section, "Performance by Initial Bias Magnitude"
    reduction_by_bias_data = {
        'Iterative Naive': [100.00, 11.49, 26.77, 21.92, 30.95, 47.38],
        'Unified POLQUAD': [100.00, 17.62, 35.03, 24.37, 48.11, 47.28],
        'Full POLQUAD':    [100.00, 88.82, 73.69, 92.93, 83.34, 62.41]
    }
    # X-axis labels
    categories = ['Low (1.1-2.5)', 'Mid-Low (2.5-4.0)', 'Medium (4.0-5.5)', 
                  'Mid-High (5.5-7.0)', 'High (7.0-8.5)', 'Extreme (8.5-10.0)']
    
    # Create DataFrame with proper index
    df_reduction = pd.DataFrame(reduction_by_bias_data, index=categories)
    
    plot_reduction_by_bias(df_reduction, 
                           'Bias Reduction vs. Initial Bias (OpenAI, n=98)', 
                           'outputs/figures/fig3_reduction_vs_bias.png')

if __name__ == '__main__':
    main()