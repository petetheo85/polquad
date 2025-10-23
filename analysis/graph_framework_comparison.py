import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))
from config import polquad_configs

def generate_graphs():
    """Loads the framework output and generates performance map."""
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent
    data_file_path = project_root / "results/outputs/framework_output.json"
    bias_threshold = polquad_configs['bias_threshold']
    
    print(f"Loading data from: {data_file_path}")
    try:
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Could not find the file at {data_file_path}.")
        return

    # Process data
    records = []
    for item in data:
        # Skip items that might have errors or are missing data
        if "framework_comparison" not in item:
            continue

        # Get the averages for each framework if statement was biased
        if item['initial_bias_magnitude'] <= bias_threshold:
            best_framework = 'Initially Neutral'
        else:
            full_avg = item['framework_comparison']['full_polquad']['averages']
            unified_avg = item['framework_comparison']['unified_polquad']['averages']
            naive_avg = item['framework_comparison']['naive']['averages']

            # Sort frameworks by performance
            performance = [
                (full_avg.get('avg_bias_reduction', 0), full_avg.get('avg_iterations', 0), "Full POLQUAD"),
                (unified_avg.get('avg_bias_reduction', 0), unified_avg.get('avg_iterations', 0), "Unified POLQUAD"),
                (naive_avg.get('avg_bias_reduction', 0), naive_avg.get('avg_iterations', 0), "Naive")
            ]
            performance.sort(key=lambda x: (-x[0], x[1]))

            best_perf = performance[0]
            second_best_perf = performance[1]
            best_framework = 'Equal'

            # Main comparison: Average Bias Reduction
            if best_perf[0] > second_best_perf[0] + 0.001:
                best_framework = best_perf[2]

            # Tie-breaker when two iterative frameworks tie for bias reduction
            elif best_perf[2] != "Naive" and second_best_perf[2] != "Naive":
                if best_perf[1] < second_best_perf[1]:
                    best_framework = f"{best_perf[2]} (Fewer Iterations)"
        
        records.append({
            'initial_x': item['initial_bias_coords']['x'],
            'initial_y': item['initial_bias_coords']['y'],
            'best_framework': best_framework
        })

    df = pd.DataFrame(records)

    # Adding jitter to prevent datapoint stacking
    jitter_strength = 0.15
    df['jitter_x'] = df['initial_x'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))
    df['jitter_y'] = df['initial_y'] + np.random.uniform(-jitter_strength, jitter_strength, size=len(df))

    # Generate political compass graph
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))

    palette = {
        'Full POLQUAD': 'darkgreen',
        'Unified POLQUAD': 'darkblue',
        'Naive': "darkorange",
        'Full POLQUAD (Fewer Iterations)': 'mediumseagreen',
        'Unified POLQUAD (Fewer Iterations)': 'cornflowerblue',
        'Equal': 'lightgray',
        'Initially Neutral': 'plum'
    }

    sns.scatterplot(
        data=df, x='jitter_x', y='jitter_y', hue='best_framework',
        palette=palette, s=100, ax=ax, alpha=0.8, edgecolor='black'
    )

    ax.axhline(0, color='gray', linestyle='-')
    ax.axvline(0, color='gray', linestyle='-')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_title('Political Compass: Which Framework Performs Best?', fontsize=16, weight='bold')
    ax.set_xlabel('Economic: Left <--> Right', fontsize=12)
    ax.set_ylabel('Social: Authoritarian <--> Libertarian', fontsize=12)
    ax.legend(title='Best Performing Framework', loc='upper left', bbox_to_anchor=(1.02, 1))

    ax.text(0.05, 0.95, 'Libertarian Left', transform=ax.transAxes, ha='left', va='top', fontsize=12, alpha=0.7)
    ax.text(0.95, 0.95, 'Libertarian Right', transform=ax.transAxes, ha='right', va='top', fontsize=12, alpha=0.7)
    ax.text(0.05, 0.05, 'Authoritarian Left', transform=ax.transAxes, ha='left', va='bottom', fontsize=12, alpha=0.7)
    ax.text(0.95, 0.05, 'Authoritarian Right', transform=ax.transAxes, ha='right', va='bottom', fontsize=12, alpha=0.7)
    
    ax.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(project_root / "results" / "graphs"/ "compass_performance.png")
    plt.show()

if __name__ == '__main__':
    generate_graphs()