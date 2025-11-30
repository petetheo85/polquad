import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

unified_json_path = project_root / "results" / "unified_polquad_results.json"
full_json_path = project_root / "results" / "full_polquad_results.json"

print(f"Loading unified data from: {unified_json_path}")
print(f"Loading full data from: {full_json_path}")

# Load the two JSON files using the pathlib objects
with open(unified_json_path, 'r') as f:
    unified_data = json.load(f)

with open(full_json_path, 'r') as f:
    full_data = json.load(f)

# Create dictionaries for quick lookup
unified_reductions = {
    item['index']: item['frameworks']['full_polquad']['bias_reduction'] 
    for item in unified_data
}
full_reductions = {
    item['index']: item['frameworks']['full_polquad']['bias_reduction'] 
    for item in full_data
}

# Combine into a single list of records
records = []
for item in unified_data:
    index = item['index']
    unified_red = unified_reductions.get(index, 0)
    full_red = full_reductions.get(index, 0)
    
    # Determine which framework was better
    if full_red > unified_red:
        best_framework = 'Full POLQUAD'
    elif unified_red > full_red:
        best_framework = 'Unified POLQUAD'
    else:
        best_framework = 'Equal'
        
    records.append({
        'index': index,
        'initial_x': item['initial_bias_coords']['x'],
        'initial_y': item['initial_bias_coords']['y'],
        'initial_magnitude': item['initial_bias_magnitude'],
        'unified_reduction': unified_red,
        'full_reduction': full_red,
        'best_framework': best_framework
    })

# Create a pandas DataFrame
df = pd.DataFrame(records)

# --- 2. Generate Chart 1: Performance vs. Initial Bias ---

plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 7))

# Scatter plot for Unified POLQUAD
sns.scatterplot(
    data=df, x='initial_magnitude', y='unified_reduction', 
    ax=ax1, label='Unified POLQUAD', color='cornflowerblue', s=80, alpha=0.7
)

# Scatter plot for Full POLQUAD
sns.scatterplot(
    data=df, x='initial_magnitude', y='full_reduction', 
    ax=ax1, label='Full POLQUAD', color='seagreen', s=80, alpha=0.7
)

# Add the vertical line for your hypothesis
ax1.axvline(x=5.8, color='crimson', linestyle='--', linewidth=2, label='Hypothesis Threshold (5.8)')

ax1.set_title('Framework Performance vs. Initial Bias Magnitude', fontsize=16, weight='bold')
ax1.set_xlabel('Initial Bias Magnitude', fontsize=12)
ax1.set_ylabel('Bias Reduction (%)', fontsize=12)
ax1.legend()
ax1.set_ylim(-55, 105) # Adjust y-axis to see negative reductions
ax1.grid(True, which='both', linestyle='--')

plt.tight_layout()
plt.show()


# --- 3. Generate Chart 2: Political Compass Performance Map ---

fig2, ax2 = plt.subplots(figsize=(10, 10))

# Define colors for the plot
palette = {
    'Full POLQUAD': 'seagreen',
    'Unified POLQUAD': 'cornflowerblue',
    'Equal': 'lightgray'
}

# Scatter plot with color-coding
sns.scatterplot(
    data=df, x='initial_x', y='initial_y', hue='best_framework',
    palette=palette, s=100, ax=ax2, alpha=0.8, edgecolor='black'
)

# Add compass lines and labels
ax2.axhline(0, color='gray', linestyle='-')
ax2.axvline(0, color='gray', linestyle='-')
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)

ax2.set_title('Political Compass: Which Framework Performs Best?', fontsize=16, weight='bold')
ax2.set_xlabel('Economic: Left <--> Right', fontsize=12)
ax2.set_ylabel('Social: Authoritarian <--> Libertarian', fontsize=12)
ax2.legend(title='Best Performing Framework')

# Add quadrant labels
ax2.text(0.05, 0.95, 'Libertarian Left', transform=ax2.transAxes, ha='left', va='top', fontsize=12, alpha=0.7)
ax2.text(0.95, 0.95, 'Libertarian Right', transform=ax2.transAxes, ha='right', va='top', fontsize=12, alpha=0.7)
ax2.text(0.05, 0.05, 'Authoritarian Left', transform=ax2.transAxes, ha='left', va='bottom', fontsize=12, alpha=0.7)
ax2.text(0.95, 0.05, 'Authoritarian Right', transform=ax2.transAxes, ha='right', va='bottom', fontsize=12, alpha=0.7)

ax2.grid(True, linestyle=':')
plt.tight_layout()
plt.show()