import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
import statistics
from matplotlib.lines import Line2D
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

from polquad.config import polquad_configs
from polquad.utils.formatter import (
    print_startup_screen,
    print_main_header,
    print_statement_header,
    print_framework_header,
    print_error,
    print_summary
)
from polquad.utils.gemini_client import GeminiClient
from polquad.utils.openai_client import ChatGPTClient
from polquad.utils.claude_client import ClaudeClient
from polquad.utils.bias_calculator import BiasCalculator
from polquad.frameworks.polquad import PolquadFramework

# --- Configuration (can be modified for testing) ---
# Define a single statement for plotting
STATEMENT_FOR_PLOT = {
    "text": "Private corporations should be completely free from government regulation, and we must enforce strict, traditional national values to ensure social order.",
    "quadrant": "Authoritarian Right" # Optional: for your reference
}
# Only run Full Polquad once for the plot
NUMBER_OF_RUNS = 1

# --- Mode Selection ---
# Set to "run" to run the analysis and save results
# Set to "plot" to load results and create plots only
MODE = "plot"  # Change to "plot" to skip analysis and just visualize
RESULTS_FILE = project_root / "outputs/runs/polquad_indv_results.json"

# --- Utility Functions ---
def save_results_to_json(statement, history, initial_bias, initial_mag, final_bias_mag, bias_reduction, filename=RESULTS_FILE):
    """Save analysis results to a JSON file for later plotting."""
    data = {
        "statement": statement,
        "initial_bias": initial_bias,
        "initial_bias_magnitude": initial_mag,
        "final_bias_magnitude": final_bias_mag,
        "bias_reduction_percent": bias_reduction,
        "moderation_history": history
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filename}")

def load_results_from_json(filename=RESULTS_FILE):
    """Load analysis results from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# --- Plotting Function ---
def plot_polquad_iteration_history(statement, history, filename="polquad_iteration_plot.png"):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set the limits of the plot based on max coordinate values or a fixed range
    all_coords = []
    if history:
        for entry in history.values():
            if 'bias' in entry and entry['bias'] is not None:
                all_coords.append((entry['bias'].get('x', 0), entry['bias'].get('y', 0)))
    
    if all_coords:
        max_abs_coord = max(abs(c) for x, y in all_coords for c in (x, y))
        plot_range = max(10, int(max_abs_coord * 1.2 / 10) * 10 + 10) # Ensure a nice rounded range
    else:
        plot_range = 100 # Default range if no coordinates

    ax.set_xlim([-plot_range, plot_range])
    ax.set_ylim([-plot_range, plot_range])

    # Plot Quadrants with updated colors
    # Authoritarian Left (Top-Left): Blue-ish
    ax.add_patch(patches.Rectangle((-plot_range, 0), plot_range, plot_range, facecolor='lightsteelblue', alpha=0.3)) 
    # Authoritarian Right (Top-Right): Red-ish
    ax.add_patch(patches.Rectangle((0, 0), plot_range, plot_range, facecolor='lightcoral', alpha=0.3)) 
    # Libertarian Left (Bottom-Left): Green-ish
    ax.add_patch(patches.Rectangle((-plot_range, -plot_range), plot_range, plot_range, facecolor='lightgreen', alpha=0.3)) 
    # Libertarian Right (Bottom-Right): Yellow/Orange-ish
    ax.add_patch(patches.Rectangle((0, -plot_range), plot_range, plot_range, facecolor='khaki', alpha=0.3)) 

    # Plot Axes
    ax.axhline(0, color='black', linewidth=2)
    ax.axvline(0, color='black', linewidth=2)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.6)

    # Title and Labels
    ax.set_title(f"Iterative Refinement History", fontsize=24)
    ax.set_xlabel("Economic Axis (Left-Right)", fontsize=18)
    ax.set_ylabel("Social Axis (Auth-Lib)", fontsize=18)

    # Extract coordinates from history for plotting
    iteration_coords = []
    
    # Sort history by keys to ensure correct order
    sorted_history_keys = sorted(history.keys())

    # Define colors for each iteration
    iteration_colors_dict = {
        0: 'red',      # Original
        1: 'orange',   # Iter 1
        2: 'green'     # Iter 2
    }
    
    # Get initial (original) statement bias
    original_entry = history.get(0) or history.get("0")
    if original_entry and 'bias' in original_entry:
        original_x = original_entry['bias'].get('x', 0)
        original_y = original_entry['bias'].get('y', 0)
        iteration_coords.append((original_x, original_y))
        
        # Plot original point with defined color
        ax.plot(original_x, original_y, 'o', color=iteration_colors_dict[0], markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5, label='Original Statement', zorder=5)
        
        # Add label for the original statement
        ax.text(original_x, original_y - 0.95, f"Original", fontsize=14, color='black', 
                ha='left', va='bottom', zorder=5)

    # Collect all moderation steps (excluding original)
    moderation_steps = []
    for key in sorted_history_keys:
        if key == 0 or key == "0":
            continue
        entry = history[key]
        if 'bias' in entry and entry['bias'] is not None:
            x = entry['bias'].get('x', 0)
            y = entry['bias'].get('y', 0)
            moderation_steps.append((key, x, y))
    
    # Plot Moderation History points and arrows
    for step_idx, (key, x, y) in enumerate(moderation_steps):
        iteration_coords.append((x, y))
        
        # Get color from dictionary
        key_int = int(key)
        current_dot_color = iteration_colors_dict.get(key_int, 'gray')
        
        # Plot current point with black outline
        ax.plot(x, y, 'o', color=current_dot_color, markersize=10, 
                markeredgecolor='black', markeredgewidth=1.5, zorder=4)

        # Draw arrow from previous point with black line
        if len(iteration_coords) > 1:
            prev_x, prev_y = iteration_coords[-2]
            ax.annotate("", xy=(x, y), xytext=(prev_x, prev_y),
                        arrowprops=dict(facecolor='black', edgecolor='black', shrink=0.015,
                                      width=1, headwidth=8, headlength=8),
                        zorder=3)
        
        # Add label for the iteration
        ax.text(x + 0.25, y - 0.75, f"Refinement #{key}", fontsize=14, color='black', 
                ha='left', va='bottom', zorder=5)
    
    # Extract statement texts from history for display box
    statement_texts = []
    for key in sorted(history.keys(), key=lambda x: int(x)):
        entry = history[key]
        key_int = int(key)
        
        if key_int == 0:
            stmt = entry.get('original_statement', 'N/A')
            label = "Original"
        else:
            stmt = entry.get('moderated_statement', 'N/A')
            label = f"Refinement #{key_int}"
        
        color = iteration_colors_dict.get(key_int, 'gray')
        statement_texts.append((label, stmt, color))
    
    # Create text box with statements at the bottom of the plot
    if statement_texts:
        start_y = -0.25 
        line_height_pixels = 20 
        
        # Calculate total height needed for text box
        total_text_height_pixels = 0
        text_lines_to_draw = []
        for label, stmt, color in statement_texts:
            text_line = f"{label}: {stmt}"
            lines = 1 + (len(text_line) // 40) 
            total_text_height_pixels += lines * line_height_pixels
            text_lines_to_draw.append((label, text_line, color, lines))
        
        fig_height_pixels = fig.get_window_extent().height
        total_text_height_fig_frac = (total_text_height_pixels / fig_height_pixels) + 0.05
        
        plt.subplots_adjust(bottom=total_text_height_fig_frac + 0.05, top=0.95)
        
        # Draw the box
        rect = patches.Rectangle(
            (0.01, 0.01), 
            0.98, 
            total_text_height_fig_frac - 0.02,
            transform=fig.transFigure, 
            facecolor='wheat', 
            edgecolor='black',
            linewidth=1.5,
            zorder=50 
        )
        fig.patches.append(rect)

        # --- Draw the Text ---
        current_y_fig_frac = total_text_height_fig_frac - 0.02
        
        for label, text_line, color, num_lines in text_lines_to_draw:
            fig.text(0.035, current_y_fig_frac - 0.005, "●", 
                    transform=fig.transFigure, 
                    fontsize=16, 
                    color=color, 
                    ha='left', 
                    va='top',
                    fontweight='bold',
                    zorder=51) 
            
            # Draw the text line (with wrap)
            fig.text(0.07, current_y_fig_frac, text_line, 
                    transform=fig.transFigure, 
                    fontsize=12,
                    verticalalignment='top', 
                    horizontalalignment='left', 
                    wrap=True,
                    zorder=51) 
            
            # Move Y down for next line
            current_y_fig_frac -= 0.05 
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Plot saved to {filename}")
    plt.close(fig)

# --- Main function to run the POLQUAD framework and then plot ---
def run_analysis():
    """Run the POLQUAD framework and save results."""
    print("\n\n\n")
    print_startup_screen()

    # Initialize LLM
    print("Initializing LLM Client...", end="", flush=True)
    client = GeminiClient()
    print("DONE.")

    # Initialize bias calcualtor
    print("Intializing Bias Calculator...")
    bias_calculator = BiasCalculator(client, polquad_configs)

    # Instantiate only the Full Polquad framework
    print("Initializing Full POLQUAD framework...")
    full_polquad_runner = PolquadFramework(config=polquad_configs, client=client, bias_calculator=bias_calculator)

    print("Setup complete!")

    original_statement = STATEMENT_FOR_PLOT['text']
    true_quadrant = STATEMENT_FOR_PLOT.get('quadrant', 'N/A')
    print_statement_header(0, original_statement)

    # Calculate initial bias
    initial_bias_coords, initial_bias_mag = bias_calculator.calculate_bias(original_statement)   
    initial_classification = get_quadrant_label(initial_bias_coords)
    
    print("\n" + "-"*20)
    print(f"Original Content: {original_statement}")
    print(f"Original Bias Coordinates: {initial_bias_coords}")
    print(f"Original Bias Magnitude: {initial_bias_mag:.4f}")
    print(f"Original Classification: {initial_classification}")
    print(f"True Label (from input): {true_quadrant}")
    print("-"*20)
    
    # Run Full POLQUAD framework
    print_framework_header("full_polquad", NUMBER_OF_RUNS)

    try:
        # Pass STATEMENT_FOR_PLOT as statement_data
        result = full_polquad_runner.run_analysis(0, STATEMENT_FOR_PLOT, initial_bias_coords, initial_bias_mag)

        # Extract history for plotting
        full_polquad_result = result.get("frameworks", {}).get("full_polquad", {})
        moderation_history = full_polquad_result.get('moderation_history', {})
        final_bias_mag = full_polquad_result.get('final_bias_magnitude', 0)
        bias_reduction = full_polquad_result.get('bias_reduction', 0)

        print(f"\n  --- full_polquad Summary ---")
        print(f"  • Run 1/1: Final Bias Magnitude = {final_bias_mag:.4f}, Bias Reduction = {bias_reduction:.2f}%, Iterations = {full_polquad_result.get('num_iterations', 0)}")

        # Save results to JSON
        save_results_to_json(
            statement=original_statement,
            history=moderation_history,
            initial_bias=initial_bias_coords,
            initial_mag=initial_bias_mag,
            final_bias_mag=final_bias_mag,
            bias_reduction=bias_reduction
        )

        print_main_header("ANALYSIS COMPLETE")

    except Exception as e:
        print_error(1, 0, e)

def plot_only():
    """Load saved results and create plot only."""
    print(f"\nLoading results from {RESULTS_FILE}...")
    
    try:
        data = load_results_from_json(RESULTS_FILE)
        
        statement = data['statement']
        history = data['moderation_history']
        
        print(f"Loaded statement: {statement[:80]}...")
        print(f"Number of iterations: {len([k for k in history.keys() if k != 0])}")
        
        # Create plot in outputs/figures/
        output_plot_path = project_root / "outputs/figures/full_polquad_iteration_plot.png"
        plot_polquad_iteration_history(
            statement=statement,
            history=history,
            filename=str(output_plot_path)
        )
        
        print_main_header("PLOT COMPLETE")
        
    except FileNotFoundError:
        print(f"Error: Results file '{RESULTS_FILE}' not found.")
        print("Please run with MODE='run' first to generate results.")
    except Exception as e:
        print(f"Error loading or plotting results: {e}")

def main():
    """Main entry point - choose between running analysis or plotting only."""
    if MODE == "run":
        run_analysis()
    elif MODE == "plot":
        plot_only()
    else:
        print(f"Invalid MODE: {MODE}. Set to 'run' or 'plot'.")

# Helper function for quadrant labels
def get_quadrant_label(coords):
    """Classifies bias coordinates into a political quadrant."""
    try:
        x = coords.get('x', 0)
        y = coords.get('y', 0)

        center_threshold = 0.5 

        is_x_center = abs(x) < center_threshold
        is_y_center = abs(y) < center_threshold

        if is_x_center and is_y_center:
            return "Neutral / Centrist"
        
        if is_x_center:
            if y > 0:
                return "Authoritarian Center"
            else:
                return "Libertarian Center"
        
        if is_y_center:
            if x > 0:
                return "Right (Economic)"
            else:
                return "Left (Economic)"

        if x > 0 and y > 0:
            return "Authoritarian Right"
        elif x < 0 and y > 0:
            return "Authoritarian Left"
        elif x < 0 and y < 0:
            return "Libertarian Left"
        elif x > 0 and y < 0:
            return "Libertarian Right"
            
    except Exception:
        return "N/A"
    return "Unknown"


if __name__ == "__main__":
    main()