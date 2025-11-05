import json
import statistics
from config import polquad_configs
from polquad.utils.formatter import (
    print_startup_screen,
    print_main_header,
    print_statement_header,
    print_framework_header,
    print_error,
    print_summary
)
from polquad.utils.gemini import GeminiClient
from polquad.utils.bias_calculator import BiasCalculator

from polquad.frameworks.naive import NaiveFramework
from polquad.frameworks.polquad import PolquadFramework
from polquad.frameworks.unified import UnifiedPolquadFramework

# === Configuration ===

# Define the specific statements you want to run the showcase on.
# You can add as many as you like.
STATEMENTS_TO_RUN = [
    {
        "text": "This is a placeholder for your first handpicked biased statement.",
        "quadrant": "Authoritarian Left" # Optional: for your reference
    },
    {
        "text": "This is a placeholder for your second handpicked biased statement.",
        "quadrant": "Libertarian Right" # Optional: for your reference
    },
]

# Set number of runs to 1 for the showcase
NUMBER_OF_RUNS = 1 
OUTPUT_FILE_PATH = polquad_configs.get('output_file_path', 'showcase_results.json')
BIAS_THRESHOLD = polquad_configs.get('bias_threshold', 0.5) # Kept from original


# === Helper Functions ===

def save_results(all_results, output_path):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

def get_quadrant_label(coords):
    """Classifies bias coordinates into a political quadrant."""
    try:
        x = coords.get('x', 0)
        y = coords.get('y', 0)
        if x >= 0 and y >= 0:
            return "Authoritarian Right"
        elif x < 0 and y >= 0:
            return "Authoritarian Left"
        elif x < 0 and y < 0:
            return "Libertarian Left"
        elif x >= 0 and y < 0:
            return "Libertarian Right"
    except Exception:
        return "N/A"
    return "Neutral" # Should not be hit if logic is correct

def print_full_polquad_details(result_data):
    """Prints the detailed output for the Full Polquad framework."""
    print("\n=== FULL POLQUAD ===")
    try:
        # Assumes structure: result_data['opinion_agent_responses']['auth_left']['response']
        opinion_responses = result_data.get('opinion_agent_responses', {})
        print("Responses from Opinion Agents:")
        
        al = opinion_responses.get('auth_left', {})
        print(f"Authoritarian Left: {al.get('response', 'N/A')}")
        print(f"AL Bias Coordinates: {al.get('bias_coords', {})}")
        
        ar = opinion_responses.get('auth_right', {})
        print(f"Authoritarian Right: {ar.get('response', 'N/A')}")
        print(f"AR Bias Coordinates: {ar.get('bias_coords', {})}")
        
        ll = opinion_responses.get('lib_left', {})
        print(f"Libertarian Left: {ll.get('response', 'N/A')}")
        print(f"LL Bias Coordinates: {ll.get('bias_coords', {})}")
        
        lr = opinion_responses.get('lib_right', {})
        print(f"Libertarian Right: {lr.get('response', 'N/A')}")
        print(f"LR Bias Coordinates: {lr.get('bias_coords', {})}")

        # Assumes structure: result_data['judge_agent_responses'] is a list of dicts
        judge_responses = result_data.get('judge_agent_responses', [])
        if not judge_responses:
            print("\nNo Judge Agent responses found.")
        for i, judge_res in enumerate(judge_responses, 1):
            print(f"\nJudge Agent Response {i}: {judge_res.get('response', 'N/A')}")
            print(f"Judge Agent Response {i} Bias Coordinates: {judge_res.get('bias_coords', {})}")
            print(f"Judge Agent Response {i} Bias Magnitude: {judge_res.get('bias_magnitude', 'N/A')}")
            print(f"Judge Agent Response {i} Label: {judge_res.get('label', 'N/A')}")
    except Exception as e:
        print(f"Error printing Full Polquad details: {e}. Check result structure.")

def print_unified_polquad_details(result_data):
    """Prints the detailed output for the Unified Polquad framework."""
    print("\n=== UNIFIED POLQUAD ===")
    try:
        # Assumes structure: result_data['unified_agent_responses'] is a list of dicts
        unified_responses = result_data.get('unified_agent_responses', [])
        if not unified_responses:
            print("\nNo Unified Agent responses found.")
        for i, res in enumerate(unified_responses, 1):
            print(f"\nUnified Agent Response {i}: {res.get('response', 'N/A')}")
            print(f"Unified Agent Response {i} Bias Coordinates: {res.get('bias_coords', {})}")
            print(f"Unified Agent Response {i} Bias Magnitude: {res.get('bias_magnitude', 'N/A')}")
            print(f"Unified Agent Response {i} Label: {res.get('label', 'N/A')}")
    except Exception as e:
        print(f"Error printing Unified Polquad details: {e}. Check result structure.")

def print_naive_details(result_data):
    """Prints the detailed output for the Naive framework."""
    print("\n=== NAIVE ===")
    try:
        # Assumes structure: result_data['naive_agent_responses'] is a list of dicts
        naive_responses = result_data.get('naive_agent_responses', [])
        if not naive_responses:
            print("\nNo Naive Agent responses found.")
        for i, res in enumerate(naive_responses, 1):
            print(f"\nNaive Agent Response {i}: {res.get('response', 'N/A')}")
            print(f"Naive Agent Response {i} Bias Coordinates: {res.get('bias_coords', {})}")
            print(f"Naive Agent Response {i} Bias Magnitude: {res.get('bias_magnitude', 'N/A')}")
            print(f"Naive Agent Response {i} Label: {res.get('label', 'N/A')}")
    except Exception as e:
        print(f"Error printing Naive details: {e}. Check result structure.")


def main():
    print("\n\n\n")
    print_startup_screen()

    # Get dataset - REMOVED
    # print("\nGenerating Dataframe from Dataset...")
    # df = create_dataframe(...)

    # Initialize LLM
    print("Initializing LLM Client...", end="", flush=True)
    client = GeminiClient()
    print("DONE.")

    # Initialize bias calcualtor
    print("Intializing Bias Calculator...")
    bias_calculator = BiasCalculator(client, polquad_configs)

    # Instantiate framework runners
    print("Initializing frameworks...")
    frameworks = {
        "full_polquad": PolquadFramework(config=polquad_configs, client=client, bias_calculator=bias_calculator),
        "unified_polquad": UnifiedPolquadFramework(config=polquad_configs, client=client, bias_calculator=bias_calculator),
        "naive": NaiveFramework(config=polquad_configs, client=client, bias_calculator=bias_calculator)
    }

    # all_results = load_checkpoint(OUTPUT_FILE_PATH) # REMOVED
    # processed_indices = get_processed_indices(all_results) or set() # REMOVED
    all_results = []

    print("Setup complete!")

    # Main Loop that iterates through each handpicked statement
    for index, statement_data in enumerate(STATEMENTS_TO_RUN):
        # Skip if already processed - REMOVED
        # if index in processed_indices: ...
        
        original_statement = statement_data['text']
        true_quadrant = statement_data.get('quadrant', 'N/A')
        print_statement_header(index, original_statement)

        # Calculate inital bias
        initial_bias_coords, initial_bias_mag = bias_calculator.calculate_bias(original_statement)   
        initial_classification = get_quadrant_label(initial_bias_coords)
        
        # --- DETAILED ORIGINAL STATEMENT OUTPUT ---
        print("\n" + "-"*20)
        print(f"Original Content: {original_statement}")
        print(f"Original Bias Coordinates: {initial_bias_coords}")
        print(f"Original Bias Magnitude: {initial_bias_mag:.4f}")
        print(f"Original Classification: {initial_classification}")
        print(f"True Label (from input): {true_quadrant}")
        print("-"*20)
        
        statement_results = {
            "index": index,
            "original_statement": original_statement,
            "true_label": true_quadrant,
            "initial_bias_coords": initial_bias_coords,
            "initial_bias_magnitude": initial_bias_mag,
            "initial_classification": initial_classification,
            "framework_comparison": {}
        }

        # Check if statement is unbiased - REMOVED
        # if initial_bias_mag <= BIAS_THRESHOLD: ...

        # Framework Loop - runs all frameworks on each statement
        for name, runner in frameworks.items():
            print_framework_header(name, NUMBER_OF_RUNS)

            run_data = []
            metrics_to_average = {
                "final_magnitude": [],
                "iterations": [],
                "bias_reduction": []
            }

            # Trial Loop - REMOVED (code inside is run once)
            # for i in range(NUMBER_OF_RUNS):
            i = 0 # Run ID is 1
            try:
                # Use statement_data (which is a dict) instead of 'row'
                result = runner.run_analysis(index, statement_data, initial_bias_coords, initial_bias_mag)

                # Extract initial bias info on first run
                if i == 0 and name == "full_polquad":
                    statement_results["initial_bias_coords"] = result.get("initial_bias_coords", initial_bias_coords)
                    statement_results["initial_bias_magnitude"] = result.get("initial_bias_magnitude", initial_bias_mag)

                # Extract relevant metrics
                fw_result = result.get("frameworks", {}).get(name, {})
                
                # --- PRINT DETAILED FRAMEWORK OUTPUT ---
                if name == "full_polquad":
                    print_full_polquad_details(fw_result)
                elif name == "unified_polquad":
                    print_unified_polquad_details(fw_result)
                elif name == "naive":
                    print_naive_details(fw_result)
                # --- END DETAILED FRAMEWORK OUTPUT ---

                final_mag = fw_result.get("final_bias_magnitude", initial_bias_mag) # Default to initial
                num_iters = fw_result.get("num_iterations", 0)
                bias_reduction = fw_result.get("bias_reduction", 0)
                converged = fw_result.get("converged", False)

                run_data.append({
                    "run_id": i + 1,
                    "final_bias_magnitude": final_mag,
                    "num_iterations": num_iters,
                    "bias_reduction": bias_reduction,
                    "converged": converged
                })

                # Metrics to be averaged
                metrics_to_average['final_magnitude'].append(final_mag)
                metrics_to_average['iterations'].append(num_iters)
                metrics_to_average['bias_reduction'].append(bias_reduction)
                
                print(f"\n  --- {name} Summary ---")
                print(f"  • Run {i+1}/{NUMBER_OF_RUNS}: Final Bias Magnitude = {final_mag:.4f}, Bias Reduction = {bias_reduction:.2f}%, Iterations = {num_iters}")

            except Exception as e:
                print_error(i + 1, index, e)
                run_data.append({
                    "run_id": i + 1,
                    "status": "FAILED",
                    "error": str(e),
                })

            # Calculate "averages" (which is just the single run's data)
            averages = {
                "avg_final_magnitude": statistics.mean(metrics_to_average['final_magnitude']) if metrics_to_average['final_magnitude'] else 0,
                "avg_iterations": statistics.mean(metrics_to_average['iterations']) if metrics_to_average['iterations'] else 0,
                "avg_bias_reduction": statistics.mean(metrics_to_average['bias_reduction']) if metrics_to_average['bias_reduction'] else 0
            }
            # print_summary(name, averages) # This is redundant with the print above

            statement_results["framework_comparison"][name] = {
                "runs": run_data,
                "averages": averages,
                "details": fw_result # Store the raw details in the JSON output too
            }

        # Add statement to results and save progress
        all_results.append(statement_results)
        # save_results(all_results, OUTPUT_FILE_PATH) # REMOVED - will save at the end

    # Output to JSON
    print(f"\n{'='*20} Showcase Complete {'='*20}")
    print(f"Saving all results to {OUTPUT_FILE_PATH}...", end="", flush=True)
    save_results(all_results, OUTPUT_FILE_PATH)
    print("DONE.")

    print_main_header("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()