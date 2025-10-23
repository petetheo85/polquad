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
from polquad.utils.chat_gpt import ChatGPTClient
from polquad.utils.claude import ClaudeClient
from polquad.utils.bias_calculator import BiasCalculator
from polquad.utils.data_helper import create_dataframe
from polquad.frameworks.naive import NaiveFramework
from polquad.frameworks.polquad import PolquadFramework
from polquad.frameworks.unified import UnifiedPolquadFramework

# Configuration
SAMPLE_SIZE = polquad_configs['sample_size']
BALANCED_SAMPLING = polquad_configs['balanced_sampling']
RANDOM_SEED = polquad_configs['random_seed']
DATASET_PATH = polquad_configs['dataset_path']
DATASET_SOURCE_TYPE = polquad_configs['dataset_source_type']
BIAS_THRESHOLD = polquad_configs['bias_threshold']
NUMBER_OF_RUNS = polquad_configs['num_runs']
OUTPUT_FILE_PATH = polquad_configs['output_file_path']

# Save results to JSON
def save_results(all_results, output_path):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

# Resume processing from checkpoint
def load_checkpoint(output_path):
    """Load existing results from checkpoint file if it exists."""
    try:
        with open(output_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded checkpoint from {output_path} with {len(results)} statements processed.")
        return results
    except FileNotFoundError:
        print(f"No checkpoint found. Starting fresh.")
        return []
    
def get_processed_indices(all_results):
    """Get set of statement indices already processed."""
    return set(result['index'] for result in all_results)

def main():
    print("\n\n\n")
    print_startup_screen()

    # Get dataset
    print("\nGenerating Dataframe from Dataset...")
    df = create_dataframe(
        DATASET_PATH, 
        SAMPLE_SIZE, 
        RANDOM_SEED,
        balanced=BALANCED_SAMPLING,
        source=DATASET_SOURCE_TYPE)

    # Initialize LLM
    print("Initializing LLM Client...", end="", flush=True)
    client = ClaudeClient()
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

    all_results = load_checkpoint(OUTPUT_FILE_PATH)
    processed_indices = get_processed_indices(all_results) or set()

    print("Setup complete!")

    # Main Loop that iterates through each statement
    for index, row in df.iterrows():
        # Skip if already processed
        if index in processed_indices:
            print(f"Skipping statement {index} (already processed).")
            continue
        original_statement = row['text']
        print_statement_header(index, original_statement)

        # Calculate inital bias
        initial_bias_coords, initial_bias_mag = bias_calculator.calculate_bias(original_statement)   
        print(f"Initial Bias Coordinates: ({initial_bias_coords['x']}, {initial_bias_coords['y']})") 
        print(f"Initial Bias Magnitude: {initial_bias_mag}")
        
        statement_results = {
            "index": index,
            "original_statement": original_statement,
            "true_label": row['quadrant'] if 'quadrant' in row else None,
            "initial_bias_coords": initial_bias_coords,
            "initial_bias_magnitude": initial_bias_mag,
            "framework_comparison": {}
        }

        # Check if statement is unbiased
        if initial_bias_mag <= BIAS_THRESHOLD:
            print(f"--> Initial bias is below threshold. Skipping framework runs.")
            for name in frameworks.keys():
                statement_results['framework_comparison'][name] = {
                    'runs': [{'run_id': i + 1, 'status': "SKIPPED"} for i in range(NUMBER_OF_RUNS)],
                    'averages': {
                        'avg_final_magnitude': initial_bias_mag,
                        'avg_iterations': 0,
                        'avg_bias_reduction': 0.0
                    }
                }
            all_results.append(statement_results)
            save_results(all_results, OUTPUT_FILE_PATH)
            continue

        # Framework Loop - runs all frameworks on each statement
        for name, runner in frameworks.items():
            print_framework_header(name, NUMBER_OF_RUNS)

            run_data = []
            metrics_to_average = {
                "final_magnitude": [],
                "iterations": [],
                "bias_reduction": []
            }

            # Trial Loop - runs each framework multiple times for smoothing
            for i in range(NUMBER_OF_RUNS):
                try:
                    result = runner.run_analysis(index, row, initial_bias_coords, initial_bias_mag)

                    # Extract initial bias info on first run
                    if i == 0 and name == "full_polquad":
                        statement_results["initial_bias_coords"] = result.get("initial_bias_coords")
                        statement_results["initial_bias_magnitude"] = result.get("initial_bias_magnitude")

                    # Extract relevant metrics
                    fw_result = result.get("frameworks", {}).get(name, {})
                    final_mag = fw_result.get("final_bias_magnitude", 0)
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
                    print(f"  • Run {i+1}/{NUMBER_OF_RUNS}: Final Bias Magnitude = {final_mag:.2f}, Bias Reduction = {bias_reduction:.2f}%, Iterations = {num_iters}")

                except Exception as e:
                    print_error(i + 1, index, e)
                    run_data.append({
                        "run_id": i + 1,
                        "status": "FAILED",
                        "error": str(e),
                    })

            # Calculate averages
            averages = {
                "avg_final_magnitude": statistics.mean(metrics_to_average['final_magnitude']) if metrics_to_average['final_magnitude'] else 0,
                "avg_iterations": statistics.mean(metrics_to_average['iterations']) if metrics_to_average['iterations'] else 0,
                "avg_bias_reduction": statistics.mean(metrics_to_average['bias_reduction']) if metrics_to_average['bias_reduction'] else 0
            }
            print_summary(name, averages)

            statement_results["framework_comparison"][name] = {
                "runs": run_data,
                "averages": averages
            }

        # Add statement to results and save progress
        all_results.append(statement_results)
        save_results(all_results, OUTPUT_FILE_PATH)

    # Output to JSON
    print(f"\n{'='*20} Analysis Complete {'='*20}")
    print(f"Saving all results to {OUTPUT_FILE_PATH}...", end="", flush=True)
    with open(OUTPUT_FILE_PATH, 'w') as f:
        json.dump(all_results, f, indent = 2)
    print("DONE.")

    print_main_header("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
