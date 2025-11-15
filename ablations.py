import json
import statistics
from tqdm import tqdm
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
from polquad.utils.data_helper import create_dataframe
from polquad.frameworks.polquad import PolquadFramework

# Configuration
SAMPLE_SIZE = polquad_configs['sample_size']
BALANCED_SAMPLING = polquad_configs['balanced_sampling']
RANDOM_SEED = polquad_configs['random_seed']
DATASET_PATH = polquad_configs['dataset_path']
DATASET_SOURCE_TYPE = polquad_configs['dataset_source_type']
BIAS_THRESHOLD = polquad_configs['bias_threshold']
NUMBER_OF_RUNS = polquad_configs['num_runs']
OUTPUT_FILE_PATH = 'results/outputs/ablation_output.json'

# Define ablation combinations
ABLATION_COMBINATIONS = {
    # "full_polquad": ['lib_left', 'lib_right', 'auth_left', 'auth_right'],
    "al_only": ['auth_left'],
    "ar_only": ['auth_right'],
    "ll_only": ['lib_left'],
    "lr_only": ['lib_right'],
    "al_ar": ['auth_left', 'auth_right'],
    "ll_lr": ['lib_left', 'lib_right'],
    "al_ll": ['auth_left', 'lib_left'],
    "ar_lr": ['auth_right', 'lib_right'],
    "al_lr": ['auth_left', 'lib_right'],
    "ar_ll": ['auth_right', 'lib_left']
}

def save_results(all_results, output_path):
    """Save results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_path}")

def main():
    print("\n\n\n")
    print_startup_screen()

    # Get dataset (same for all ablations)
    print("\nGenerating Dataframe from Dataset...")
    df = create_dataframe(
        DATASET_PATH, 
        SAMPLE_SIZE, 
        RANDOM_SEED,
        balanced=BALANCED_SAMPLING,
        source=DATASET_SOURCE_TYPE)

    # Initialize LLM
    print("Initializing LLM Client...", end="", flush=True)
    client = GeminiClient()
    print("DONE.")

    # Initialize bias calculator
    print("Initializing Bias Calculator...")
    bias_calculator = BiasCalculator(client, polquad_configs)

    all_results = []

    # Pre-calculate initial bias for all statements (once)
    print("\nPre-calculating initial bias for all statements...")
    initial_biases = {}
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating initial biases"):
        statement = row['text']
        initial_bias_coords, initial_bias_mag = bias_calculator.calculate_bias(statement)
        initial_biases[index] = {
            'coords': initial_bias_coords,
            'magnitude': initial_bias_mag
        }
    print(f"Initial bias calculated for {len(initial_biases)} statements.")

    # Ablation Loop - iterate through each ablation combination
    for ablation_name, opinion_agents in ABLATION_COMBINATIONS.items():
        print(f"\n{'='*60}")
        print(f"Running Ablation: {ablation_name.upper()}")
        print(f"Opinion Agents: {', '.join(opinion_agents)}")
        print(f"{'='*60}\n")

        # Initialize framework with specified agents
        print("Initializing framework...")
        framework = PolquadFramework(
            config=polquad_configs, 
            client=client, 
            bias_calculator=bias_calculator,
            opinion_agents=opinion_agents
        )

        # Statement Loop - run ablation on all statements
        for index, row in df.iterrows():
            original_statement = row['text']
            initial_bias_coords = initial_biases[index]['coords']
            initial_bias_mag = initial_biases[index]['magnitude']

            print_statement_header(index, original_statement)
            print(f"Initial Bias Coordinates: ({initial_bias_coords['x']}, {initial_bias_coords['y']})")
            print(f"Initial Bias Magnitude: {initial_bias_mag}")

            statement_results = {
                "index": index,
                "original_statement": original_statement,
                "true_label": row['quadrant'] if 'quadrant' in row else None,
                "initial_bias_coords": initial_bias_coords,
                "initial_bias_magnitude": initial_bias_mag,
                "ablation_name": ablation_name,
                "opinion_agents": opinion_agents,
                "framework_results": {}
            }

            # Check if statement is unbiased
            if initial_bias_mag <= BIAS_THRESHOLD:
                print(f"--> Initial bias is below threshold. Skipping framework runs.")
                statement_results["framework_results"] = {
                    'runs': [{'run_id': i + 1, 'status': "SKIPPED"} for i in range(NUMBER_OF_RUNS)],
                    'averages': {
                        'avg_final_magnitude': initial_bias_mag,
                        'avg_iterations': 0,
                        'avg_bias_reduction': 0.0
                    }
                }
                all_results.append(statement_results)
                continue

            # Trial Loop - run framework multiple times for smoothing
            print_framework_header(ablation_name, NUMBER_OF_RUNS)
            run_data = []
            metrics_to_average = {
                "final_magnitude": [],
                "iterations": [],
                "bias_reduction": []
            }

            for i in range(NUMBER_OF_RUNS):
                try:
                    result = framework.run_analysis(index, row, initial_bias_coords, initial_bias_mag)

                    # Extract relevant metrics
                    fw_result = result.get("frameworks", {}).get("full_polquad", {})
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
            print_summary(ablation_name, averages)

            statement_results["framework_results"] = {
                "runs": run_data,
                "averages": averages
            }

            all_results.append(statement_results)

    # Save all results
    print(f"\n{'='*20} Analysis Complete {'='*20}")
    print(f"Saving all ablation results to {OUTPUT_FILE_PATH}...", end="", flush=True)
    save_results(all_results, OUTPUT_FILE_PATH)
    print("DONE.")

    print_main_header("ABLATION ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()