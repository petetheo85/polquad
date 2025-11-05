import json

# ============================================================================
# EDIT THESE FILE PATHS BELOW
# ============================================================================

FILE_ENDING = "seed42_100_high_bias_openai.json"
OLD_RESULTS_FILE = f"results/outputs/framework_output_{FILE_ENDING}"
NEW_NAIVE_FILE = f"results/outputs/framework_output_naive_{FILE_ENDING}"
OUTPUT_FILE = f"results/outputs/merged_output_{FILE_ENDING}"

# ============================================================================
# END OF CONFIGURATION
# ============================================================================

def merge_naive_results(old_file_path, new_file_path, output_file_path):
    """
    Merge iterative naive results from new file into old file.
    
    Args:
        old_file_path: Path to original results file with all frameworks
        new_file_path: Path to new naive-only results file
        output_file_path: Path to write merged results
    """
    
    try:
        # Load files
        print(f"Loading old results from: {old_file_path}")
        with open(old_file_path, 'r') as f:
            old_results = json.load(f)
        
        print(f"Loading new naive results from: {new_file_path}")
        with open(new_file_path, 'r') as f:
            new_results = json.load(f)
        
        # Create lookup for new results by index
        new_results_by_index = {result['index']: result for result in new_results}
        
        # Merge results
        merged_count = 0
        not_found_count = 0
        
        for old_result in old_results:
            index = old_result['index']
            
            if index in new_results_by_index:
                new_result = new_results_by_index[index]
                
                # Replace naive results in framework_comparison
                if 'framework_comparison' in old_result and 'naive' in new_result.get('framework_comparison', {}):
                    old_result['framework_comparison']['naive'] = new_result['framework_comparison']['naive']
                    merged_count += 1
                else:
                    print(f"Warning: Statement {index} missing naive results in new file")
                    not_found_count += 1
            else:
                print(f"Warning: Statement {index} not found in new results file")
                not_found_count += 1
        
        # Write merged results
        print(f"Writing merged results to: {output_file_path}")
        with open(output_file_path, 'w') as f:
            json.dump(old_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Merge complete!")
        print(f"{'='*60}")
        print(f"Successfully merged: {merged_count} statements")
        if not_found_count > 0:
            print(f"Warnings: {not_found_count} statements had issues")
        print(f"Output saved to: {output_file_path}")
        print(f"{'='*60}\n")
        
    except FileNotFoundError as e:
        print(f"\nERROR: Could not find file - {e}")
        print("Please check the file paths at the top of the script.\n")
    except json.JSONDecodeError as e:
        print(f"\nERROR: Invalid JSON format - {e}")
        print("Please check that your input files are valid JSON.\n")
    except Exception as e:
        print(f"\nERROR: {e}\n")

if __name__ == "__main__":
    merge_naive_results(OLD_RESULTS_FILE, NEW_NAIVE_FILE, OUTPUT_FILE)