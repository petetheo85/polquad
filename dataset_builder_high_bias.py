import sys
import time
import csv
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from google.genai import errors as genai_errors

# Add project root to path to allow config/utils import
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
sys.path.append(str(project_root))

from config import polquad_configs
from src.polquad.utils.gemini import GeminiClient
from src.polquad.utils.bias_calculator import BiasCalculator

# --- Configuration ---
DATASET_NAME = "m-newhauser/senator-tweets"
TARGET_COUNT = 250
BIAS_THRESHOLD = 7.5
OUTPUT_FILE_PATH = project_root / "data/high_bias_dataset.csv"
MIN_STATEMENT_LENGTH = 25

def build_dataset():
    """
    Scans a Hugging Face dataset to find and collect statements with a bias
    magnitude above a specified threshold. Saves progress incrementally and can
    resume from the last processed statement.
    """
    print("\n--- Biased Dataset Builder ---")

    # --- Resume Logic ---
    start_index = 0
    found_statements_count = 0
    if OUTPUT_FILE_PATH.exists():
        print(f"Found existing dataset at {OUTPUT_FILE_PATH}. Resuming...")
        try:
            found_df = pd.read_csv(OUTPUT_FILE_PATH)
            if not found_df.empty:
                last_index = found_df['original_index'].iloc[-1]
                start_index = last_index + 1
                found_statements_count = len(found_df)
                print(f"Resuming scan from dataset index {start_index}.")
        except pd.errors.EmptyDataError:
             print("Existing dataset is empty. Starting from the beginning.")
             # Create file with header if it's empty
             pd.DataFrame(columns=['original_index', 'text']).to_csv(OUTPUT_FILE_PATH, index=False)
    else:
        # If no file exists, create it and write the header
        print("No existing dataset found. Creating new file.")
        OUTPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=['original_index', 'text']).to_csv(OUTPUT_FILE_PATH, index=False)

    if found_statements_count >= TARGET_COUNT:
        print(f"Target count of {TARGET_COUNT} statements already met. Exiting.")
        return

    # --- Initialization ---
    print("Initializing LLM Client...", end="", flush=True)
    client = GeminiClient()
    print("DONE.")

    print("Intializing Bias Calculator...")
    bias_calculator = BiasCalculator(client, polquad_configs)
    
    print(f"Loading dataset: {DATASET_NAME}...", end="", flush=True)
    full_dataset = load_dataset(DATASET_NAME, split='train')
    print("DONE.")
    
    pbar = tqdm(initial=found_statements_count, total=TARGET_COUNT, desc="Finding Biased Statements")

    # --- Main Loop ---
    # Open the output file in append mode to save progress
    with open(OUTPUT_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        for i, item in enumerate(full_dataset):
            # Skip records that have already been processed
            if i < start_index:
                continue
            
            # Stop if we've reached our target
            if pbar.n >= TARGET_COUNT:
                break

            statement_text = item.get('text')

            # --- Validation ---
            is_web_link = statement_text.strip().startswith('http')
            is_too_short = len(statement_text.strip()) < MIN_STATEMENT_LENGTH
            if not statement_text or is_web_link or is_too_short:
                continue

            try:
                time.sleep(0.5) # Throttle requests
                _, magnitude = bias_calculator.calculate_bias(statement_text)
                
                if magnitude >= BIAS_THRESHOLD:
                    # Write the new row directly to the CSV
                    writer.writerow([i, statement_text])
                    f.flush() # Ensure it's written immediately to disk
                    pbar.update(1)

            except genai_errors.ServerError as e:
                # Cleaner, single-line error message for server issues
                print(f"\n[SERVER ERROR] API unavailable for index {i}. Waiting 30s. Details: {e}")
                time.sleep(30)
                continue
            except Exception as e:
                # Cleaner, single-line error message for all other issues
                print(f"\n[ERROR] Skipping statement at index {i} due to: {type(e).__name__}")
                continue
            
    pbar.close()
    print(f"\nProcess complete. Total biased statements in file: {pbar.n}")
    print(f"Dataset saved at: {OUTPUT_FILE_PATH}")

if __name__ == "__main__":
    build_dataset()

