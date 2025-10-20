import pandas as pd
from datasets import load_dataset

def create_balanced_sample(df, sample_size, random_seed):
    """Create a balanced sample with statements from each quadrant"""
    
    print("  ↳ [Data Helper] Creating a balanced sample from the dataset...", end="", flush=True)
    samples_per_quadrant = sample_size // 4
    sample_dfs = []

    for quadrant in ['auth_left', 'auth_right', 'lib_left', 'lib_right']:
        quadrant_df = df[df['quadrant'] == quadrant]
        sampled = quadrant_df.sample(n=samples_per_quadrant, random_state=random_seed)
        sample_dfs.append(sampled)
    
    balanced_sample = pd.concat(sample_dfs).reset_index(drop=True)
    print("DONE.")

    return balanced_sample

def create_dataframe(data_path, sample_size, random_seed, balanced: bool = True, source: str = 'local'):
    """Create a pandas dataframe for testing from a local CSV or HuggingFace."""

    if source == "hf":
        print("  ↳ [Data Helper] Loading dataset from Hugging Face...", end="", flush=True)
        dataset = load_dataset(data_path)
        full_df = dataset['train'].to_pandas()
    else:
        print("  ↳ [Data Helper] Loading dataset from local CSV file...", end="", flush=True)
        full_df = pd.read_csv(data_path)
    print("DONE.")
    

    if balanced and 'quadrant' in full_df.columns:
        print("  ↳ [Data Helper] Creating a balanced sample of political quadrants...", end="", flush=True)
        sample_df = create_balanced_sample(full_df, sample_size, random_seed)
        print("DONE.")
    else:
        if balanced and 'quadrant' not in full_df.columns:
            print("  ↳ [Data Helper] 'quadrant' column not found. Creating random sample instead...")
        print("  ↳ [Data Helper] Creating a random sample from the dataset...", end="", flush=True)
        sample_df = full_df.sample(n=sample_size, random_state=random_seed)
        print("DONE.")

    temp_filename = f"temp_data_sample.csv"
    sample_df.to_csv(temp_filename, index=False)
    return sample_df    
