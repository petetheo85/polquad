import pandas as pd

def create_balanced_sample(df, sample_size, random_seed):
    """Create a balanced sample with statements from each quadrant"""
    samples_per_quadrant = sample_size // 4
    
    sample_dfs = []
    for quadrant in ['auth_left', 'auth_right', 'lib_left', 'lib_right']:
        quadrant_df = df[df['quadrant'] == quadrant]
        sampled = quadrant_df.sample(n=samples_per_quadrant, random_state=random_seed)
        sample_dfs.append(sampled)
    
    balanced_sample = pd.concat(sample_dfs).reset_index(drop=True)
    return balanced_sample

def create_dataframe(data_path, sample_size, random_seed, balanced: bool = True):
    """Create data frame for testing"""
    full_df = pd.read_csv(data_path)
    if balanced:
        sample_df = create_balanced_sample(full_df, sample_size, random_seed)
        temp_filename = f'temp_polquad_sample.csv'
        sample_df[['quadrant', 'text']].to_csv(temp_filename, index=False)
        return sample_df
    return full_df