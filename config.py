polquad_configs = {
    "max_iters": 3,
    "bias_threshold": 1.0,
    "bias_calc_runs": 3,
    "sample_size": 100,
    "balanced_sampling": False,
    "random_seed": 42,
    "dataset_path": 'data/high_bias_dataset.csv',
    "dataset_source_type": 'local', # local or hf (hugging face)
    "output_file_path": 'results/outputs/framework_output.json',
    "num_runs": 3,
    "verbose": False
}
