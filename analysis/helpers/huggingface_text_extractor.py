import argparse
from datasets import load_dataset

def extract_text_from_hf(dataset_name, output_file):
    """
    Loads a dataset from Hugging Face, extracts the 'text' column,
    and saves it to a new text file. Each entry is separated by a newline.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.
        output_file (str): The path to the output text file where results will be saved.
    """
    try:
        print(f"Loading dataset '{dataset_name}' from Hugging Face...")
        # Load the dataset from the Hugging Face Hub
        ds = load_dataset(dataset_name)
        print("Dataset loaded successfully.")

        # Datasets from Hugging Face are often split into 'train', 'test', etc.
        # We'll assume the data is in the 'train' split.
        if 'train' not in ds:
            print(f"Error: 'train' split not found in the dataset.")
            print(f"Available splits are: {list(ds.keys())}")
            return

        dataset_split = ds['train']

        with open(output_file, 'w', encoding='utf-8') as outfile:
            print(f"Extracting 'text' column to '{output_file}'...")
            # Loop through each record in the dataset split
            for record in dataset_split:
                text_content = record.get('text')
                if text_content:
                    # Write the text content to the file, followed by a newline
                    outfile.write(text_content + '\n')

        print(f"\nSuccessfully extracted text from '{dataset_name}' to '{output_file}'")

    except FileNotFoundError:
         print(f"Error: Could not write to output file '{output_file}'. Please check permissions.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("\nPlease ensure you have an internet connection and have installed the required library with: pip install datasets")

if __name__ == '__main__':
    # Set up an argument parser for command-line use
    parser = argparse.ArgumentParser(
        description="A script to extract the 'text' column from a Hugging Face dataset."
    )
    # The main argument is now the dataset name instead of a file path
    parser.add_argument(
        "dataset_name",
        help="The name of the dataset on Hugging Face (e.g., 'm-newhauser/senator-tweets')."
    )
    # Optional argument for the output file path
    parser.add_argument(
        "-o", "--output_file",
        default="extracted_text.txt",
        help="The name for the output text file (default: extracted_text.txt)."
    )

    args = parser.parse_args()

    # Run the extraction function with the provided arguments
    extract_text_from_hf(args.dataset_name, args.output_file)

