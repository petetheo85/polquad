import csv
import argparse

def extract_text_column(input_file, output_file):
    """
    Extracts the 'text' column from a given CSV file and saves it to a text file.

    This script reads a CSV, finds the column named 'text', and writes the
    content of that column for each row into a new text file. Each entry
    is separated by a newline.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output text file where results will be saved.
    """
    try:
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:

            # Create a CSV reader object to process the file
            reader = csv.reader(infile)

            # Read the header row to find the index of the 'text' column
            header = next(reader)
            try:
                # Find the column index for 'text'
                text_column_index = header.index('text')
            except ValueError:
                # Handle cases where the 'text' column is not found
                print(f"Error: 'text' column not found in the header of {input_file}")
                print(f"Available columns are: {', '.join(header)}")
                return

            # Loop through the remaining rows in the CSV
            for row in reader:
                # Ensure the row has enough columns to prevent errors
                if len(row) > text_column_index:
                    text_content = row[text_column_index]
                    # Write the extracted text to the output file, followed by a newline
                    outfile.write(text_content + '\n')

        print(f"Successfully extracted the 'text' column from '{input_file}' to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # Set up an argument parser to make the script easy to use from the command line
    parser = argparse.ArgumentParser(
        description="A script to extract the 'text' column from a specified CSV file."
    )
    # Argument for the input file path
    parser.add_argument(
        "input_file",
        help="The path to the input CSV file."
    )
    # Optional argument for the output file path
    parser.add_argument(
        "-o", "--output_file",
        default="extracted_text.txt",
        help="The path for the output text file (default: extracted_text.txt)."
    )

    args = parser.parse_args()

    # Run the extraction function with the provided file paths
    extract_text_column(args.input_file, args.output_file)
