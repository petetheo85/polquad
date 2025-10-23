import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path to allow config import
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent
sys.path.append(str(project_root))

def get_quadrant(x: float, y: float) -> str:
    """Determines the political quadrant based on x and y coordinates."""
    if x >= 0 and y >= 0:
        return "lib_right"
    elif x < 0 and y >= 0:
        return "lib_left"
    elif x < 0 and y < 0:
        return "auth_left"
    else: # x >= 0 and y < 0
        return "auth_right"

def analyze_mismatches():
    """
    Loads framework output, creates a confusion matrix, and flags
    each misclassified statement.
    """
    data_file_path = project_root / "results" / "outputs" / "framework_output.json"
    
    print(f"Loading data from: {data_file_path}")
    try:
        with open(data_file_path, 'r') as f:
            data = json.load(f)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Could not find the file at {data_file_path}.")
        return

    # --- Setup dictionaries for analysis ---
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    misclassified_statements = defaultdict(list)
    quadrants = ["auth_left", "auth_right", "lib_left", "lib_right"]
    total_count = 0

    for item in data:
        if 'true_label' in item and 'initial_bias_coords' in item and item['initial_bias_coords']:
            true_label = item['true_label']
            coords = item['initial_bias_coords']
            calculated_quadrant = get_quadrant(coords['x'], coords['y'])
            
            confusion_matrix[true_label][calculated_quadrant] += 1
            total_count += 1
            
            if true_label != calculated_quadrant:
                misclassified_statements[true_label].append({
                    "statement": item['original_statement'],
                    "classified_as": calculated_quadrant
                })

    # --- Print the Confusion Matrix ---
    header = f"{'TRUE LABEL':<15}" + "".join([f"{q:<15}" for q in quadrants])
    print("\n" + "="*75)
    print("                LLM-CALCULATED QUADRANT")
    print("-"*75)
    print(header)
    print("-"*75)

    for true_label in quadrants:
        row_str = f"{true_label:<15}"
        for calc_quadrant in quadrants:
            count = confusion_matrix[true_label][calc_quadrant]
            row_str += f"{count:<15}"
        print(row_str)
    
    print("="*75)
    print(f"Analyzed {total_count} statements.")

    # --- Print the Misclassified Statements ---
    print("\n" + "="*75)
    print("           ANALYSIS OF MISCLASSIFIED STATEMENTS")
    print("="*75)
    
    for quadrant in quadrants:
        if misclassified_statements[quadrant]:
            print(f"\nStatements with true label: {quadrant.upper()}")
            for misclassification in misclassified_statements[quadrant]:
                statement_text = misclassification['statement']
                classified_as = misclassification['classified_as']
                if len(statement_text) > 80:
                    statement_text = statement_text[:77] + "..."
                
                print(f"  - Classified as {classified_as}: \"{statement_text}\"")
        else:
             print(f"\n✓ All statements in {quadrant.upper()} were classified correctly.")


if __name__ == '__main__':
    analyze_mismatches()