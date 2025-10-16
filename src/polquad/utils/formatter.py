def print_main_header(text: str):
    """Prints a main header for the start/end of the script."""
    width = 80
    print(f"\n╔{'═' * (width - 2)}╗")
    print(f"║ {text.center(width - 4)} ║")
    print(f"╚{'═' * (width - 2)}╝")

def print_statement_header(index: int, statement: str):
    """Prints a header for each new statement being processed."""
    width = 80
    inner_width = width - 4
    header = f"Processing Statement #{index+1}"

    # Truncate statement if too long
    if len(statement) > inner_width - 4:
        statement = statement[:inner_width - 7] + "..."

    print(f"\n┌{'─' * (width - 2)}┐")
    print(f"│ {header.ljust(inner_width)} │")
    print(f"├{'─' * (width - 2)}┤")
    print(f'│ "{statement}"{" ".ljust(inner_width - len(statement) - 2)} │')
    print(f"└{'─' * (width - 2)}┘")

def print_framework_header(name: str, num_runs: int):
    """Prints a sub-header for each framework run."""
    text = f"Running {name.upper().replace('_', ' ')} Framework for {num_runs} trials"
    print(f"\n--- {text} ---")

def print_error(run_id: int, statement_index: int, error: Exception):
    """Prints a formatted error message."""
    print(f"  • Run {run_id} for statement {statement_index}: ERROR - {error}")

def print_summary(framework_name: str, averages: dict):
    """Prints the final averaged results for a framework."""
    print(f"  Summary for {framework_name.upper()}:")
    print(f"    - Average Bias Reduction: {averages['avg_bias_reduction']:.2f}%")
    print(f"    - Average Final Bias Magnitude: {averages['avg_final_magnitude']:.2f}")
    print(f"    - Average Iterations: {averages['avg_iterations']:.2f}")
