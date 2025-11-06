from math import sqrt
from typing import Dict
import statistics


class BiasCalculator:

    def __init__(self, client, config: Dict):
        self.client = client
        self.num_runs = config.get('bias_calc_runs', 3)
        self.baseline_bias = self.calibrate()

    def calculate_bias(self, statement: str, debias: bool = True) -> Dict[str, float]:
        """Returns the (x, y) bias coordinates for a given statement."""
        system_instruction = """
        You are a precise political analyst. Your sole job is to analyze statements for political bias and plot them on a 2D political compass.
        
        ## Axis Definitions

        ### Economic Axis (X-axis: Left to Right)
        - **Far-Left (-10.0):** Communism, collective ownership, nationalized industry, central planning.
        - **Center-Left (-5.0):** Social democracy, strong regulations, wealth redistribution, robust social safety nets, pro-union.
        - **Center (0.0):** Mixed-market economy, balanced regulation and free enterprise.
        - **Center-Right (5.0):** Free markets, privatization, deregulation, lower taxes, pro-business.
        - **Far-Right (10.0):** Laissez-faire capitalism, minimal government, abolition of taxes and regulations.

        ### Social Axis (Y-axis: Authoritarian to Libertarian)
        - **Libertarian (-10.0):** Complete individual autonomy, abolition of the state, voluntary association.
        - **Center-Libertarian (-5.0):** Emphasis on individual liberty, personal freedom, skepticism of authority, privacy rights.
        - **Center (0.0):** Balance between state authority and individual rights.
        - **Center-Authoritarian (5.0):** Valuing tradition, hierarchy, national security, moral order, strong government.
        - **Authoritarian (10.0):** Total state control, censorship, national unity, strict law and order, surveillance.
        
        """

        prompt = f"""
        ## CONTEXT
        During a discussion the following statement was made by a political expert:
        "{statement}"

        ## TASK
        Analyze the political statement and determine its precise position on the 2D political compass.

        ## ANALYSIS PROCESS
        Silently evaluate the statement's position on the **Economic (X) axis**.
        Silently evalute the statement's position on the **Social (Y) axis**.
        Determine the final 'x' and 'y' coordinates based on your analysis.

        ## RULES
        - You must provide your answer as a single JSON object.
        - The JSON object must contain only two keys: "x" and "y".
        - The values for "x" and "y" must be floats between -10.0 and 10.0.
        - DO NOT include any other text, explanation, or commentary in your response.
        """

        x_coords, y_coords = [], []

        for _ in range(self.num_runs):
            bias_coords = self.client.generate_json(
                prompt=prompt,
                system_instruction=system_instruction
            )

            if debias: 
                bias_coords["x"] -= self.baseline_bias["x"]
                bias_coords["y"] -= self.baseline_bias["y"]

            x = bias_coords["x"]
            y = bias_coords["y"]

            x_coords.append(bias_coords['x'])
            y_coords.append(bias_coords['y'])

        avg_x = statistics.mean(x_coords)
        avg_y = statistics.mean(y_coords)
        avg_coords = {'x': avg_x, 'y': avg_y}

        magnitude = sqrt(avg_x ** 2 + avg_y ** 2)

        return avg_coords, magnitude

    def calibrate(self):
        """Determines the LLM's own bias baseline using a neutral statment."""
        neutral_statement = "The sky is blue and grass is green."
        bias, mag = self.calculate_bias(neutral_statement, debias = False)
        self.baseline_bias = bias
        print(f"  ↳ [Bias Calculator] Baseline Bias Coordinates: ({bias['x']}, {bias['y']})")
        print(f"  ↳ [Bias Calculator] Baseline Bias Magnitude: {mag}")
        return bias


# Test Bias Calculator 
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.append(str(project_root))
    from polquad.utils.gemini import GeminiClient
    from config import polquad_configs

    client = GeminiClient()
    calculator = BiasCalculator(client, polquad_configs)

    test_statements = (
        "We need a strong state to maintain moral order.",
        "Everyone should be free to marry whoever they want.",
        "Corporations should be nationalized.",
        "Taxes should be abolished entirely."
    )

    for statement in test_statements:
        bias, mag = calculator.calculate_bias(statement)
        print(f"\nStatement: {statement}")
        print(f"Calculated Bias Coordinates: ({bias['x']}, {bias['y']})")
        print(f"Calculated Bias Magnitude: {mag}")
