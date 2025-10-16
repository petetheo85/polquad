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
        You are a political analyst capable of analyzing statements for economic
        and social bias on a scale of -10.0 to 10.0. 
        
        The X-axis represents economic scale (Left to Right) .
        The Y-axis represents the social scale (Authoritarian to Libertarian). 

        Coordinate System Guide:
        - Libertarian Right (Pro-Free Market, Pro-Individual Liberty): Positive X, Positive Y
        - Libertarian Left (Pro-Social Justice, Pro-Individual Liberty): Negative X, Positive Y
        - Authoritarian Left (Pro-Social Justice, Pro-State Control): Negative X, Negative Y
        - Authoritarian Right (Pro-Free Market, Pro-State Control): Positive X, Negative Y
        """

        prompt = f"""
        Analyze the following political statement and determine its position 
        on a 2D politcal compass: "{statement}".

        Provide your answer as a JSON object with float values for 'x' and 'y' 
        from -10.0 to 10.0. Do not include any other text or explanation.
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
