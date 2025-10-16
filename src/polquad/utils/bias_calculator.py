from math import sqrt
from typing import Dict


class BiasCalculator:

    def __init__(self, client):
        self.client = client
        self.baseline_bias = self.calibrate()

    def calculate_bias(self, statement: str, debias: bool = True) -> Dict[str, float]:
        """Returns the (x, y) bias coordinates for a given statement."""
        system_instruction = """
        You are a political analyst. Analyze the following statement for economic
        and social bias on a scale of -10.0 to 10.0. X is economic (Left/Right) 
        and Y is social (Authoritarian/Libertarian)."""

        bias = self.client.generate_json(
            prompt=statement,
            system_instruction=system_instruction
        )

        if debias: 
            bias["x"] -= self.baseline_bias["x"]
            bias["y"] -= self.baseline_bias["y"]

        x = bias["x"]
        y = bias["y"]
        magnitude = sqrt(x ** 2 + y ** 2)

        return bias, magnitude

    def calibrate(self):
        """Determines the LLM's own bias baseline using a neutral statment."""
        neutral_statement = "The sky is blue and grass is green."
        bias, mag = self.calculate_bias(neutral_statement, debias = False)
        self.baseline_bias = bias
        print(f"  [Bias Calculator] Baseline Bias Coordinates: ({bias['x']}, {bias['y']})")
        print(f"  [Bias Calculator] Bias Magnitude: {mag}")
        return bias


# Test Bias Calculator 
if __name__ == "__main__":
    from .gemini import GeminiClient
    client = GeminiClient()
    calculator = BiasCalculator(client)
    bias, mag = calculator.calculate_bias("Tax cuts for corporations always help the economy.")
    print(f"Adjusted bias result: {bias}")
    print(f"Adjusted mag result: {mag}")
