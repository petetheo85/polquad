from .base_framework import BaseFramework
from polquad.agents.naive_agent import NaiveAgent


class NaiveFramework(BaseFramework):
    def __init__(self, config, bias_calculator, client):
        super().__init__(config, bias_calculator, client, framework_name="naive")

        # Initialize Unified Agent
        print("  ↳ [Naive] Initializing Naive Agent...", end="", flush="True")
        self.naive_agent = NaiveAgent(self.client)
        print("DONE.")

    def run_analysis(self, index, row, initial_bias_coords, initial_bias_mag):
        """Overrides iterative method from parent"""
        statement = row['text']
        

        # Make only one attempt to neutralize
        moderated_statement = self.naive_agent.neutralize(statement)
        moderated_bias, moderated_mag = self.bias_calculator.calculate_bias(moderated_statement)
        if self.verbose:
            print(f"  Moderated Statement: '{moderated_statement}'")
            print(f"  Moderated Bias Coordinates: ({moderated_bias['x']}, {moderated_bias['y']})")
            print(f"  Moderated Magnitude: {moderated_mag:.2f}")

        converged = moderated_mag <= self.bias_threshold

        # Build a simplidifed history to keep output consistent
        history = {
            0: {
                "original_statement": statement, 
                "bias": initial_bias_coords, 
                "magnitude": initial_bias_mag
            },
            1: {
                "moderated_statement": moderated_statement,
                "bias": moderated_bias,
                "magnitude": moderated_mag,
                "converged": converged
            }
        }

        return self._build_result_dict(index, row, initial_bias_coords, initial_bias_mag, history, {})
