from .base_framework import BaseFramework
from polquad.agents.naive_agent import NaiveAgent


class NaiveFramework(BaseFramework):
    def __init__(self, config, bias_calculator, client):
        super().__init__(config, bias_calculator, client, framework_name="naive")

        # Initialize Unified Agent
        print("  ↳ [Naive] Initializing Naïve Agent...", end="", flush="True")
        self.naive_agent = NaiveAgent(self.client)
        print("DONE.")

    def _get_moderated_statement(self, history, bias_mag, specific_data, is_first_run):
        """Generates and returns a moderated statement from the Naive agent."""
        return self.naive_agent.neutralize(history, bias_mag, is_first_run)
