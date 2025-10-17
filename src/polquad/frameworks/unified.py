from .base_framework import BaseFramework
from polquad.agents.unified_agent import UnifiedAgent


class UnifiedPolquadFramework(BaseFramework):
    def __init__(self, config, bias_calculator, client):
        super().__init__(config, bias_calculator, client, framework_name="unified_polquad")

        # Initialize Unified Agent
        print("  ↳ [Unified POLQUAD] Initializing Unified Agent...", end="", flush="True")
        self.unified_agent = UnifiedAgent(self.client)
        print("DONE.")

    def _get_moderated_statement(self, history, bias_mag, specific_data, is_first_run):
        """Implements the moderation logic for the Unified Agent"""
        return self.unified_agent.neutralize(history, bias_mag, is_first_run)