from .base_framework import BaseFramework
from polquad.agents.opinion_agent import OpinionAgent
from polquad.agents.judge_agent import JudgeAgent


class PolquadFramework(BaseFramework):

    def __init__(self, config, bias_calculator, client):
        super().__init__(config, bias_calculator, client, framework_name="full_polquad")

        # Initialize agents
        print("  [Full POLQUAD] Initializing Opinion Agents...", end="", flush=True)
        self.agents = {
            'lib_left': OpinionAgent(self.client, 'lib_left'),
            'lib_right': OpinionAgent(self.client, 'lib_right'),
            'auth_left': OpinionAgent(self.client, 'auth_left'),
            'auth_right': OpinionAgent(self.client, 'auth_right')
        }
        print("DONE.")

        print("  [Full POLQUAD] Initializing Judge Agent...", end="", flush="True")
        self.judge = JudgeAgent(self.client)
        print("DONE.")

    def _get_moderated_statement(self, history, specific_data, is_first_run):
        """Generates and returns a moderated statement from the Judge agent"""
        opinions = specific_data.get("opinions", {})
        return self.judge.neutralize_opinions(opinions, history, is_first_run)

    def _get_specific_framework_data(self, statement):
        """Generates and returns a dictionary of opinions from Opinion agents"""
        opinions = {}
        for _, agent in self.agents.items():
            opinion_text = agent.provide_opinion(statement)
            bias, mag = self.bias_calculator.calculate_bias(opinion_text)
            opinions[agent.quadrant] = {
                "opinion": opinion_text, 
                "bias": bias,
                "magnitude": mag
            }
        
        return {"opinions": opinions}
