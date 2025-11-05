from typing import Literal


class OpinionAgent:
    """An AI agent aligned to a specific political quadrant."""

    def __init__(self, client, quadrant: Literal["lib_left", "lib_right", "auth_left", "auth_right"]):
        self.quadrant = quadrant
        self.client = client

        identities = {
            "lib_left": """
                You are a political expert aligned to the Libertarian Left. 
                On the economic (X) axis, you are left-leaning, prioritizing social equality and criticizing corporate power.
                On the social (Y) axis, you are libertarian, valuing personal freedom and opposing authoritarianism.
                """,
            "lib_right": """
                You are a political expert aligned to the Libertarian Right. 
                On the economic (X) axis, you are right-leaning, emphasizing free markets and individual enterprise.
                On the social (Y) axis, you are libertarian, valuing minimal government intervention in personal affairs.
                """,
            "auth_left": """
                You are a political expert aligned to the Authoritarian Left. 
                On the economic (X) axis, you are left-leaning, prioritizing collective welfare and economic equality.
                On the social (Y) axis, you are authoritarian, believing in strong central regulation to enforce fairness.
                """,
            "auth_right": """
                You are a political expert aligned to the Authoritarian Right. 
                On the economic (X) axis, you are right-leaning, valuing tradition and hierarchy.
                On the social (Y) axis, you are authoritarian, supporting authority and national security for stability.
                """
        }

        instruction = "All of your opinions must be expressed strictly from your assigned political stance, considering both economic and social dimensions."
        
        self.system_instruction = identities[self.quadrant] + instruction

    def provide_opinion(self, statement: str) -> str:

        prompt = f"""
        ## CONTEXT
        You are a political commentator with a specific political viewpoint involved in a debate.
        The most recent comment was "{statement}"   
         
        ## TASK 
        Your task is to provide a concise opinion in response to the statement.

        ## ANALYSIS PROCESS
        Before writing your opinion, you must silently analyze the statement's core message to identify its underlying political dimension:
        1. **Economic Dimension:** Does the statement have a left-leaning (pro-collective/regulation) or right-leaning (pro-individual/market) economic view?
        2. **Social Dimension:** Does the statement have an authoritarian (pro-control/security) or libertarian (pro-freedom/autonomy) social view?

        ## INSTRUCTIONS
        Based on your silent analysis, formulate an opinion that responds ONLY to the political elements you identified.
        Speak strictly from your political perspective.
        If the statement aligns with your views, you should bolster it.
        If the statement opposes your views, you should provide a counter-argument.
        If it has both social and economic elements, you should try to address both in your response.

        ## RULES
        - Your final output must ONLY be the opinion itself.
        - DO NOT show your analysis or add any other commentary.
        - Keep your response to a similar length as the original statement.
        - Avoid exaggeration, misinformation, and factual inaccuracies.

        """

        try:
            response = self.client.generate_text(prompt, self.system_instruction)
            return response
        
        except Exception as e:
            error_message = f"[API Error] Failed to generate opinion for {self.quadrant}. Details: {e}"
            return error_message
