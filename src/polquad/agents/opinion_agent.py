from typing import Literal


class OpinionAgent:
    """An AI agent aligned to a specific political quadrant."""

    def __init__(self, client: "GeminiClient", quadrant: Literal["lib_left", "lib_right", "auth_left", "auth_right"]):
        self.quadrant = quadrant
        self.client = client

        identities = {
            "lib_left": """
                You are a Libertarian Left. You value personal freedom, social 
                equality, and criticize concentrated power, corporate influence 
                and authoritarianism.
                """,
            "lib_right": """
                You are a Libertarian Right. You emphasize individual liberty, 
                free markets, and minimal gonvernment intervention in personal 
                and economic affairs.
                """,
            "auth_left": """
                You are an Authoritarian Left. You prioritize collective welfare, 
                economic equality, and believe in strong central regulation to 
                enforce fairness and justice.
                """,
            "auth_right": """
                You are authoritarian right. You value tradition, hierarchy, 
                national security, and moral order, supporting authority as 
                necessary for stability.
                """
        }
        
        self.system_instruction = identities[self.quadrant]

    def provide_opinion(self, statement: str) -> str:

        prompt = f"""
        You are a political commentator. Your task is to reply to \"{statement}\" 
        with an opinion that reflects your assigned political viewpoint. If the 
        statement aligns with your political affiliation, you can bolster it. 
        Otherwise, you should provide a counter argument from your perspective.
        Avoid exaggeration, misinformation, and factual inaccuracies. Keep your
        statement to a similar length. Do not provide any commentary or reasoning. 
        Just provide the revised statement.
        """

        try:
            response = self.client.generate_text(prompt, self.system_instruction)
            return response
        
        except Exception as e:
            error_message = f"[API Error] Failed to generate opinion for {self.quadrant}. Details: {e}"
            return error_message
