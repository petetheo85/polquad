from typing import Dict, Any


class UnifiedAgent:
    """An AI agent with no political alignment."""

    def __init__(self, client: "GeminiClient"):
        self.client = client
        self.system_instruction = """
            You are a neutral and objective AI editor capable of considering
            biased statements from multiple political perspectives and 
            synthesizing them into a single unbiased statement.
            """

    def neutralize(
            self, 
            history: Dict[int, Dict[str, Any]],
            initial_call: bool = True
        ) -> str:
        
        statement = history[0]["original_statement"]

        if initial_call:
            prompt = f"""
                Consider the following political perspectives:
                1. Libertarian Left: value personal freedom, social equality, 
                and criticize concentrated power, corporate influence and authoritarianism.
                2. Libertarian Right: emphasize individual liberty, free markets, 
                and minimal gonvernment intervention in personal and economic affairs.
                3. Authoritarian Left: prioritize collective welfare, economic equality, 
                and believe in strong central regulation to enforce fairness and justice.
                4. Authoritarian Right: value tradition, hierarchy, national security, 
                and moral order, supporting authority as necessary for stability.

                Your task is to imagine the four viewpoints as they pertain to the 
                statement \"{statement}\". Rewrite the original statement synthesizing 
                the viewpoints to remove any political bias without changing the meaning. 
                Avoid exaggeration,  misinformation, and factual inaccuracies. Keep 
                your statement to a similar length as the original statement. Do not 
                provide any commentary or reasoning. Just provide the revised statement.
                """
        else:
            prompt = f"""
                Consider the following political perspectives:
                1. Libertarian Left: value personal freedom, social equality, 
                and criticize concentrated power, corporate influence and authoritarianism.
                2. Libertarian Right: emphasize individual liberty, free markets, 
                and minimal gonvernment intervention in personal and economic affairs.
                3. Authoritarian Left: prioritize collective welfare, economic equality, 
                and believe in strong central regulation to enforce fairness and justice.
                4. Authoritarian Right: value tradition, hierarchy, national security, 
                and moral order, supporting authority as necessary for stability.

                Your task is to imagine the four viewpoints as they pertain to the 
                statement \"{statement}\". Rewrite the original statement synthesizing 
                the viewpoints to remove any political bias without changing the meaning. 
                You have attempted this task before, 
                but your moderated statement has a bias that is above the required 
                threshold. Please make another attempt to synthesize the opinions. 
                Your new attempt must be substantially different than others in the 
                history below; DO NOt just swap synonyms.Avoid exaggeration,  
                misinformation, and factual inaccuracies. Keep your statement to 
                a similar length as the original statement. Do not provide any
                commentary or reasoning. Just provide the revised statement.

                Moderation History: {history}
                """

        try:
            response = self.client.generate_text(prompt, self.system_instruction)
            return response
        
        except Exception as e:
            error_message = f"[API Error] Failed to generate moderated statement. Details: {e}"
            return error_message
