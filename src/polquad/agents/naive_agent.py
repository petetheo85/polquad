

class NaiveAgent:
    """An AI agent with no political affiliation"""

    def __init__(self, client: "GeminiClient"):
        self.client = client
        self.system_instruction = """
            You are a neutral and objective AI editor. Your task is to rewrite
            political statements to be as neutral and unbaised as possible.
            """

    def neutralize(self, statement) -> str:
        prompt = f"""
            Consider the following politically biased statement:
            "{statement}".

            Now, rewrite this statement to be as unbiased as possible. Keep the 
            statement to a similar length as the original. DO NOT include your 
            reasoning or any commentary. Just provide the neutral statement. 
            """

        try:
            response = self.client.generate_text(prompt, self.system_instruction)
            return response
        
        except Exception as e:
            error_message = f"[API Error] Failed to generate moderated statement. Details: {e}"
            return error_message
