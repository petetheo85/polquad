from typing import Dict, Any

class JudgeAgent:
    """An AI agent tasked with neutralizing diverse political statemnents."""

    def __init__(self, client: "GeminiClient"):
        self.client = client
        self.system_instruction = """
        You are a neutral AI judge with no political affiliation with expertise 
        in political neutralization. You are capable of synthesizing multiple 
        politicaly biased statements into one politically neutral statement.
        """

    def neutralize_opinions(
            self, 
            opinions: Dict[str, Dict[str, Any]],
            history: Dict[int, Dict[str, Any]],
            initial_call: bool = True
        ) -> str:

        if initial_call:
            statement = history[0]["original_statement"]
            prompt = f"""
            Political experts have provide their opinion on the following statement:
            \n"{statement}\n"

            Your task is to review and synthesize the opinions of political experts 
            with varied political affiliations and return a single, unbiased 
            version of their statements. Keep your statement to a similar length 
            as the original. Do not provide any commentary or reasoning. Just 
            provide the neutral statement.

            Opinions: {opinions}
            """

        else:
            last_iter_num = max(history.keys())
            last_attempt = history[last_iter_num]
            last_statement = last_attempt['moderated_statement']
            last_magnitude = last_attempt['magnitude']
            original_statement = history[0]["original_statement"]

            prompt = f"""
            The original statement was "{original_statement}."
            Political experts provided these biased opinions: {opinions}

            Your last attempt to neutralize this was "{last_statement}."
            This attempt was measured and had a bias magnitude of {last_magnitude:.2f}
            This is still above the required threshold.

            Your goal is to rewrite your PREVIOUS attempt to move its political
            coordinates closer to the neutral origin (0, 0), further reducing 
            its bias magnitude.

            Your new attempt must be substantially different than others in the 
            history below. DO NOT just swap synonyms. Keep your statement to a 
            similar length as the original. Do not provide any commentary or 
            reasoning. Just provide the neutral statement.

            Moderation History: {history}
            """
        
        try:
            response = self.client.generate_text(prompt, self.system_instruction)
            return response
        
        except Exception as e:
            error_message = f"[API Error] Failed to generate moderated statement. Details: {e}"
            return error_message
