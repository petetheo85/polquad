from typing import Dict, Any

class JudgeAgent:
    """An AI agent tasked with neutralizing diverse political statemnents."""

    def __init__(self, client: "GeminiClient"):
        self.client = client
        self.system_instruction = """
        You are a neutral AI judge with no political affiliation. You are 
        capable of synthesizing multiple politicaly biased statements into
        one politically neutral statement.
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
            statement = history[0]["original_statement"]
            prompt = f"""
            Political experts have provided their opinion on the following statement:
            \n"{statement}\n"

            Your task is to review and synthesize the opinions of political experts 
            with varied political affiliations and return a single, unbiased 
            version of their statements. You have attempted this task before, 
            but your moderated statement has a bias that is above the required 
            threshold. Please make another attempt to synthesize the opinions. 
            Your new attempt must be substantially different than others in the 
            history below; DO NOt just swap synonyms. Keep your statement to a 
            similar length as the original. Do not provide any commentary or 
            reasoning. Just provide the neutral statement.

            Opinions: {opinions}
            Moderation History: {history}
            """
            
        try:
            response = self.client.generate_text(prompt, self.system_instruction)
            return response
        
        except Exception as e:
            error_message = f"[API Error] Failed to generate moderated statement. Details: {e}"
            return error_message
