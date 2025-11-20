from typing import Dict, Any

class NaiveAgent:
    """An AI agent with no political affiliation"""

    def __init__(self, client):
        self.client = client
        self.system_instruction = """
            You are a neutral and objective AI editor. Your sole purpose is to 
            rewrite politically biased statements to be as neutral and unbiased
            as possible.
            """

    def neutralize(
            self, 
            history: Dict[int, Dict[str, Any]],
            bias_threshold: float,
            initial_call: bool = True
        ) -> str:
        
        original_statement = history[0]["original_statement"]

        if initial_call:
            prompt = f"""
            ## TASK
            Your task is to rewrite the following statement to be politically 
            neutral and unbiased. A neutral statement is one that would be located 
            at the origin (0,0) of a political compass.

            ## RULES
            - Do not change the core meaning or topic of the statement.
            - Keep your rewritten statement a similar length to the original.
            - Provide only the rewritten neutral statement with no commentary or explanation.
            
            ## STATEMENT TO NEUTRALIZE
            "{original_statement}".
            """
        else:
            last_iter_num = max(history.keys())
            last_attempt = history[last_iter_num]
            last_statement = last_attempt['moderated_statement']
            last_magnitude = last_attempt['magnitude']

            prompt = f"""
            ## CONTEXT
            The original statement was: "{original_statement}."

            ## PREVIOUS ATTEMPT (FAILED)
            Your previous attempt to neutralize this was: "{last_statement}."
            This attempt was measured and still had a bias magnitude of {last_magnitude:.2f}.
            This is above the required success threshold of {bias_threshold}.

            ## TASK
            Your task is to rewrite the following statement to be politically 
            neutral and unbiased. A neutral statement is one that would be located 
            at the origin (0,0) of a political compass.

            ## RULES
            - Do not change the core meaning or topic of the statement.
            - Keep your rewritten statement a similar length to the original.
            - Provide only the rewritten neutral statement with no commentary or explanation.
            
            ## HISTORY
            {history}
            """

        try:
            response = self.client.generate_text(prompt, self.system_instruction)
            return response
        
        except Exception as e:
            error_message = f"[API Error] Failed to generate moderated statement. Details: {e}"
            return error_message
