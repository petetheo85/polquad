from typing import Dict, Any

class JudgeAgent:
    """An AI agent tasked with synthesizing diverse political statements into neutrality."""

    def __init__(self, client):
        self.client = client
        self.system_instruction = """
        You are an apolitical AI judge with expertise in political neutralization. 
        Your sole purpose is to synthesize multiple biased statements from 
        different political quadrants into a single, politically neutral statement.
        """

    def neutralize_opinions(
            self, 
            opinions: Dict[str, Dict[str, Any]],
            history: Dict[int, Dict[str, Any]],
            bias_threshold: float,
            initial_call: bool = True
        ) -> str:

        # Print opinions received
        # print(f"Opinions received: {opinions}")

        original_statement = history[0]["original_statement"]

        if initial_call:
            prompt = f"""
            ## TASK
            Your task is to synthesize the following expert opinions into a single,
            politically neutral and unbiased rephrasing of the original statement. 
            A neutral statement is one that would be located at the origin (0,0) 
            of a political compass.

            ## CONTEXT
            The original statement being debated was: "{original_statement}
            
            ## RULES
            - Your synthesis must be a single, cohesive rewrite of the original statement.
            - Do not change the core meaning or topic of the original statement.
            - Keep your rewritten statement a similar length to the original.
            - Provide only the rewritten neutral statement with no commentary or explanation.

            ## EXPERT OPINIONS TO SYNTHESIZE
            # {opinions}
            """

        else:
            last_iter_num = max(history.keys())
            last_attempt = history[last_iter_num]
            last_statement = last_attempt['moderated_statement']
            last_magnitude = last_attempt['magnitude']

            prompt = f"""
            ## CONTEXT
            The original statement was "{original_statement}"
            Political experts provided these opinions: {opinions}

            ## PREVIOUS ATTEMPT (FAILED)
            Your previous attempt to neutralize this was: "{last_statement}."
            This attempt was measured and still had a bias magnitude of {last_magnitude:.2f}.
            This is above the required success threshold of {bias_threshold}.

            ## TASK
            Your task is to rewrite your PREVIOUS synthesis to make it more neutral.
            Your goal is to create a new statement whose coordinates are closer 
            to the neutral origin (0,0). Note the bias of the original statement
            as well as the bias of your previous attempts to continue refining.

            ## RULES
            - Do not change the core meaning or topic of the statement.
            - Keep your rewritten statement a similar length to the original.
            - Your new attempt MUST be substantially different than others in
            the history below. DO NOT just swap synonyms.
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
