import os
import json
from anthropic import Anthropic
from typing import Optional, Dict
from .retry_handler import retry_with_backoff

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "[ERROR] ANTHROPIC_API_KEY not found in environment."
    )
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"


class ClaudeClient:
    """Wrapper class for interacting with the Gemini API."""

    def __init__(self, api_key: str = ANTHROPIC_API_KEY, model: str = ANTHROPIC_MODEL):
        self.api_key = api_key
        self.client = Anthropic(api_key=self.api_key)
        self.model = model

    # API call to generate text (OpinionAgent, JudgeAgentF, UnifiedAgent)
    @retry_with_backoff(retries=3, base_delay=5)
    def generate_text(
            self,
            prompt: str,
            system_instruction: Optional[str] = None,
            temperature: float = 0.7,
            max_output_tokens: int = 1024,
    ) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                temperature=temperature,
                system=system_instruction,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Handle errors
            if not response.content or response.content[0].text is None:
                print(f"[ClaudeClient] --- Generation Failure ---")
                reason = "UNKNOWN"
                if response.candidates and response.candidates[0].finish_reason:
                     reason = response.stop_reason
                return ValueError(f"[Error] LLM returned no text. Finish reason: {reason}")
            
            return response.content[0].text.strip()
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ClaudeClient] --- CRITICAL API ERROR ---")
            print(f"The actual underlying exception was: {e}")
            raise
        
    @retry_with_backoff(retries=3, base_delay=5)
    def generate_json(
            self,
            prompt: str,
            system_instruction: Optional[str] = None,
            temperature: float = 0.0,
    ) -> Dict[str, float]:

        try:
            # Use tools for JSON schema
            tools = [
                {
                    "name": "extract_bias_coordinates",
                    "description": "Extract the x and y bias coordinates from political statement analysis",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "number",
                                "description": "The economic bias score from -10.0 (Left) to 10.0 (Right)."
                            },
                            "y": {
                                "type": "number",
                                "description": "The social bias score from -10.0 (Authoritarian) to 10.0 (Libertarian)."
                            }
                        },
                        "required": ["x", "y"]
                    }
                }
            ]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=temperature,
                system=system_instruction if system_instruction else "",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                tools=tools,
                tool_choice={"type": "tool", "name": "extract_bias_coordinates"}
            )   

            if not response.content:
                print(f"[ClaudeClient] --- Generation Failure (JSON) ---")
                reason = response.stop_reason or "UNKNOWN"
                return ValueError(f"[Error] JSON generation failed. Stop reason: {reason}")
            
            # Extract tool use from response
            for block in response.content:
                if hasattr(block, 'type') and block.type == "tool_use":
                    return block.input
            
            raise ValueError("No tool use found in response")
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ClaudeClient] --- CRITICAL API ERROR (JSON) ---")
            print(f"The actual underlying exception was: {e}")
            raise

# Test Gemini API call
if __name__ == "__main__":
    # List all available models
    # client=Anthropic()
    # models = client.models.list(limit=20)
    # print(models)

    client = ClaudeClient()
    system_instruction = "You are a cat. Your name is Neko."
    contents = "Hello there." 
    print("TESTING TEXT API: \n", client.generate_text(contents, system_instruction))

    json_system_instruction = """
        You are a political scientist. Analyze the following statement for economic
        and social bias on a scale of -10.0 to 10.0. X is economic (Left/Right) 
        and Y is social (Authoritarian/Libertarian)."""
    json_contents = "National defense spending should be doubled, and all taxes should be eliminated."
    print("\nTESTING JSON API: ", client.generate_json(json_contents, json_system_instruction))
