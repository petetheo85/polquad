import os
import json
from google import genai
from google.genai import types
from typing import Optional, Dict

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "[ERROR] GOOGLE_API_KEY not found in environment. "
        "Make sure it is set in your shell or venv."
    )
GEMINI_MODEL = "gemini-2.5-flash-lite"


class GeminiClient:
    """Wrapper class for interacting with the Gemini API."""

    def __init__(self, api_key: str = GOOGLE_API_KEY, model: str = GEMINI_MODEL):

        self.api_key = api_key
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    # API call to generate text (OpinionAgent, JudgeAgentF, UnifiedAgent)
    def generate_text(
            self,
            prompt: str,
            system_instruction: Optional[str] = None,
            temperature: float = 0.7,
            max_output_tokens: int = 1024,
    ) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,

                )
            )
            # Handle errors
            if response.text is None:
                print(f"[GeminiClient] --- Generation Failure ---")
                if response.candidates and response.candidates[0].finish_reason:
                     print(f"Reason: {response.candidates[0].finish_reason.name}")
                print(f"Response: {response}")
                return "[Error] LLM returned no text. Check failure reason above."
            
            return response.text.strip()
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[GeminiClient] --- CRITICAL API ERROR ---")
            print(f"The actual underlying exception was: {e}")
            return f"[Error] API Call Failed: {e}"
        
    def generate_json(
            self,
            prompt: str,
            system_instruction: Optional[str] = None,
            temperature: float = 0.0,
    ) -> Dict[str, float]:
        
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "x": {"type": "NUMBER", "description": "The economic bias score from -10.0 (Left) to 10.0 (Right)."},
                "y": {"type": "NUMBER", "description": "The social bias score from -10.0 (Authoritarian) to 10.0 (Libertarian)."}
            },
            "required": ["x", "y"]
        }

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                )
            )

            if response.text is None:
                print(f"[GeminiClient] --- Generation Failure (JSON) ---")
                if response.candidates and response.candidates[0].finish_reason:
                     print(f"Reason: {response.candidates[0].finish_reason.name}")
                return {"x": 0.0, "y": 0.0}
            
            return json.loads(response.text.strip())
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[GeminiClient] --- CRITICAL API ERROR (JSON) ---")
            print(f"The actual underlying exception was: {e}")
            return {"x": 0.0, "y": 0.0}

# Test Gemini API call
if __name__ == "__main__":
    client = GeminiClient()
    system_instruction = "You are a cat. Your name is Neko."
    contents = "Hello there." 
    print("TESTING TEXT API: \n", client.generate_text(contents, system_instruction))

    json_system_instruction = """
        You are a political scientist. Analyze the following statement for economic
        and social bias on a scale of -10.0 to 10.0. X is economic (Left/Right) 
        and Y is social (Authoritarian/Libertarian)."""
    json_contents = "National defense spending should be doubled, and all taxes should be eliminated."
    print("\nTESTING JSON API: ", client.generate_json(json_contents, json_system_instruction))


        