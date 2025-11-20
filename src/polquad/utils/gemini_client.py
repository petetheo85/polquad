import os
import json
from google import genai
from google.genai import types
from typing import Optional, Dict
from .retry_handler import retry_with_backoff

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


class GeminiClient:
    """Wrapper class for interacting with the Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_GEMINI_MODEL):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "[ERROR] GOOGLE_API_KEY not found in environment or passed to client."
            )
        self.client = genai.Client(api_key=self.api_key)
        self.model = model

    # API call to generate text (OpinionAgent, JudgeAgent, UnifiedAgent)
    @retry_with_backoff(retries=3, base_delay=5)
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
                reason = "UNKNOWN"
                if response.candidates and response.candidates[0].finish_reason:
                     reason = response.candidates[0].finish_reason.name
                raise ValueError(f"[Error] LLM returned no text. Finish reason: {reason}")
            
            return response.text.strip()
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[GeminiClient] --- CRITICAL API ERROR ---")
            print(f"The actual underlying exception was: {e}")
            raise
        
    @retry_with_backoff(retries=3, base_delay=5)
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
                reason = "UNKNOWN"
                if response.candidates and response.candidates[0].finish_reason:
                     reason = response.candidates[0].finish_reason.name
                raise ValueError(f"[Error] JSON generation failed. Finish reason: {reason}")
            
            return json.loads(response.text.strip())
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[GeminiClient] --- CRITICAL API ERROR (JSON) ---")
            print(f"The actual underlying exception was: {e}")
            raise