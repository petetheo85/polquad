import os
import json
from openai import OpenAI
from typing import Optional, Dict
from .retry_handler import retry_with_backoff

DEFAULT_GPT_MODEL = "gpt-4o-mini"


class ChatGPTClient:
    """Wrapper class for interacting with the OpenAI API."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_GPT_MODEL):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "[ERROR] OPENAI_API_KEY not found in environment or passed to client."
            )
        self.client = OpenAI(api_key=self.api_key)
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
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  
                temperature=temperature,
                max_tokens=max_output_tokens,
                )
            
            # Handle errors
            if response.choices[0].message.content is None:
                print(f"[ChatGPTClient] --- Generation Failure ---")
                reason = response.choices[0].finish_reason or "UNKNOWN"
                raise ValueError(f"[Error] LLM returned no text. Finish reason: {reason}")
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ChatGPTClient] --- CRITICAL API ERROR ---")
            print(f"The actual underlying exception was: {e}")
            raise
        
    @retry_with_backoff(retries=3, base_delay=5)
    def generate_json(
            self,
            prompt: str,
            system_instruction: Optional[str] = None,
            temperature: float = 0.0,
    ) -> Dict[str, float]:
        
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "BiasAnalysis",
                "schema": {
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
                    "required": ["x", "y"],
                    "additionalProperties": False
                }
            }
        }

        try:
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  
                temperature=temperature,
                response_format=json_schema,
            )
            
            # Handle errors
            if response.choices[0].message.content is None:
                print(f"[ChatGPTClient] --- Generation Failure ---")
                reason = response.choices[0].finish_reason or "UNKNOWN"
                raise ValueError(f"[Error] JSON generation failed. Finish reason: {reason}")
            
            return json.loads(response.choices[0].message.content.strip())
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n[ChatGPTClient] --- CRITICAL API ERROR (JSON) ---")
            print(f"The actual underlying exception was: {e}")
            raise
