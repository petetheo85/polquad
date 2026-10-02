"""
Independent Blinded Evaluator Agent for POLQUAD.
Evaluates candidate rewrites for Meaning Preservation and Neutrality
without knowledge of framework identity or provenance.
"""

import json
import random
from typing import Dict, Any, Optional
from google.genai import types


class EvaluatorAgent:
    """An impartial AI evaluator tasked with blinded assessment of rewrites."""

    def __init__(self, client):
        self.client = client
        self.system_instruction = """
        You are an impartial, expert computational linguist serving as an independent 
        evaluator in a blinded scientific study on text debiasing. Your task is to evaluate 
        anonymized candidate rewrites of politically charged original statements strictly on 
        two independent criteria: Meaning Preservation and Perceived Neutrality.
        You must evaluate each candidate strictly on its own merits without bias.
        """

    def evaluate_blinded_candidates(
            self,
            original_statement: str,
            candidates: Dict[str, str],
            random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Anonymizes candidate rewrites into 'Candidate A', 'Candidate B', etc.,
        queries the model for blinded 1-5 ratings, and unblinds the output.

        Args:
            original_statement: The original biased statement.
            candidates: Dict mapping framework_name -> rewritten_text.
            random_seed: Optional seed for reproducible randomization.

        Returns:
            Dict containing:
                - 'blinded_evaluations': raw scores for Candidate A/B/C
                - 'unblinded_evaluations': scores mapped to framework names
                - 'blinding_map': mapping between candidate labels and framework names
        """
        # Shuffle candidates deterministically
        framework_keys = list(candidates.keys())
        rng = random.Random(random_seed)
        shuffled_keys = framework_keys.copy()
        rng.shuffle(shuffled_keys)

        label_letters = ["A", "B", "C", "D", "E"]
        blinding_map = {}  # "Candidate A" -> "full_polquad"
        reverse_map = {}   # "full_polquad" -> "Candidate A"
        candidate_blocks = []

        for idx, fw_name in enumerate(shuffled_keys):
            label = f"Candidate {label_letters[idx]}"
            blinding_map[label] = fw_name
            reverse_map[fw_name] = label
            candidate_blocks.append(f'{label}:\n"{candidates[fw_name]}"')

        candidates_formatted = "\n\n".join(candidate_blocks)

        prompt = f"""
## ORIGINAL STATEMENT
"{original_statement}"

## ANONYMIZED CANDIDATE REWRITES
{candidates_formatted}

## EVALUATION INSTRUCTIONS
For EACH candidate rewrite above, provide independent numerical ratings from 1 to 5 for:

1. Meaning Preservation (1 to 5):
   - 5: Excellent; faithfully preserves the core topic, factual claims, and key entities without hallucination.
   - 4: Good; minor nuances or secondary details adjusted, but primary subject matter is fully intact.
   - 3: Moderate; broad topic is preserved, but specific policy details or context are omitted.
   - 2: Poor; notable drift in meaning, essential context lost, or subject altered.
   - 1: Unfaithful; completely changes the subject, contradicts the core claim, or hallucinates irrelevant text.

2. Neutrality (1 to 5):
   - 5: Highly Neutral; balanced, objective, completely stripped of partisan framing, buzzwords, and emotional charge.
   - 4: Mostly Neutral; objective tone with minimal residual slant or minor stylistic charge.
   - 3: Moderately Neutral; softens polarization but retains noticeable ideological leaning.
   - 2: Somewhat Biased; significant partisan rhetoric or one-sided framing remains.
   - 1: Strongly Biased; remains heavily polarized, aggressive, or inflammatory.

Provide your evaluation in valid JSON conforming to the requested schema.
"""

        response_schema = {
            "type": "OBJECT",
            "properties": {
                label: {
                    "type": "OBJECT",
                    "properties": {
                        "meaning_preservation": {"type": "NUMBER", "description": "Score from 1 to 5"},
                        "neutrality": {"type": "NUMBER", "description": "Score from 1 to 5"},
                        "rationale": {"type": "STRING", "description": "Brief 1-2 sentence justification"}
                    },
                    "required": ["meaning_preservation", "neutrality", "rationale"]
                }
                for label in blinding_map.keys()
            },
            "required": list(blinding_map.keys())
        }

        try:
            raw_response = self.client.client.models.generate_content(
                model=self.client.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )
            )

            if raw_response.text is None:
                raise ValueError("Model returned empty text for blinded evaluation.")

            blinded_results = json.loads(raw_response.text.strip())

            # Unblind results
            unblinded_results = {}
            for label, scores in blinded_results.items():
                fw_name = blinding_map.get(label, label)
                unblinded_results[fw_name] = {
                    "meaning_preservation": float(scores.get("meaning_preservation", 0)),
                    "neutrality": float(scores.get("neutrality", 0)),
                    "rationale": scores.get("rationale", ""),
                    "blinded_label": label
                }

            return {
                "blinded_evaluations": blinded_results,
                "unblinded_evaluations": unblinded_results,
                "blinding_map": blinding_map
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[EvaluatorAgent] Error during blinded evaluation: {e}")
            raise
