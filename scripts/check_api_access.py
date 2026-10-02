"""
Quick API Connectivity and Model Accessibility Diagnostic for POLQUAD.
Tests Gemini, OpenAI, and Anthropic clients with minimal 1-token requests.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def check_keys():
    print("=" * 60)
    print("1. ENVIRONMENT VARIABLES / API KEYS")
    print("=" * 60)
    keys = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    for k, v in keys.items():
        status = f"PRESENT (ends with ...{v[-4:]})" if v else "MISSING"
        print(f"  {k:<20}: {status}")
    print()
    return keys

def test_gemini():
    print("=" * 60)
    print("2. TESTING GOOGLE GEMINI")
    print("=" * 60)
    try:
        from polquad.utils.gemini_client import GeminiClient, DEFAULT_GEMINI_MODEL
        print(f"  Configured model: {DEFAULT_GEMINI_MODEL}")
        client = GeminiClient()
        res = client.generate_text("Say 'OK'", max_output_tokens=5)
        print(f"  [SUCCESS] Response: {res.strip()}")
        return True, DEFAULT_GEMINI_MODEL
    except Exception as e:
        print(f"  [FAILED] Error: {e}")
        # Test alternative model names if 2.5-flash-lite fails
        for alt in ["gemini-2.0-flash-lite", "gemini-1.5-flash"]:
            try:
                print(f"  Trying alternative model '{alt}'...")
                from polquad.utils.gemini_client import GeminiClient
                client = GeminiClient(model=alt)
                res = client.generate_text("Say 'OK'", max_output_tokens=5)
                print(f"  [SUCCESS with '{alt}'] Response: {res.strip()}")
                return True, alt
            except Exception as e2:
                print(f"    '{alt}' failed: {e2}")
        return False, None

def test_openai():
    print("=" * 60)
    print("3. TESTING OPENAI")
    print("=" * 60)
    try:
        from polquad.utils.openai_client import ChatGPTClient, DEFAULT_GPT_MODEL
        print(f"  Configured model: {DEFAULT_GPT_MODEL}")
        client = ChatGPTClient()
        res = client.generate_text("Say 'OK'", max_output_tokens=5)
        print(f"  [SUCCESS] Response: {res.strip()}")
        return True, DEFAULT_GPT_MODEL
    except Exception as e:
        print(f"  [FAILED] Error: {e}")
        return False, None

def test_claude():
    print("=" * 60)
    print("4. TESTING ANTHROPIC CLAUDE")
    print("=" * 60)
    from anthropic import Anthropic
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [FAILED] ANTHROPIC_API_KEY is missing")
        return False, None
    client = Anthropic(api_key=api_key)
    legacy_model = "claude-3-5-haiku-20241022"
    print(f"  Testing legacy model from paper: {legacy_model}")

    try:
        res = client.messages.create(
            model=legacy_model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'OK'"}],
        )
        print(f"  [SUCCESS with {legacy_model}] Response: {res.content[0].text.strip()}")
        return True, legacy_model
    except Exception as e:
        print(f"  [LEGACY MODEL RETIRED] {legacy_model} is no longer accessible ({e})")

    # Test current available haiku
    current_model = "claude-haiku-4-5-20251001"
    print(f"  Testing current available model: {current_model}...")
    try:
        res = client.messages.create(
            model=current_model,
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'OK'"}],
        )
        print(f"  [SUCCESS with current model '{current_model}'] Response: {res.content[0].text.strip()}")
        return True, f"{current_model} (successor to retired 3-5-haiku)"
    except Exception as e:
        print(f"  Current model test failed: {e}")
        return False, None

def main():
    check_keys()
    gem_ok, gem_m = test_gemini()
    oai_ok, oai_m = test_openai()
    cla_ok, cla_m = test_claude()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Gemini   : {'READY (' + str(gem_m) + ')' if gem_ok else 'UNAVAILABLE'}")
    print(f"  OpenAI   : {'READY (' + str(oai_m) + ')' if oai_ok else 'UNAVAILABLE'}")
    print(f"  Claude   : {'READY (' + str(cla_m) + ')' if cla_ok else 'UNAVAILABLE'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
