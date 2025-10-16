import time
import random
from functools import wraps

def retry_with_backoff(retries=3, base_delay=5, max_delay=60):
    """Retries LLM calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= retries:
                        print(f"  [Retry] Final attempt failed. Raising Exception: {e}")
                        raise

                    delay = min(base_delay * (2 ** attempts) + random.uniform, max_delay)

                    print (f"  [Retry] API error encountered: {e}. Retrying in {delay:.2f} seconds... ({attempts}/{retries})")
                    time.sleep(delay)
        return wrapper
    return decorator