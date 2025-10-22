import time
import random
from functools import wraps

def retry_with_backoff(retries=3, base_delay=5, max_delay=180):
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
                    error_str = str(e)

                    if "529" in error_str or "Overloaded" in error_str:
                        if attempts < retries:
                            delay = min(base_delay * (10 ** attempts) + random.uniform(0,1), max_delay)
                            print (f"  [Retry] API error encountered: {e}. Retrying in {delay:.2f} seconds... ({attempts}/{retries})")
                            time.sleep(delay)
                        else:
                            print(f"  [Retry] Final attempt failed. Raising Exception: {e}")
                            raise
                    else:
                        if attempts < retries:
                            delay = min(base_delay * (2 ** attempts) + random.uniform(0,1), max_delay)
                            print (f"  [Retry] API error encountered: {e}. Retrying in {delay:.2f} seconds... ({attempts}/{retries})")
                            time.sleep(delay)
                        else:
                            print(f"  [Retry] Final attempt failed. Raising Exception: {e}")
                            raise

        return wrapper
    return decorator