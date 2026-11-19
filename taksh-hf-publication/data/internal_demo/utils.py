import re
from typing import Optional

def sanitize_input(text: str, max_length: int = 1000) -> str:
    text = text.strip()[:max_length]
    text = re.sub(r'[<>]', '', text)
    return text

def format_currency(amount: float, currency: str = "USD") -> str:
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
    symbol = symbols.get(currency, "$")
    return f"{symbol}{amount:,.2f}"

def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator