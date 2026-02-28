import tiktoken
from config import MODEL, TOKEN_PRICE_PER_1K

encoding = tiktoken.encoding_for_model(MODEL)

def count_tokens(messages):
    total = 0
    for msg in messages:
        if isinstance(msg.get("content"), str):
            total += len(encoding.encode(msg["content"]))
    return total

def estimate_cost(token_count):
    return (token_count / 1000) * TOKEN_PRICE_PER_1K