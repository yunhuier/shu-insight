"""Deterministic 63-bit, character-bigram SimHash (fits SQLite signed INTEGER)."""
import hashlib
import re
from collections import Counter


def simhash(text: str) -> int:
    compact = re.sub(r"\s+", "", text.casefold())
    tokens = Counter(compact[i:i + 2] for i in range(max(1, len(compact) - 1)))
    votes = [0] * 63
    for token, count in tokens.items():
        if not token:
            continue
        digest = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        for bit in range(63):
            votes[bit] += count if digest & (1 << bit) else -count
    return sum(1 << bit for bit, vote in enumerate(votes) if vote > 0)
