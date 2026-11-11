"""M1 - Text preprocessing for the Ask NIRF chatbot."""
from __future__ import annotations

import re

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30}
STOP = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "me", "show", "give", "please", "tell",
        "what", "which", "how", "do", "does", "did", "has", "have", "had", "with", "by", "from", "between", "across", "over", "at", "that",
        "their", "its", "all", "any", "can", "you", "i", "us", "about", "than", "then", "this", "these", "those", "be", "been", "it"}


def normalise(text: str) -> str:
    t = str(text).lower().strip()
    t = t.replace("–", "-").replace("—", "-").replace("’", "'")
    t = re.sub(r"\bvs\.", "vs", t)
    t = re.sub(r"'s\b", "", t).replace("'", "")          # possessives: "IIT Bombay's rank" -> "iit bombay rank"
    for w, n in _NUM_WORDS.items():
        t = re.sub(rf"\b{w}\b", str(n), t)
    t = re.sub(r"[^a-z0-9&\-\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def tokens(text: str) -> list[str]:
    return normalise(text).split()
