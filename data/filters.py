import re

AADHAAR = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)")


def scrub_pii(text: str) -> str:
    text = AADHAAR.sub("[AADHAAR]", text)
    text = PAN.sub("[PAN]", text)
    text = EMAIL.sub("[EMAIL]", text)
    text = PHONE.sub("[PHONE]", text)
    return text


def quality_check(
    text: str,
    min_chars: int = 200,
    max_chars: int = 200_000,
    min_alpha_ratio: float = 0.6,
) -> tuple[bool, str]:
    if not (min_chars <= len(text) <= max_chars):
        return False, "length"
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) < min_alpha_ratio:
        return False, "alpha_ratio"
    return True, ""


class ExactDedup:
    def __init__(self):
        self._seen: set[str] = set()

    def add(self, text: str) -> bool:
        key = hashlib_key(text)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def __len__(self) -> int:
        return len(self._seen)


def hashlib_key(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
