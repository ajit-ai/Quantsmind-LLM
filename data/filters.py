import re
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


def word_shingles(text: str, k: int = 8) -> set:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def _shingle_hash(s: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.sha1(s.encode("utf-8")).digest()[:8], "big")


class MinHashDedup:
    def __init__(self, num_perms: int = 64, bands: int = 16, threshold: float = 0.8):
        assert num_perms % bands == 0
        import numpy as np

        self.np = np
        self.num_perms = num_perms
        self.rows = num_perms // bands
        self.threshold = threshold
        self.signatures_index: dict = {}
        rng = np.random.default_rng(42)
        self._a = rng.integers(1, 2**61, size=num_perms, dtype=np.uint64)
        self._b = rng.integers(0, 2**61, size=num_perms, dtype=np.uint64)
        self._p = np.uint64((1 << 61) - 1)

    def signature(self, text: str):
        sh = self.np.fromiter(
            (_shingle_hash(s) for s in word_shingles(text)), dtype=self.np.uint64
        )
        if sh.size == 0:
            return self.np.array([], dtype=self.np.uint64)
        return ((self._a[:, None] * sh[None, :] + self._b[:, None]) % self._p).min(axis=1)

    def add(self, text: str) -> bool:
        sig = self.signature(text)
        if sig.size == 0:
            return True
        for band in range(self.num_perms // self.rows):
            key = (band, sig[band * self.rows : (band + 1) * self.rows].tobytes())
            if key in self.signatures_index:
                if jaccard(sig, self.signatures_index[key]) >= self.threshold:
                    return False
            else:
                self.signatures_index[key] = sig
        return True


def jaccard(s1, s2) -> float:
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    return float((s1 == s2).mean())


class Decontaminator:
    def __init__(self, forbidden_texts: list, k: int = 8):
        self.k = k
        self.forbidden: set = set()
        for t in forbidden_texts:
            self.forbidden |= word_shingles(t, k)

    def is_contaminated(self, text: str) -> bool:
        return bool(word_shingles(text, self.k) & self.forbidden)
