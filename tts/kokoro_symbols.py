"""
178-slot atomic Yoruba vocabulary aligned with Kokoro-82M / StyleTTS2.

Encodes Yoruba oral/nasal vowels with High (ˊ), Mid (ˉ), Low (ˋ) tones,
syllabic nasals, and labial-velar consonants (kp, ɡb) as single atomic tokens
inside Kokoro's fixed 178-slot embedding table (0..177).
"""

from typing import Dict, List

BASE_KOKORO_VOCAB: Dict[str, int] = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, '"': 11,
    "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "\u0303": 17, "ʣ": 18, "ʥ": 19,
    "ʦ": 20, "ʨ": 21, "ᵝ": 22, "ꭧ": 23, "A": 24, "I": 25, "O": 31, "Q": 33,
    "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45,
    "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54,
    "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62,
    "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70,
    "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81,
    "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99,
    "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114,
    "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125, "ɻ": 126,
    "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136,
    "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156,
    "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173,
    "ᵻ": 177,
}

ATOMIC_YORUBA_SLOTS: Dict[str, int] = {
    "aˉ": 43, "eˉ": 47, "ɛˉ": 86, "iˉ": 51, "oˉ": 57, "ɔˉ": 76, "uˉ": 63,
    "aˊ": 7, "aˋ": 26, "eˊ": 8, "eˋ": 27, "ɛˊ": 28, "ɛˋ": 29,
    "iˊ": 30, "iˋ": 32, "oˊ": 34, "oˋ": 37, "ɔˊ": 38, "ɔˋ": 40,
    "uˊ": 49, "uˋ": 73, "ãˊ": 74, "ãˉ": 79, "ãˋ": 84,
    "ɛ̃ˊ": 88, "ɛ̃ˉ": 89, "ɛ̃ˋ": 91, "ĩˊ": 93, "ĩˉ": 94, "ĩˋ": 95,
    "ɔ̃ˊ": 96, "ɔ̃ˉ": 97, "ɔ̃ˋ": 98, "ũˊ": 100, "ũˉ": 104, "ũˋ": 105,
    "mˊ": 106, "mˋ": 107, "nˊ": 108, "nˋ": 109, "ŋˊ": 117, "ŋˋ": 121,
    "kp": 122, "ɡb": 124, "dʒ": 82,
}

_KEYS = sorted(ATOMIC_YORUBA_SLOTS.keys(), key=len, reverse=True)
YORUBA_KOKORO_VOCAB: Dict[str, int] = dict(BASE_KOKORO_VOCAB)
YORUBA_KOKORO_VOCAB.update(ATOMIC_YORUBA_SLOTS)

symbols = list(YORUBA_KOKORO_VOCAB.keys())
dicts = YORUBA_KOKORO_VOCAB


def to_kokoro_tokens(text: str) -> List[int]:
    """Convert an oro_g2p IPA string into atomic Kokoro token IDs."""
    tokens: List[int] = []
    i = 0
    n = len(text)
    while i < n:
        for k in _KEYS:
            if text.startswith(k, i):
                tokens.append(YORUBA_KOKORO_VOCAB[k])
                i += len(k)
                break
        else:
            ch = text[i]
            if ch in YORUBA_KOKORO_VOCAB:
                tokens.append(YORUBA_KOKORO_VOCAB[ch])
            i += 1
    return tokens
