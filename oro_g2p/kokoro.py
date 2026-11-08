"""
oro_g2p.kokoro
~~~~~~~~~~~~~~
Kokoro-82M vocabulary integration and tokenization.
Maps Yoruba IPA phones and tones to the native 178-token vocabulary table.
"""

import unicodedata
from typing import Tuple, List

# Official Kokoro-82M base vocabulary dictionary (exact 114 occupied slots from hexgrad/Kokoro-82M config.json)
BASE_KOKORO_VOCAB = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11,
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
    "ᵻ": 177
}

# Assign Yoruba's 3 Tone Markers to verified vacant slots in Kokoro's embedding table
YORUBA_KOKORO_VOCAB = dict(BASE_KOKORO_VOCAB)
YORUBA_KOKORO_VOCAB["ˊ"] = 7   # High Tone (Mí)  -> Vacant Slot 7
YORUBA_KOKORO_VOCAB["ˉ"] = 8   # Mid Tone  (Ré)  -> Vacant Slot 8
YORUBA_KOKORO_VOCAB["ˋ"] = 26  # Low Tone  (Dò)  -> Vacant Slot 26


def to_kokoro_tokens(phonemes_str: str) -> Tuple[List[int], List[str]]:
    """
    Takes an IPA phoneme string, decomposes nasal characters into (vowel + \\u0303),
    and maps each symbol directly to its corresponding Kokoro token ID.

    Returns:
      (token_ids, missing_tokens)
    """
    # Normalize ASCII hyphens/en-dashes to spaces so compound words (e.g. ọmọ-ọba) never drop/fail
    cleaned_str = phonemes_str.replace("-", " ").replace("–", "—")

    # Decompose nasal vowels (e.g. ã -> a + combining tilde \u0303 in Slot 17)
    normalized = unicodedata.normalize("NFD", cleaned_str)

    token_ids: List[int] = []
    missing_tokens: List[str] = []

    for ch in normalized:
        if ch in YORUBA_KOKORO_VOCAB:
            token_ids.append(YORUBA_KOKORO_VOCAB[ch])
        else:
            missing_tokens.append(ch)

    return token_ids, missing_tokens
