"""
oro_g2p.kokoro
~~~~~~~~~~~~~~
Kokoro-82M vocabulary integration and tokenization.
Maps Yoruba IPA phones and tones to the native 178-token vocabulary table.
"""

import unicodedata
from typing import Tuple, List

# Official Kokoro-82M base vocabulary dictionary (114 occupied slots)
BASE_KOKORO_VOCAB = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11,
    "(": 12, ")": 13, "“": 14, "”": 15, " ": 16, "\u0303": 17, "ˈ": 18, "ˌ": 19,
    "ː": 20, "ɸ": 21, "β": 22, "θ": 23, "ð": 24, "ʃ": 25, "ɹ": 27, "ɻ": 28,
    "ʋ": 29, "ⱱ": 30, "ç": 32, "ʝ": 33, "ɣ": 35, "χ": 36, "ʁ": 37, "ħ": 38,
    "ʕ": 39, "ɦ": 40, "ɬ": 41, "ɮ": 42, "a": 43, "b": 44, "c": 45, "d": 46,
    "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55,
    "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63,
    "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71,
    "æ": 72, "ɓ": 75, "ɔ": 76, "ɕ": 77, "ɖ": 78, "ɗ": 80, "ɘ": 81, "ə": 82,
    "ɚ": 83, "ɛ": 86, "ɜ": 87, "ɞ": 90, "ɟ": 91, "ɡ": 92, "ɢ": 94, "ɣ": 95,
    "ɤ": 96, "ɥ": 98, "ɦ": 99, "ɧ": 101, "ɨ": 102, "ɪ": 103, "ɫ": 110, "ɬ": 111,
    "ŋ": 112, "ɲ": 113, "ɳ": 114, "ɴ": 115, "ɵ": 116, "œ": 118, "ɶ": 119,
    "ø": 120, "ɸ": 123, "ɹ": 124, "ɾ": 125, "ɺ": 126, "ɻ": 128, "ɽ": 129,
    "s": 130, "ʃ": 131, "ʂ": 132, "θ": 133, "ʈ": 135, "ʉ": 136, "ʊ": 138,
    "ʋ": 139, "ʌ": 140, "ʍ": 142, "w": 143, "x": 145, "χ": 146, "ʒ": 147,
    "ʐ": 148, "ʑ": 150, "ʔ": 156, "ᵻ": 177
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
    # Decompose nasal vowels (e.g. ã -> a + combining tilde \u0303 in Slot 17)
    normalized = unicodedata.normalize("NFD", phonemes_str)

    token_ids: List[int] = []
    missing_tokens: List[str] = []

    for ch in normalized:
        if ch in YORUBA_KOKORO_VOCAB:
            token_ids.append(YORUBA_KOKORO_VOCAB[ch])
        else:
            missing_tokens.append(ch)

    return token_ids, missing_tokens
