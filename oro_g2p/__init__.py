"""
oro_g2p: Research-grade Yoruba Grapheme-to-Phoneme converter and Kokoro TTS frontend.
"""

from typing import Tuple, List, Optional
from .normalizer import normalize_yoruba
from .tokenizer import tokenize_graphemes
from .kokoro import to_kokoro_tokens, YORUBA_KOKORO_VOCAB

__version__ = "0.1.1"
__all__ = [
    "g2p",
    "tokenize_graphemes",
    "normalize_yoruba",
    "to_kokoro_tokens",
    "yoruba_to_kokoro",
    "YORUBA_KOKORO_VOCAB",
]

# Standard IPA mappings for Yoruba
CONSONANT_IPA = {
    "b": "b",
    "d": "d",
    "f": "f",
    "g": "ɡ",     # Single-story IPA ɡ (\u0261)
    "gb": "ɡb",   # Labial-velar voiced stop
    "h": "h",
    "j": "dʒ",    # Voiced palato-alveolar affricate
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "ŋ": "ŋ",     # Velar nasal
    "p": "kp",    # Voiceless labial-velar stop /kp/
    "r": "ɾ",     # Alveolar flap/tap
    "s": "s",
    "ṣ": "ʃ",     # Voiceless postalveolar fricative
    "t": "t",
    "w": "w",
    "y": "j",     # Palatal approximant (IPA 'j')
}

VOWEL_IPA = {
    # Oral vowels
    "a": "a",
    "e": "e",
    "ẹ": "ɛ",
    "i": "i",
    "o": "o",
    "ọ": "ɔ",
    "u": "u",
    # Nasal vowels
    "an": "ã",
    "ẹn": "ɛ̃",
    "in": "ĩ",
    "ọn": "ɔ̃",
    "un": "ũ",
}

TONE_IPA = {
    "H": "ˊ",  # High tone (Mí)
    "M": "ˉ",  # Mid tone  (Ré)
    "L": "ˋ",  # Low tone  (Dò)
}


def g2p(text: str) -> str:
    """
    Converts Yoruba text into a tone-marked IPA phoneme string.

    Example:
      >>> from oro_g2p import g2p
      >>> g2p("Ẹ kú àárọ̀")
      'ɛˉ kuˊ aˋaˊɾɔˋ'
    """
    tokens = tokenize_graphemes(text)
    phonemes: List[str] = []

    for grapheme, tone in tokens:
        if grapheme in VOWEL_IPA:
            phonemes.append(VOWEL_IPA[grapheme] + (TONE_IPA[tone] if tone else ""))
        elif grapheme in CONSONANT_IPA:
            sound = CONSONANT_IPA[grapheme]
            if tone is not None:
                sound += TONE_IPA[tone]
            phonemes.append(sound)
        else:
            phonemes.append(grapheme)

    return "".join(phonemes)


def yoruba_to_kokoro(text: str) -> Tuple[str, List[int], List[str]]:
    """
    End-to-end pipeline: Raw Yoruba text -> IPA string -> Kokoro token IDs.

    Returns:
      (ipa_string, token_ids, missing_tokens)
    """
    ipa_str = g2p(text)
    token_ids, missing = to_kokoro_tokens(ipa_str)
    return ipa_str, token_ids, missing
