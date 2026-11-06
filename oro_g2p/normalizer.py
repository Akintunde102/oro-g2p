"""
oro_g2p.normalizer
~~~~~~~~~~~~~~~~~~
Unicode decomposition and diacritic normalization for Yoruba text.
Handles:
  - Standard combining underdot (\\u0323 on ẹ, ọ, ṣ)
  - Common keyboard/OCR typo substitutions (cedilla \\u0327, vertical line \\u0329)
  - Tone extraction (High, Low, Mid, Rising caron, Falling circumflex)
"""

import unicodedata

# 1. Underdot variants found in keyboards, scanned texts, and African corpora
UNDERDOT = "\u0323"         # Standard combining dot below
CEDILLA = "\u0327"          # Combining cedilla (frequently typed on Turkish/Latin keyboards as ş, ȩ)
VERT_LINE_BELOW = "\u0329"  # Combining vertical line below
ALL_UNDERDOTS = {UNDERDOT, CEDILLA, VERT_LINE_BELOW}

# 2. Tone Marks (Simple + Contour Glides)
LOW_TONE = "\u0300"         # Grave (`) -> Low tone (Dò)
HIGH_TONE = "\u0301"        # Acute (´) -> High tone (Mí)
FALLING_TONE = "\u0302"     # Circumflex (ˆ) -> Falling contour (áà)
MACRON_TONE = "\u0304"      # Macron (¯) -> Explicit Mid tone (Ré)
RISING_TONE = "\u030c"      # Caron (ˇ) -> Rising contour (àá)
ALL_TONES = {LOW_TONE, HIGH_TONE, FALLING_TONE, MACRON_TONE, RISING_TONE}


def normalize_yoruba(text: str) -> list[str]:
    """
    Decomposes Yoruba text into canonical NFD building blocks.

    Rules applied:
    1. Un-glues letters, underdots, and tone diacritics via NFD.
    2. Maps any underdot variant (cedilla \\u0327, vertical line \\u0329) on e, o, s
       into the true canonical Yoruba underdot (\\u0323).
    3. Handles out-of-order typing (e.g. tone typed before underdot).
    4. Keeps tone diacritics separated right after their base character for downstream prosody.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())

    result: list[str] = []
    for char in decomposed:
        if char in ALL_UNDERDOTS and result:
            # Check if a tone diacritic was typed BEFORE the underdot: [base, tone, underdot]
            if result[-1] in ALL_TONES and len(result) >= 2 and result[-2] in {"e", "o", "s"}:
                result[-2] = unicodedata.normalize("NFC", result[-2] + UNDERDOT)
            elif result[-1] in {"e", "o", "s"}:
                result[-1] = unicodedata.normalize("NFC", result[-1] + UNDERDOT)
        else:
            result.append(char)

    return result
