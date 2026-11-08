"""
oro_g2p.tokenizer
~~~~~~~~~~~~~~~~~
Linguistic tokenizer for Standard Yoruba.
Encodes:
  - Digraph preservation ('gb', 'sh' -> 'ṣ')
  - Strict tone attachment to Tone-Bearing Units (TBUs)
  - Nasal vowel classification (an, ẹn, in, ọn, un)
  - Post-nasal vowel assimilation harmony ('m', 'n' -> 'ọmọ', 'inú', 'mí')
  - Dynamic homorganic chameleon syllabic nasals ('ń', 'ǹ', 'm')
"""

from typing import Optional
from .normalizer import (
    normalize_yoruba,
    ALL_TONES,
    HIGH_TONE,
    LOW_TONE,
    MACRON_TONE,
    RISING_TONE,
    FALLING_TONE,
)

VOWELS = {"a", "e", "ẹ", "i", "o", "ọ", "u"}
NASAL_MAP = {
    "a": "an",
    "ẹ": "ẹn",
    "i": "in",
    "ọ": "ọn",
    "u": "un",
    "o": "ọn",
    "e": "ẹn",
}
CONSONANTS = {
    "b", "d", "f", "g", "h", "j", "k", "l",
    "m", "n", "p", "r", "s", "ṣ", "t", "w", "y"
}


def get_next_consonant(pieces: list[str], start_idx: int) -> Optional[str]:
    """
    Peeks ahead past spaces and stacked syllabic nasals (e.g. 'nǹkan')
    to determine the next consonant environment for homorganic nasal assimilation.
    """
    j = start_idx
    while j < len(pieces):
        if pieces[j] == " ":
            j += 1
            continue
        # Skip past a stacked syllabic nasal (e.g. first 'n' peeking past 'ǹ' to 'k' in 'nǹkan')
        if pieces[j] in ("n", "m") and (j + 1 < len(pieces)) and pieces[j + 1] in ALL_TONES:
            j += 2
            continue
        break
    if j < len(pieces):
        if pieces[j] == "g" and (j + 1 < len(pieces)) and pieces[j + 1] == "b":
            return "gb"
        if pieces[j] == "s" and (j + 1 < len(pieces)) and pieces[j + 1] == "h":
            return "ṣ"
        if pieces[j] in CONSONANTS:
            return pieces[j]
    return None


def resolve_syllabic_nasal(next_cons: Optional[str]) -> str:
    """
    Applies homorganic place assimilation to syllabic nasals:
      - Labials (b, f, m)                     -> 'm' (e.g. ń bọ̀ -> ḿ bọ̀)
      - Alveolars (t, d, s, ṣ, j, n)          -> 'n' (e.g. ń ṣe -> ń ṣe)
      - Velars, liquids, glides (k, g, gb, p, w, h, y, l, r) or pause -> 'ŋ' (e.g. Mo ń lọ -> Mo ŋ́ lọ)
    """
    if next_cons in {"b", "f", "m"}:
        return "m"
    elif next_cons in {"t", "d", "s", "ṣ", "j", "n"}:
        return "n"
    else:
        return "ŋ"


APOSTROPHES = {"'", "’", "‘", "`"}

def tokenize_graphemes(text: str) -> list[tuple[str, Optional[str]]]:
    """
    Tokenizes Yoruba text into a list of (grapheme, tone) tuples.

    Returns:
      List of tuples: (grapheme_string, tone_symbol)
      Where tone_symbol is 'H' (High), 'M' (Mid), 'L' (Low), or None for non-TBUs.
    """
    pieces = normalize_yoruba(text)
    tokens: list[tuple[str, Optional[str]]] = []
    i = 0
    n = len(pieces)

    while i < n:
        ch = pieces[i]

        # 0. Standalone or stray apostrophes: skip cleanly so they never pollute IPA tokens
        if ch in APOSTROPHES:
            i += 1
            continue

        # 1. Digraph checks: 'gb' and informal 'sh' -> 'ṣ' (including elision e.g. gb'ọ́rọ̀)
        if ch == "g" and (i + 1 < n) and pieces[i + 1] == "b":
            tokens.append(("gb", None))
            i += 2
            if i < n and pieces[i] in APOSTROPHES:
                i += 1
            continue
        if ch == "s" and (i + 1 < n) and pieces[i + 1] == "h":
            tokens.append(("ṣ", None))
            i += 2
            if i < n and pieces[i] in APOSTROPHES:
                i += 1
            continue

        # 2. Clitic Apostrophe Contractions on any consonant (e.g. n'ílé, l'ọ́jà, k'ó, t'ó, s'ọ́dọ̀, f'ọwọ́)
        if ch in CONSONANTS and (i + 1 < n) and pieces[i + 1] in APOSTROPHES:
            next_after_apos = pieces[i + 2] if (i + 2 < n) else None
            # If followed by a vowel, it is an elision contraction
            if next_after_apos in VOWELS:
                tokens.append((ch, None))
                i += 2  # Skip consonant + apostrophe
                continue
            # If word-initial standalone nasal contraction (e.g. n' before space or consonant)
            elif ch in ("n", "m"):
                tokens.append((ch, "H"))
                i += 2
                continue

        # 3. Vowel handling (oral, nasal, contours, and post-nasal harmony)
        if ch in VOWELS:
            vowel = ch
            tones = ["M"]  # Unmarked vowels are canonical Mid tone (Ré)
            had_explicit_tone = False
            i += 1

            # Extract tone marks (including contour expansions)
            if i < n and pieces[i] in ALL_TONES:
                t_mark = pieces[i]
                had_explicit_tone = True
                if t_mark == HIGH_TONE:
                    tones = ["H"]
                elif t_mark == LOW_TONE:
                    tones = ["L"]
                elif t_mark == MACRON_TONE:
                    tones = ["M"]
                elif t_mark == RISING_TONE:
                    tones = ["L", "H"]  # Rising caron: àá
                elif t_mark == FALLING_TONE:
                    tones = ["H", "L"]  # Falling circumflex: áà
                i += 1

            # Nasal vowel checking (vowel + 'n')
            if i < n and pieces[i] == "n" and vowel in NASAL_MAP:
                next_1 = pieces[i + 1] if (i + 1 < n) else None
                next_2 = pieces[i + 2] if (i + 2 < n) else None

                # Standard nasal vowel: 'n' not followed by another vowel or tone
                if next_1 not in VOWELS and next_1 not in ALL_TONES:
                    vowel = NASAL_MAP[vowel]
                    i += 1
                # Catch dataset typos: tone mistakenly placed on word-final 'n' (e.g. ìbọǹ)
                elif (not had_explicit_tone) and (next_1 in {HIGH_TONE, LOW_TONE}) and (next_2 not in VOWELS and next_2 not in CONSONANTS):
                    vowel = NASAL_MAP[vowel]
                    tones = ["H"] if next_1 == HIGH_TONE else ["L"]
                    i += 2

            # Post-nasal harmony: oral vowels immediately following 'm'/'n' or a nasalized identical geminate vowel (e.g. náà, máa, míì)
            if vowel in {"a", "ẹ", "i", "ọ", "u"} and tokens:
                prev_grapheme = tokens[-1][0]
                if prev_grapheme in {"m", "n"} or prev_grapheme == NASAL_MAP[vowel]:
                    vowel = NASAL_MAP[vowel]

            for t in tones:
                tokens.append((vowel, t))
            continue

        # 3. Syllabic nasals ('n' or 'm' as independent tone-bearing syllables)
        if ch in ("n", "m"):
            if (i + 1 < n) and pieces[i + 1] in ALL_TONES:
                t_mark = pieces[i + 1]
                tone = "H" if t_mark == HIGH_TONE else ("L" if t_mark == LOW_TONE else "M")
                next_cons = get_next_consonant(pieces, i + 2)
                nasal_sound = "m" if ch == "m" else resolve_syllabic_nasal(next_cons)
                tokens.append((nasal_sound, tone))
                i += 2
                continue
            elif (i + 1 < n) and pieces[i + 1] in CONSONANTS:
                next_cons = get_next_consonant(pieces, i + 1)
                nasal_sound = "m" if ch == "m" else resolve_syllabic_nasal(next_cons)
                tokens.append((nasal_sound, "M"))
                i += 1
                continue

        # 4. Consonants, punctuation, spaces
        tokens.append((ch, None))
        i += 1

    return tokens
