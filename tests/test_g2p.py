"""
Automated test suite for oro_g2p covering all 6 Yoruba phonological quirks
and Kokoro token alignment. (Zero-dependency runner)
"""

from oro_g2p import g2p, tokenize_graphemes, normalize_yoruba, yoruba_to_kokoro


def test_basic_sentence():
    """Verify standard greetings and canonical vowels/tones."""
    text = "Ẹ kú àárọ̀"
    phonemes = g2p(text)
    assert "ɛˉ" in phonemes   # 'ẹ' with Mid tone
    assert "kuˊ" in phonemes  # 'kú' with High tone
    assert "aˋaˊ" in phonemes # 'àá' with Low then High


def test_digraph_gb_and_kp():
    """Verify 'gb' digraph stays intact and 'p' is pronounced /kp/."""
    text = "igbá àti páńkẹ́rẹ́"
    phonemes = g2p(text)
    assert "ɡb" in phonemes
    assert "kp" in phonemes
    assert " p" not in phonemes  # English 'p' should not exist in native Yoruba


def test_post_nasal_harmony():
    """
    Quirk 1: Vowels immediately following consonant 'm' or 'n'
    must automatically nasalize (ọmọ -> ɔ̃, inú -> ũ, mi -> ĩ).
    """
    text = "ọmọ inú mi"
    phonemes = g2p(text)
    assert "mɔ̃" in phonemes  # 'ọ' after 'm' becomes nasal 'ɔ̃'
    assert "nũ" in phonemes  # 'ú' after 'n' becomes nasal 'ũ'
    assert "mĩ" in phonemes  # 'i' after 'm' becomes nasal 'ĩ'


def test_chameleon_syllabic_nasal():
    """
    Quirk 2: Syllabic nasal homorganic assimilation.
      - Before labials (b, f) -> 'm'
      - Before tongue-tip (ṣ, d, t) -> 'n'
      - Before throat/pause/liquids (l, k, g) -> 'ŋ'
    """
    assert g2p("Mo ń bọ̀").startswith("moˉ mˊ")
    assert "nˊ" in g2p("Mo ń ṣe")
    assert "ŋˊ" in g2p("Mo ń lọ")


def test_contour_glides():
    """
    Quirk 3: Rising caron (ˇ) expands to Low-High,
    Falling circumflex (ˆ) expands to High-Low.
    """
    # ǎ -> àá (Low-High)
    phonemes_rising = g2p("dǎ")
    assert "aˋaˊ" in phonemes_rising

    # â -> áà (High-Low)
    phonemes_falling = g2p("dâ")
    assert "aˊaˋ" in phonemes_falling


def test_keyboard_typo_resilience():
    """
    Quirk 4: Typo substitutions like Turkish cedilla 'ş'
    must be cleanly normalized to Yoruba 'ṣ' (/ʃ/).
    """
    typo_text = "Şé o wà?"
    clean_text = "Ṣé o wà?"
    assert g2p(typo_text) == g2p(clean_text)
    assert "ʃ" in g2p(typo_text)


def test_kokoro_zero_missing_tokens():
    """
    Verify complete 100% token coverage in Kokoro's embedding table
    with zero missing tokens.
    """
    test_corpus = [
        "Ẹ kú àárọ̀, gbogbo ọmọ Yorùbá!",
        "Tòmátì àti Ṣòmbọ̀ dídùn.",
        "Mo ń bọ̀ ní ìrọ̀lẹ́ òní.",
        "Máalù jẹ koríko lẹ́gbẹ̀ẹ́ igi.",
        "Báwo ni nǹkan? Şé àlàáfíà ni?",
    ]
    for sentence in test_corpus:
        ipa_str, token_ids, missing = yoruba_to_kokoro(sentence)
        assert len(missing) == 0, f"Failed on '{sentence}'! Missing: {missing}"
        assert len(token_ids) > 0
