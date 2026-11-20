# oro-g2p 🗣️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**`oro-g2p`** is a zero-dependency, research-grade Yoruba Grapheme-to-Phoneme (G2P) engine and phonetic tokenizer designed for speech synthesis (TTS), automatic speech recognition (ASR), and forced alignment.

It provides out-of-the-box native token mapping for **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** and **StyleTTS2**.

---

## Live Kokoro Yorùbá TTS (v1 Release)

- **Interactive Gradio Demo (Free Google Colab):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akintunde102/oro-g2p/blob/main/tts/Kokoro_Yoruba_Demo.ipynb)
- **Instant Audio Showcase:** [huggingface.co/spaces/Akintunde102/kokoro-yoruba-demo](https://huggingface.co/spaces/Akintunde102/kokoro-yoruba-demo)
- **Model Weights (`526 MB`):** [huggingface.co/Akintunde102/kokoro-yoruba-v1](https://huggingface.co/Akintunde102/kokoro-yoruba-v1)
- **Inference & Universal StyleTTS2/Kokoro Colab Fixes:** [`tts/`](tts/) (`tts_yoruba.py` + `patch_styletts2.py`)

---

## Key Features

1. **Deterministic Linguistic Engine:** Pure, rule-based Yoruba phonology — zero hallucinations, reproducible output.
2. **Post-Nasal Harmony:** Automatically captures unwritten phonetic nasalization on vowels following nasal consonants `m` and `n` (*ọmọ* → `/ɔˉmɔ̃ˉ/`, *inú* → `/iˉnũˊ/`).
3. **Chameleon Syllabic Nasals:** Dynamically applies homorganic place assimilation to `ń` and `ǹ`:
   - Labials (`b`, `f`): `/m/` (*Mo ń bọ̀* → `moˉ mˊ bɔˋ`)
   - Alveolars (`d`, `t`, `s`, `ṣ`): `/n/` (*Mo ń ṣe* → `moˉ nˊ ʃeˉ`)
   - Velars & Glides (`l`, `k`, `g`): `/ŋ/` (*Mo ń lọ* → `moˉ ŋˊ lɔˉ`)
4. **Typo Resilience:** Seamlessly normalizes Turkish cedillas (`ş`), combining vertical lines (`\u0329`), and out-of-order diacritic keystrokes into canonical Yoruba underdots (`ẹ`, `ọ`, `ṣ`).
5. **Contour Glides:** Expands carons (`ǎ` → `àá` Low-High) and circumflexes (`â` → `áà` High-Low).
6. **Native Kokoro Alignment:** Maps 100% of Yoruba phonemes and tones directly into Kokoro-82M's 178-token embedding table with **0 missing tokens**.

---

## Installation

```bash
pip install oro-g2p
```

---

## Quickstart

### 1. Basic G2P Conversion
```python
from oro_g2p import g2p

# Converts Yoruba text to tone-marked IPA
phonemes = g2p("Ẹ kú àárọ̀, ọmọ Yorùbá!")
print(phonemes)
# Output: 'ɛˉ kuˊ aˋaˊɾɔˋ, ɔˉmɔ̃ˉ joˉɾuˋbaˊ!'
```

### 2. Kokoro-82M Tokenization
```python
from oro_g2p import yoruba_to_kokoro

text = "Ẹ kú àárọ̀"
ipa_str, token_ids, missing = yoruba_to_kokoro(text)

print(f"IPA:       {ipa_str}")
print(f"Token IDs: {token_ids}")
print(f"Missing:   {missing}")  # [] -> 100% coverage!
```

---

## Architecture: Kokoro-82M Token Mapping

Kokoro has a fixed embedding matrix size of `178` (`0` to `177`). While 114 slots were occupied by Western languages, 64 slots were unassigned.

`oro-g2p` resolves Yoruba without modifying any pre-trained weights or embedding tensor dimensions:
- **High Tone (Mí):** Assigned to Vacant Slot `7` (`ˊ`)
- **Mid Tone (Ré):** Assigned to Vacant Slot `8` (`ˉ`)
- **Low Tone (Dò):** Assigned to Vacant Slot `26` (`ˋ`)
- **Nasal Vowels:** Decomposed into `(vowel + \u0303)`, cleanly utilizing Kokoro's native Room `17`.

---

## Running Tests

```bash
python -m pytest tests/
```

---

## License

MIT License. Free for academic, personal, and commercial speech technology research.
