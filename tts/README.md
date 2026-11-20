# Kokoro Yorùbá TTS & Universal StyleTTS2 Colab Fixes

This folder separates PyTorch / StyleTTS2 training & inference tools from the zero-dependency `oro-g2p` library.

- **Interactive Gradio Demo (Free Google Colab):** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Akintunde102/oro-g2p/blob/main/tts/Kokoro_Yoruba_Demo.ipynb)
- **Live Audio Showcase:** https://huggingface.co/spaces/Akintunde102/kokoro-yoruba-demo
- **Model Weights (`526 MB`, CC BY-SA 4.0):** https://huggingface.co/Akintunde102/kokoro-yoruba-v1

---

## 1. Quick Inference (Runs Anywhere)

```bash
git clone https://github.com/yl4579/StyleTTS2.git
pip install oro-g2p munch torchdiffeq einops einops-exts transformers librosa soundfile pyyaml huggingface_hub
```

```python
from tts.tts_yoruba import YorubaTTS

tts = YorubaTTS(styletts2_dir="./StyleTTS2")  # Auto-downloads Akintunde102/kokoro-yoruba-v1
tts.speak("Ẹ kú àárọ̀ o! Báwo ni àwọn ará ilé?", voice="spk8", out_path="greeting.wav")
```

*(Note: Pass diacritics/tone marks on Yorùbá text (`ọkọ̀` vs `ọkọ`) so `oro-g2p` assigns the exact tones.)*

---

## 2. Universal StyleTTS2 / `kikiri-tts` Colab Fixes (Any Language)

If you are adapting Kokoro-82M / StyleTTS2 to **any** language (tonal or non-tonal) on modern PyTorch 2.6+ / Google Colab, run:

```bash
git clone https://github.com/yl4579/StyleTTS2.git
python tts/patch_styletts2.py --styletts2_dir ./StyleTTS2
```

This idempotently fixes 4 common training & inference crashes:
1. **`Modules/istftnet.py`:** Guards `Decoder.forward` against 1D/3D `F0_curve` and `N` shapes.
2. **`Modules/slmadv.py`:** Replaces `.squeeze()` with `_wav2d()` so `batch_size=1` remainder batches never crash WavLM.
3. **`models.py`:** Adds `weights_only=False` (PyTorch 2.6+), `module.` DataParallel key stripping, and `.train()` warmup for `spectral_norm` buffers on `style_encoder` / `predictor_encoder`.
4. **`train_second.py`:** Adds `weights_only=False`, disables `set_detect_anomaly(True)` memory overhead, skips NaN `g_loss` batches instead of hanging Colab inside `pdb.set_trace()`, and clips gradients (`5.0`).

---

## 3. Atomic 178-Slot Token Table (`kokoro_symbols.py`)

Yorùbá vowel+tone (`aˊ`, `ɔˋ`, `ũˋ`), syllabic nasals (`nˋ`, `ŋˊ`), and digraphs (`kp`, `ɡb`) are packed into Kokoro-82M's vacant slots inside `0..177` without altering the embedding tensor shape.
