"""
Standalone Yoruba TTS inference wrapper for Kokoro-82M / StyleTTS2 (v1).

Loads weights and voicepacks from a local release directory or downloads
`Akintunde102/kokoro-yoruba-v1` from Hugging Face automatically.
"""

import functools
import os
import re
import sys
import types

import numpy as np
import soundfile as sf
import torch
import yaml

torch.load = functools.partial(torch.load, weights_only=False)
from oro_g2p import g2p


def _ensure_monotonic_align_stub() -> None:
    """Allow importing StyleTTS2 utils.py without compiling monotonic_align for inference."""
    try:
        import monotonic_align  # noqa: F401
    except ImportError:
        ma = types.ModuleType("monotonic_align")
        ma.maximum_path = lambda *a, **k: None
        ma.mask_from_lens = lambda *a, **k: None
        ma_core = types.ModuleType("monotonic_align.core")
        ma_core.maximum_path_c = lambda *a, **k: None
        sys.modules["monotonic_align"] = ma
        sys.modules["monotonic_align.core"] = ma_core


class YorubaTTS:
    """Synthesizes 24 kHz Yoruba speech using oro-g2p and Kokoro Yoruba v1 weights."""

    def __init__(
        self,
        release_dir: str | None = None,
        styletts2_dir: str = "./StyleTTS2",
        device: str | None = None,
        hf_repo_id: str = "Akintunde102/kokoro-yoruba-v1",
    ) -> None:
        if release_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            if os.path.exists(os.path.join(here, "kokoro_yoruba_gold.pth")):
                release_dir = here
            else:
                from huggingface_hub import snapshot_download

                release_dir = snapshot_download(repo_id=hf_repo_id)

        self.release_dir = release_dir
        self.sd = os.path.abspath(styletts2_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.sd):
            raise FileNotFoundError(
                f"StyleTTS2 directory not found at {self.sd}. "
                "Clone it first: git clone https://github.com/yl4579/StyleTTS2.git"
            )

        _ensure_monotonic_align_stub()

        here = os.path.dirname(os.path.abspath(__file__))
        for p in (here, self.release_dir, self.sd):
            if p not in sys.path:
                sys.path.insert(0, p)

        istft_p = os.path.join(self.sd, "Modules", "istftnet.py")
        if os.path.exists(istft_p):
            src = open(istft_p, encoding="utf-8").read()
            if "if F0_curve.dim() == 3:" not in src:
                old = "    def forward(self, asr, F0_curve, N, s):\n"
                new = old + (
                    "        if F0_curve.dim() == 3: F0_curve = F0_curve.squeeze(1) if F0_curve.shape[1] == 1 else F0_curve.squeeze(-1)\n"
                    "        if N.dim() == 3: N = N.squeeze(1) if N.shape[1] == 1 else N.squeeze(-1)\n"
                    "        if F0_curve.dim() == 1: F0_curve = F0_curve.unsqueeze(0)\n"
                    "        if N.dim() == 1: N = N.unsqueeze(0)\n"
                )
                open(istft_p, "w", encoding="utf-8").write(src.replace(old, new, 1))

        from kokoro_symbols import to_kokoro_tokens
        from models import build_model, load_ASR_models, load_F0_models
        from Modules.diffusion.sampler import ADPM2Sampler, DiffusionSampler, KarrasSchedule
        from utils import recursive_munch
        from Utils.PLBERT.util import load_plbert

        self.to_kokoro_tokens = to_kokoro_tokens
        cfg = yaml.safe_load(open(os.path.join(self.release_dir, "config_yoruba.yml"), encoding="utf-8"))
        res = lambda p: p if os.path.isabs(p) else os.path.join(self.sd, p)

        text_aligner = load_ASR_models(res(cfg["ASR_path"]), res(cfg["ASR_config"]))
        pitch_extractor = load_F0_models(res(cfg["F0_path"]))
        plbert = load_plbert(res(cfg["PLBERT_dir"]))

        self.model_params = recursive_munch(cfg["model_params"])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [self.model[k].to(self.device) for k in self.model]

        ckpt = torch.load(
            os.path.join(self.release_dir, "kokoro_yoruba_gold.pth"),
            map_location="cpu",
            weights_only=False,
        )
        for key, weights in ckpt["net"].items():
            if key in self.model:
                self.model[key].load_state_dict(weights, strict=False)
        self.defaults = ckpt.get(
            "defaults",
            {"alpha": 0.0, "beta": 0.55, "embedding_scale": 1.15, "diffusion_steps": 15},
        )

        # Warm up spectral_norm buffers on style/predictor encoders before eval()
        for enc in ["style_encoder", "predictor_encoder"]:
            self.model[enc].train()
            with torch.no_grad():
                _ = self.model[enc](torch.randn(1, 1, 80, 100, device=self.device))
        _ = [self.model[k].eval() for k in self.model]

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )
        self.voices = {}
        for name, fname in [("spk8", "voicepack_spk8_bible.pt"), ("spk2", "voicepack_spk2_waxal.pt")]:
            vp_p = os.path.join(self.release_dir, fname)
            if os.path.exists(vp_p):
                self.voices[name] = torch.load(vp_p, map_location=self.device, weights_only=False).to(self.device)

    @staticmethod
    def _length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(lengths.max(), device=lengths.device).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        return torch.gt(mask + 1, lengths.unsqueeze(1))

    def _synthesize_chunk(
        self,
        chunk: str,
        ref_s: torch.Tensor,
        speed: float,
        alpha: float,
        beta: float,
        diffusion_steps: int,
        embedding_scale: float,
    ) -> np.ndarray:
        ipa = g2p(chunk).replace("-", " ").replace("–", "—")
        tokens = [0] + self.to_kokoro_tokens(ipa) + [0]
        tokens_t = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens_t.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths).to(self.device)
            t_en = self.model.text_encoder(tokens_t, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens_t, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256), device=self.device).unsqueeze(1),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps,
            ).squeeze(1)

            s = beta * s_pred[:, 128:] + (1 - beta) * ref_s[:, 128:]
            ref = alpha * s_pred[:, :128] + (1 - alpha) * ref_s[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = torch.sigmoid(self.model.predictor.duration_proj(x)).sum(axis=-1) / speed
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(int(input_lengths.item()), int(pred_dur.sum().item()), device=self.device)
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                dur_i = int(pred_dur[i].item())
                pred_aln_trg[i, c_frame : c_frame + dur_i] = 1
                c_frame += dur_i

            en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0)
            asr = t_en @ pred_aln_trg.unsqueeze(0)
            if self.model_params.decoder.type == "hifigan":
                en_new = torch.zeros_like(en)
                en_new[:, :, 0] = en[:, :, 0]
                en_new[:, :, 1:] = en[:, :, :-1]
                en = en_new
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, :-1]
                asr = asr_new

            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

        wav = out.squeeze().detach().cpu().numpy()
        return wav[:-50] if len(wav) > 100 else wav

    def speak(
        self,
        text: str,
        voice: str = "spk8",
        out_path: str | None = None,
        speed: float = 1.0,
        comma_pause: float = 0.18,
        period_pause: float = 0.35,
        alpha: float | None = None,
        beta: float | None = None,
        diffusion_steps: int | None = None,
        embedding_scale: float | None = None,
    ) -> np.ndarray:
        """Synthesize Yoruba text (automatically chunked by clause) to 24 kHz audio."""
        alpha = self.defaults["alpha"] if alpha is None else alpha
        beta = self.defaults["beta"] if beta is None else beta
        diffusion_steps = self.defaults["diffusion_steps"] if diffusion_steps is None else diffusion_steps
        embedding_scale = self.defaults["embedding_scale"] if embedding_scale is None else embedding_scale
        ref_s = self.voices[voice]

        raw_chunks = re.split(r"(?<=[,;:!?.])\s+", text.strip())
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        if chunks and chunks[-1][-1] not in ".!?":
            chunks[-1] += "."

        audio_parts = []
        for idx, chunk in enumerate(chunks):
            wav_chunk = self._synthesize_chunk(chunk, ref_s, speed, alpha, beta, diffusion_steps, embedding_scale)
            audio_parts.append(wav_chunk)
            if idx < len(chunks) - 1:
                pause_sec = period_pause if chunk[-1] in ".!?" else comma_pause
                audio_parts.append(np.zeros(int(24000 * pause_sec), dtype=np.float32))

        wav = np.concatenate(audio_parts) if audio_parts else np.zeros(1, dtype=np.float32)
        if out_path:
            sf.write(out_path, wav, 24000)
        return wav
