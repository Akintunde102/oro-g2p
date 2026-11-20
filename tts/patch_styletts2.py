"""
Idempotent Colab / PyTorch 2.6+ Stability Patcher for StyleTTS2 / Kokoro-82M.

Applies universal fixes (works for both tonal and non-tonal languages) across both
upstream `yl4579/StyleTTS2` and `semidark/StyleTTS2` (`kikiri-tts`):
  1. Modules/istftnet.py  - 1D/3D F0_curve and N shape guards in Decoder.forward
  2. Modules/slmadv.py    - 2D-safe _wav2d() wrapper preventing batch=1 squeeze crashes
  3. models.py            - PyTorch 2.6 weights_only=False, module. prefix adapter,
                            and spectral_norm warmup on style/predictor encoders
  4. train_second.py      - weights_only=False, disable slow detect_anomaly,
                            skip NaN g_loss instead of hanging in pdb set_trace(),
                            wrap y_rec_gt_pred in no_grad(), and clip gradients
"""

import argparse
import os
import re


def patch_istftnet(sd: str) -> None:
    p = os.path.join(sd, "Modules", "istftnet.py")
    src = open(p, encoding="utf-8").read()
    if "if F0_curve.dim() == 3:" in src:
        print("✅ Modules/istftnet.py already patched")
        return
    old = "    def forward(self, asr, F0_curve, N, s):\n"
    if old not in src:
        print("⚠️ Decoder.forward not found in Modules/istftnet.py (skipped)")
        return
    new = old + (
        "        if F0_curve.dim() == 3: F0_curve = F0_curve.squeeze(1) if F0_curve.shape[1] == 1 else F0_curve.squeeze(-1)\n"
        "        if N.dim() == 3: N = N.squeeze(1) if N.shape[1] == 1 else N.squeeze(-1)\n"
        "        if F0_curve.dim() == 1: F0_curve = F0_curve.unsqueeze(0)\n"
        "        if N.dim() == 1: N = N.unsqueeze(0)\n"
    )
    open(p, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print("✅ Patched Modules/istftnet.py")


def patch_slmadv(sd: str) -> None:
    p = os.path.join(sd, "Modules", "slmadv.py")
    src = open(p, encoding="utf-8").read()
    if "def _wav2d(" in src:
        print("✅ Modules/slmadv.py already patched")
        return
    helper = "\ndef _wav2d(x):\n    return x.reshape(x.shape[0], -1)\n"
    pattern = re.compile(
        r"\b(wav|real_GP|y_pred|y_rec_gt|y_rec_gt_pred|y_rec)(\.detach\(\))?\.squeeze\(\)"
    )
    new_src, n_subs = pattern.subn(lambda m: f"_wav2d({m.group(1)}{m.group(2) or ''})", src)
    if n_subs == 0:
        print("⚠️ No target .squeeze() calls found in Modules/slmadv.py (skipped)")
        return
    open(p, "w", encoding="utf-8").write(helper + new_src)
    print(f"✅ Patched Modules/slmadv.py ({n_subs} squeeze calls wrapped)")


def patch_models(sd: str) -> None:
    p = os.path.join(sd, "models.py")
    src = open(p, encoding="utf-8").read()
    if "weights_only=False" in src and "spectral_norm warmup" in src:
        print("✅ models.py already patched")
        return

    # Match load_checkpoint regardless of single vs double quotes (black-formatted forks)
    pattern = re.compile(
        r"def load_checkpoint\(model, optimizer, path, load_only_params=True, ignore_modules=\[\]\):\n"
        r"    state = torch\.load\(path, map_location=['\"]cpu['\"]\)\n"
        r"    params = state\[['\"]net['\"]\]\n"
        r"    for key in model:\n"
        r"        if key in params and key not in ignore_modules:\n"
        r"            print\(['\"]%s loaded['\"] % key\)\n"
        r"            model\[key\]\.load_state_dict\(params\[key\], strict=False\)\n"
        r"    _ = \[model\[key\]\.eval\(\) for key in model\]"
    )
    new_block = """def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu', weights_only=False)
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            target_keys = list(model[key].state_dict().keys())
            need_mod = len(target_keys) > 0 and target_keys[0].startswith('module.')
            sd_in = {}
            for k, v in params[key].items():
                if need_mod and not k.startswith('module.'): k = 'module.' + k
                elif not need_mod and k.startswith('module.'): k = k[7:]
                sd_in[k] = v
            model[key].load_state_dict(sd_in, strict=False)
    # spectral_norm warmup so style/predictor encoders work in eval mode
    for enc in ['style_encoder', 'predictor_encoder']:
        if enc in model:
            dev = next(model[enc].parameters()).device
            model[enc].train()
            with torch.no_grad():
                _ = model[enc](torch.randn(1, 1, 80, 100, device=dev))
    _ = [model[key].eval() for key in model]"""
    new_src, n_subs = pattern.subn(new_block, src, count=1)
    if n_subs == 0:
        print("⚠️ load_checkpoint pattern not found in models.py (skipped)")
        return
    open(p, "w", encoding="utf-8").write(new_src)
    print("✅ Patched models.py")


def patch_train_second(sd: str) -> None:
    p = os.path.join(sd, "train_second.py")
    src = open(p, encoding="utf-8").read()
    changed = False

    if "weights_only" not in src:
        src = src.replace(
            "import torch\n",
            "import torch\nimport functools\ntorch.load = functools.partial(torch.load, weights_only=False)\n",
            1,
        )
        changed = True

    if "torch.autograd.set_detect_anomaly(True)" in src:
        src = src.replace("torch.autograd.set_detect_anomaly(True)", "torch.autograd.set_detect_anomaly(False)")
        changed = True

    if "NaN/Inf g_loss skipped" not in src:
        nan_pattern = re.compile(
            r"^(?P<ind>[ \t]*)g_loss\.backward\(\)"
            r"(?:\r?\n(?P=ind)if torch\.isnan\(g_loss\):\r?\n(?P=ind)[ \t]+from IPython\.core\.debugger import set_trace\r?\n(?P=ind)[ \t]+set_trace\(\))?",
            flags=re.MULTILINE,
        )
        def _nan_repl(m: re.Match) -> str:
            ind = m.group("ind")
            return (
                f"{ind}if torch.isnan(g_loss) or torch.isinf(g_loss):\n"
                f"{ind}    print(f'⚠️ NaN/Inf g_loss skipped at epoch {{epoch}} step {{i}}')\n"
                f"{ind}    optimizer.zero_grad()\n"
                f"{ind}    continue\n"
                f"{ind}g_loss.backward()\n"
                f"{ind}torch.nn.utils.clip_grad_norm_(model.bert.parameters(), 5.0)\n"
                f"{ind}torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 5.0)\n"
                f"{ind}torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 5.0)"
            )
        src, n_nan = nan_pattern.subn(_nan_repl, src, count=1)
        if n_nan:
            changed = True

    if "with torch.no_grad():\n" not in src or "y_rec_gt_pred = model.decoder(" in src:
        dec_pat = re.compile(r"^(?P<ind>[ \t]*)y_rec_gt_pred\s*=\s*model\.decoder\((.+)\)\s*$", flags=re.MULTILINE)
        # Only wrap if the previous line isn't already with torch.no_grad():
        def _wrap_dec(m: re.Match) -> str:
            start = m.start()
            prev = src[max(0, start - 40):start]
            if "torch.no_grad()" in prev:
                return m.group(0)
            ind = m.group("ind")
            return f"{ind}with torch.no_grad():\n{ind}    y_rec_gt_pred = model.decoder({m.group(2)})"
        new_src = dec_pat.sub(_wrap_dec, src, count=1)
        if new_src != src:
            src = new_src
            changed = True

    if not changed:
        print("✅ train_second.py already patched")
        return

    open(p, "w", encoding="utf-8").write(src)
    print("✅ Patched train_second.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply universal StyleTTS2 / Kokoro training patches.")
    parser.add_argument("--styletts2_dir", default="/content/StyleTTS2", help="Path to cloned StyleTTS2 repo")
    args = parser.parse_args()

    patch_istftnet(args.styletts2_dir)
    patch_slmadv(args.styletts2_dir)
    patch_models(args.styletts2_dir)
    patch_train_second(args.styletts2_dir)
    print("🎉 All StyleTTS2 patches checked!")


if __name__ == "__main__":
    main()
