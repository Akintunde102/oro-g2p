"""
Idempotent Colab / PyTorch 2.6+ Stability Patcher for StyleTTS2 / Kokoro-82M.

Applies universal fixes (works for both tonal and non-tonal languages):
  1. Modules/istftnet.py  - 1D/3D F0_curve and N shape guards in Decoder.forward
  2. Modules/slmadv.py    - 2D-safe _wav2d() wrapper to prevent batch=1 squeeze crashes
  3. models.py            - PyTorch 2.6 weights_only=False, module. prefix adapter,
                            and spectral_norm warmup on style/predictor encoders
  4. train_second.py      - weights_only=False, disable slow detect_anomaly,
                            skip NaN g_loss instead of hanging in pdb set_trace(),
                            and apply gradient clipping on bert/predictor/diffusion
"""

import argparse
import os


def patch_istftnet(sd: str) -> None:
    p = os.path.join(sd, "Modules", "istftnet.py")
    src = open(p, encoding="utf-8").read()
    old = "    def forward(self, asr, F0_curve, N, s):\n"
    if "if F0_curve.dim() == 3:" in src:
        print("✅ Modules/istftnet.py already patched")
        return
    new = old + (
        "        if F0_curve.dim() == 3: F0_curve = F0_curve.squeeze(1) if F0_curve.shape[1] == 1 else F0_curve.squeeze(-1)\n"
        "        if N.dim() == 3: N = N.squeeze(1) if N.shape[1] == 1 else N.squeeze(-1)\n"
        "        if F0_curve.dim() == 1: F0_curve = F0_curve.unsqueeze(0)\n"
        "        if N.dim() == 1: N = N.unsqueeze(0)\n"
    )
    if old not in src:
        print("⚠️ Decoder.forward not found in Modules/istftnet.py (skipped)")
        return
    open(p, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print("✅ Patched Modules/istftnet.py")


def patch_slmadv(sd: str) -> None:
    p = os.path.join(sd, "Modules", "slmadv.py")
    src = open(p, encoding="utf-8").read()
    if "def _wav2d(" in src:
        print("✅ Modules/slmadv.py already patched")
        return
    helper = "\ndef _wav2d(x):\n    return x.reshape(x.shape[0], -1)\n"
    src = helper + src
    for target in ("wav.squeeze()", "y_rec_gt.squeeze()", "y_rec_gt_pred.squeeze()", "y_rec.squeeze()"):
        src = src.replace(target, f"_wav2d({target.split('.')[0]})")
    open(p, "w", encoding="utf-8").write(src)
    print("✅ Patched Modules/slmadv.py")


def patch_models(sd: str) -> None:
    p = os.path.join(sd, "models.py")
    src = open(p, encoding="utf-8").read()
    old = """def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            model[key].load_state_dict(params[key], strict=False)
    _ = [model[key].eval() for key in model]"""
    if old not in src:
        print("✅ models.py already patched (or custom load_checkpoint)")
        return
    new = """def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
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
    open(p, "w", encoding="utf-8").write(src.replace(old, new, 1))
    print("✅ Patched models.py")


def patch_train_second(sd: str) -> None:
    p = os.path.join(sd, "train_second.py")
    src = open(p, encoding="utf-8").read()
    if "weights_only=False" not in src:
        src = src.replace(
            "import torch\n",
            "import torch\nimport functools\ntorch.load = functools.partial(torch.load, weights_only=False)\n",
            1,
        )
    src = src.replace("torch.autograd.set_detect_anomaly(True)", "torch.autograd.set_detect_anomaly(False)")
    old_nan = """            g_loss.backward()
            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()"""
    if old_nan in src:
        new_nan = """            if torch.isnan(g_loss) or torch.isinf(g_loss):
                print(f"⚠️ NaN/Inf g_loss skipped at epoch {epoch} step {i}")
                optimizer.zero_grad()
                continue
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.bert.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 5.0)"""
        src = src.replace(old_nan, new_nan, 1)
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
