"""
V6q — Natural Perturbation via a Pretrained Latent VAE

Key differences vs. V6p (TinyPatchVAE trained on mixup data):
 - Uses a PRETRAINED latent VAE (TAESD / SD-VAE) as a strong natural-image prior.
   The decoder maps any z in latent space to a plausible natural image, so
   perturbations are forced to be on the natural-image manifold.
 - Optimizes a latent residual dz on top of z_base = VAE.encode(base):
       x_adv = decode(z_base + dz)
   and clamps x_adv in pixel space to [base - eps, base + eps] for physical
   realizability.
 - Stealth losses: LPIPS-style VGG perceptual + SSIM + TV + anchor-in-latent.
 - Two-phase curriculum (content-only then full [I,hate,you,EOS]) like V6o.
 - Honest eval on all clusters x all prompts.

Expected on H100: ~2-3x faster per attack step than V6o because TAESD decode
is ~1-2ms and gradients flow through small conv stacks instead of a big DCT
basis × pixel-space MSE.

Entry points:
    run_v6q_vae_attack(ns)        -> trains attack, returns dict
    summarize_v6q_vae_attack(ns)  -> thorough eval + figure
"""

from __future__ import annotations

import math
import os
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _require(ns: Dict[str, Any], *names: str) -> None:
    missing = [n for n in names if n not in ns]
    if missing:
        raise KeyError(f"Missing notebook symbols for V6q: {missing}")


class _EarlyStop(Exception):
    """Raised internally when stop_exact / stop_asr thresholds are hit."""
    pass


def _norm01(x: torch.Tensor, max_pix: float) -> torch.Tensor:
    return (x / max_pix).clamp(0.0, 1.0)


def _psnr(a: torch.Tensor, b: torch.Tensor, max_pix: float) -> float:
    mse = F.mse_loss(_norm01(a, max_pix), _norm01(b, max_pix)).item()
    return -10.0 * math.log10(max(mse, 1e-10))


def _color_jitter(
    x: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
) -> torch.Tensor:
    """Differentiable random colour jitter applied to a (B,3,H,W) tensor in [0,1].

    Each call independently samples a shift within ±<param>.  The function is
    differentiable w.r.t. x so gradients still flow through it.
    Parameters mirror torchvision.transforms.ColorJitter conventions.
    """
    if brightness > 0:
        f = 1.0 + random.uniform(-brightness, brightness)
        x = x * f
    if contrast > 0:
        f = 1.0 + random.uniform(-contrast, contrast)
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = (x - mean) * f + mean
    if saturation > 0:
        f = 1.0 + random.uniform(-saturation, saturation)
        gray = x.mean(dim=1, keepdim=True).expand_as(x)
        x = (x - gray) * f + gray
    if hue > 0:
        # rotate hue via RGB↔HSV approximation (linear, no discontinuity)
        # use a small per-channel random shift to approximate hue rotation
        dh = random.uniform(-hue, hue)
        # shift: R+=dh, G-=dh, B stays — cheap linear hue twist in [0,1]
        x = x + x.new_tensor([dh, -dh, 0.0]).reshape(1, 3, 1, 1)
    return x.clamp(0.0, 1.0)


def _tv(x01: torch.Tensor) -> torch.Tensor:
    return (
        (x01[:, :, 1:, :] - x01[:, :, :-1, :]).abs().mean()
        + (x01[:, :, :, 1:] - x01[:, :, :, :-1]).abs().mean()
    )


def _ssim(x01: torch.Tensor, y01: torch.Tensor, win: int = 11) -> torch.Tensor:
    # simple (non-kornia) SSIM so we don't add deps; returns 1 - ssim
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x01, win, stride=1, padding=win // 2)
    mu_y = F.avg_pool2d(y01, win, stride=1, padding=win // 2)
    sx = F.avg_pool2d(x01 * x01, win, stride=1, padding=win // 2) - mu_x * mu_x
    sy = F.avg_pool2d(y01 * y01, win, stride=1, padding=win // 2) - mu_y * mu_y
    sxy = F.avg_pool2d(x01 * y01, win, stride=1, padding=win // 2) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2)
    )
    return 1.0 - ssim_map.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  VAE loader: pretrained TAESD preferred; SD-VAE fallback; TinyPatchVAE last
# ─────────────────────────────────────────────────────────────────────────────

class _PretrainedVAEWrapper:
    """Unified interface over diffusers AutoencoderTiny / AutoencoderKL.

    We expose encode(x01)->z, decode(z)->x01, with x01 in [0,1] (B,3,H,W).
    Internally we handle each model's own scaling conventions.
    """

    def __init__(self, model: nn.Module, kind: str, device, dtype=torch.float32):
        self.model = model.to(device).to(dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.kind = kind  # 'taesd' or 'sdvae'
        self.device = device
        self.dtype = dtype
        if kind == "taesd":
            self.scaling = 1.0  # taesd consumes [0,1] directly via dec/enc helpers
        else:  # sdvae
            # SD VAE expects inputs in [-1,1]; latent scale = 0.18215
            self.scaling = 0.18215

    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        x = x01.to(self.dtype)
        if self.kind == "taesd":
            return self.model.encode(x).latents
        # SD VAE
        x = x * 2.0 - 1.0
        lat = self.model.encode(x).latent_dist.mean
        return lat * self.scaling

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(self.dtype)
        if self.kind == "taesd":
            out = self.model.decode(z).sample
            return out.clamp(0.0, 1.0)
        out = self.model.decode(z / self.scaling).sample
        return ((out + 1.0) * 0.5).clamp(0.0, 1.0)


def _try_load_taesd(device, dtype=torch.float32):
    try:
        from diffusers import AutoencoderTiny
    except Exception as exc:  # noqa: BLE001
        print(f"  [VAE] diffusers.AutoencoderTiny unavailable: {exc}")
        return None
    for repo in ("madebyollin/taesd", "madebyollin/taesdxl"):
        try:
            m = AutoencoderTiny.from_pretrained(repo)
            print(f"  [VAE] Loaded TAESD from {repo}")
            return _PretrainedVAEWrapper(m, "taesd", device, dtype)
        except Exception as exc:  # noqa: BLE001
            print(f"  [VAE] {repo} failed: {exc}")
    return None


def _try_load_sdvae(device, dtype=torch.float32):
    try:
        from diffusers import AutoencoderKL
    except Exception as exc:  # noqa: BLE001
        print(f"  [VAE] diffusers.AutoencoderKL unavailable: {exc}")
        return None
    for repo in (
        "stabilityai/sd-vae-ft-mse",
        "stabilityai/sd-vae-ft-ema",
        "runwayml/stable-diffusion-v1-5",
    ):
        try:
            if "stable-diffusion" in repo:
                m = AutoencoderKL.from_pretrained(repo, subfolder="vae")
            else:
                m = AutoencoderKL.from_pretrained(repo)
            print(f"  [VAE] Loaded SD-VAE from {repo}")
            return _PretrainedVAEWrapper(m, "sdvae", device, dtype)
        except Exception as exc:  # noqa: BLE001
            print(f"  [VAE] {repo} failed: {exc}")
    return None


class _LocalPatchVAE(nn.Module):
    """Fallback: small local VAE trained on augmentations of the base image.

    Only used if no pretrained weights are available. Trained in-place at
    run time. Much stronger than V6p's mixup-trained tiny VAE because it sees
    aggressive crops/color-jitter augmentations of the actual base patch.
    """

    def __init__(self, latent_ch: int = 8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.SiLU(),
        )
        self.mu = nn.Conv2d(256, latent_ch, 3, padding=1)
        self.lv = nn.Conv2d(256, latent_ch, 3, padding=1)
        self.dec = nn.Sequential(
            nn.Conv2d(latent_ch, 256, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 3, 3, padding=1), nn.Sigmoid(),
        )

    def encode(self, x01):
        h = self.enc(x01)
        return self.mu(h), self.lv(h)

    def decode(self, z):
        return self.dec(z)


def _load_vae(device, dtype=torch.float32, base_patch01: Optional[torch.Tensor] = None):
    vae = _try_load_taesd(device, dtype)
    if vae is not None:
        return vae, "taesd"
    vae = _try_load_sdvae(device, dtype)
    if vae is not None:
        return vae, "sdvae"
    # Fallback: small local VAE trained on-the-fly for longer than V6p
    print("  [VAE] Falling back to locally-trained patch VAE (no pretrained available)")
    if base_patch01 is None:
        raise RuntimeError("No base patch provided for local VAE training")
    net = _LocalPatchVAE(latent_ch=8).to(device).to(torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    steps = 1500
    bs = 16
    H, W = base_patch01.shape[-2:]
    t0 = time.time()
    for step in range(steps):
        # aggressive augmentations of the base patch
        crop = random.uniform(0.6, 1.0)
        ch = int(H * crop)
        cw = int(W * crop)
        y0 = random.randint(0, H - ch)
        x0 = random.randint(0, W - cw)
        xb = base_patch01[:, :, y0:y0 + ch, x0:x0 + cw]
        xb = F.interpolate(xb, size=(H, W), mode="bilinear", align_corners=False)
        xb = xb.expand(bs, -1, -1, -1).clone()
        # color jitter
        xb = xb + 0.03 * torch.randn_like(xb)
        xb = xb.clamp(0, 1)
        mu, lv = net.encode(xb)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        rec = net.decode(z)
        r = F.l1_loss(rec, xb) + 0.5 * F.mse_loss(rec, xb)
        kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
        (r + 1e-3 * kl).backward()
        opt.step(); opt.zero_grad()
        if step % 300 == 0:
            print(f"    [localVAE {step:4d}/{steps}] rec={r.item():.4f} kl={kl.item():.4f}")
    print(f"  [localVAE] Trained in {time.time()-t0:.1f}s")

    # wrap in same interface
    class _Wrap:
        def __init__(self, m):
            self.model = m.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.kind = "local"

        def encode(self, x01):
            return self.model.encode(x01)[0]

        def decode(self, z):
            return self.model.decode(z).clamp(0, 1)

    return _Wrap(net), "local"


# ─────────────────────────────────────────────────────────────────────────────
#  Warm-start selection
# ─────────────────────────────────────────────────────────────────────────────

def _pick_warm_start(ns: Dict[str, Any]) -> Tuple[torch.Tensor, str, float, float]:
    for tag in ("v6o", "v6n", "v6m", "v6l", "v6k", "v6j"):
        pk, ak, ek = f"{tag}_best_patch", f"{tag}_best_asr", f"{tag}_best_exact"
        if ns.get(pk) is not None:
            return (
                ns[pk].detach().clone().float(),
                tag.upper(),
                float(ns.get(ak, 0.0)),
                float(ns.get(ek, 0.0)),
            )
    return ns["mona_base_e"].detach().clone().float(), "Base", 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Honest eval
# ─────────────────────────────────────────────────────────────────────────────

def _honest_eval(ns, patch: torch.Tensor, eval_frames: int):
    hit = ex = n = 0
    target_text = str(ns.get("v6q_target_text", "i hate you")).strip().lower()
    with torch.inference_mode():
        for cid in ns["cluster_ids"]:
            for fi in ns["cluster_frames"][cid][:eval_frames]:
                for prompt in ns["ATTACK_PROMPTS"]:
                    gen = ns["v6e_eval_gen"](patch, fi, prompt, max_new_tokens=30)
                    if any(re.search(p, gen, re.I) for p in ns["MATCH_PATTERNS"]):
                        hit += 1
                    if gen.strip().lower() == target_text:
                        ex += 1
                    n += 1
    return hit / n, ex / n, hit, ex, n


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN: run V6q attack
# ─────────────────────────────────────────────────────────────────────────────

def run_v6q_vae_attack(
    ns: Dict[str, Any],
    *,
    vae_side: int = 256,
    eps_pix: float = 0.22,
    eps_lat: float = 1.8,
    phases_cfg: Optional[List[dict]] = None,
    eval_frames: int = 3,
    eval_every: int = 75,
    log_every: int = 20,
    jitter_px: int = 3,
    aug_prob: float = 0.3,
    color_jitter_prob: float = 0.3,
    cj_brightness: float = 0.08,
    cj_contrast: float = 0.08,
    cj_saturation: float = 0.08,
    cj_hue: float = 0.02,
    prompts_per_step: int = 2,
    force_local_vae: bool = False,
    target_text: Optional[str] = None,
    stop_exact: float = 1.01,
    stop_asr: float = 1.01,
    sanity_plot_every: int = 0,
    sanity_plot_prompt: Optional[str] = None,
    sanity_plot_max_new_tokens: int = 20,
) -> Dict[str, Any]:
    # Target text: explicit kwarg > ns key > default.
    if target_text is None:
        target_text = str(ns.get("v6q_target_text", "i hate you"))
    ns["v6q_target_text"] = target_text
    _require(
        ns,
        "device", "mona_base_e", "V6E_MAX_PIX",
        "cluster_ids", "cluster_frames", "ATTACK_PROMPTS", "MATCH_PATTERNS",
        "v6e_eval_gen", "v6e_compose_view",
        "target_hijack_loss", "TARGET_IDS_J", "v6e_frz", "vgg_perceptual_loss",
    )

    device = ns["device"]
    max_pix = float(ns["V6E_MAX_PIX"])
    base_patch = ns["mona_base_e"].detach().clone().float().to(device)   # (1,3,H,W) in [0, max_pix]
    out_hw = tuple(int(v) for v in base_patch.shape[-2:])                 # (128,128) in this repo

    warm_patch, warm_tag, warm_asr, warm_exact = _pick_warm_start(ns)
    warm_patch = warm_patch.to(device)

    print("=" * 72)
    print("V6q — Pretrained-VAE natural perturbation attack")
    print("=" * 72)
    print(f"Device        : {device}")
    print(f"Warm start    : {warm_tag} (ASR={warm_asr:.0%} exact={warm_exact:.0%})")
    print(f"VAE input side: {vae_side}  |  pixel eps: {eps_pix}  |  latent eps: {eps_lat}")

    # ── Load VAE ─────────────────────────────────────────────────────────────
    base01 = _norm01(base_patch, max_pix)
    base_vae_in = F.interpolate(base01, size=(vae_side, vae_side),
                                mode="bilinear", align_corners=False)
    if force_local_vae:
        vae, vae_kind = _load_vae(device, dtype=torch.float32, base_patch01=base_vae_in)
        if vae_kind != "local":
            print("  [VAE] force_local_vae=True but pretrained loaded; ignoring override")
    else:
        vae, vae_kind = _load_vae(device, dtype=torch.float32, base_patch01=base_vae_in)
    print(f"VAE kind      : {vae_kind}")

    # ── Encode base & warm ──────────────────────────────────────────────────
    warm01 = _norm01(warm_patch, max_pix)
    warm_vae_in = F.interpolate(warm01, size=(vae_side, vae_side),
                                mode="bilinear", align_corners=False)
    with torch.no_grad():
        z_base = vae.encode(base_vae_in).float()
        z_warm = vae.encode(warm_vae_in).float()
        # sanity check decode quality
        x_rec_base = vae.decode(z_base)
        rec01 = F.interpolate(x_rec_base, size=out_hw,
                              mode="bilinear", align_corners=False)
        rec_patch = rec01 * max_pix
        rec_psnr = _psnr(rec_patch, base_patch, max_pix)
    print(f"Latent shape  : {tuple(z_base.shape)}  "
          f"({int(np.prod(z_base.shape[1:]))} params)")
    print(f"Base recon PSNR: {rec_psnr:.2f} dB")

    # ── Decode helper with pixel-space clamping for physical realizability ──
    def decode_to_patch(dz: torch.Tensor, anchor_z: torch.Tensor) -> torch.Tensor:
        z = anchor_z + dz.clamp(-eps_lat, eps_lat)
        x = vae.decode(z)  # (1,3,vae_side,vae_side) in [0,1]
        x01 = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x_pix = x01 * max_pix
        # L∞ clamp in pixel space vs base — physical realizability + anchor to base
        low = (base_patch - eps_pix).clamp(0.0, max_pix)
        high = (base_patch + eps_pix).clamp(0.0, max_pix)
        x_pix = torch.max(torch.min(x_pix, high), low)
        return x_pix.clamp(0.0, max_pix)

    # ── Initialize from warm start (or base): dz s.t. anchor + dz ≈ z_warm ──
    anchor_z = z_base.clone().detach()
    dz0 = (z_warm - z_base).clamp(-eps_lat, eps_lat)
    dz = dz0.clone().detach().requires_grad_(True)

    # tiny pixel-space residual learned on top of VAE output (picks up
    # the last few % of attack margin that the manifold misses)
    pix_residual = torch.zeros_like(base_patch, requires_grad=True)
    eps_res = 0.05  # small pixel residual budget

    # ── Phase schedule ──────────────────────────────────────────────────────
    if phases_cfg is None:
        phases_cfg = [
            dict(
                name="P1-LatentContent",
                steps=400, lr_z=0.03, lr_r=0.004,
                kappa=8.0,
                target_ids=ns["TARGET_IDS_J"][:3],
                pos_weights=[3.0, 2.0, 1.5],
                perc_w=0.25, ssim_w=0.10, tv_w=0.002, kl_w=5e-4, anchor_w=0.03,
            ),
            dict(
                name="P2-LatentFullEOS",
                steps=600, lr_z=0.02, lr_r=0.003,
                kappa=5.0,
                target_ids=ns["TARGET_IDS_J"],
                pos_weights=[3.0, 2.0, 1.5, 1.5],
                perc_w=0.35, ssim_w=0.15, tv_w=0.003, kl_w=1e-3, anchor_w=0.05,
            ),
            dict(
                name="P3-Stealth",
                steps=300, lr_z=0.008, lr_r=0.001,
                kappa=3.0,
                target_ids=ns["TARGET_IDS_J"],
                pos_weights=[3.0, 2.0, 1.5, 1.5],
                perc_w=0.70, ssim_w=0.35, tv_w=0.005, kl_w=3e-3, anchor_w=0.12,
            ),
        ]

    best = dict(asr=float(warm_asr), exact=float(warm_exact),
                patch=warm_patch.detach().clone(),
                dz=dz.detach().clone(), res=pix_residual.detach().clone(),
                phase=-1, step=-1)
    phase_results: List[dict] = []

    def evaluate(candidate: torch.Tensor, pi: int, step: int) -> bool:
        asr, ex, h, e, n = _honest_eval(ns, candidate, eval_frames)
        psnr = _psnr(candidate, base_patch, max_pix)
        improved = (ex > best["exact"]
                    or (ex == best["exact"] and asr > best["asr"]))
        tag = " *** NEW BEST" if improved else ""
        print(f"    >> ASR {h}/{n}={asr:.0%}  EXACT {e}/{n}={ex:.0%}  "
              f"PSNR={psnr:.1f}dB (best={best['exact']:.0%}){tag}")
        if improved:
            best["asr"], best["exact"] = asr, ex
            best["patch"] = candidate.detach().clone()
            best["dz"] = dz.detach().clone()
            best["res"] = pix_residual.detach().clone()
            best["phase"], best["step"] = pi, step
            best["psnr"] = psnr
            save_root = ns.get("_save_root")
            if save_root:
                torch.save(dict(
                    best_patch=best["patch"].cpu(),
                    best_dz=best["dz"].cpu(),
                    best_res=best["res"].cpu(),
                    best_asr=best["asr"], best_exact=best["exact"],
                    best_phase=best["phase"], best_step=best["step"],
                    vae_kind=vae_kind, vae_side=vae_side,
                    eps_pix=eps_pix, eps_lat=eps_lat,
                    warm_tag=warm_tag,
                ), os.path.join(save_root, "pp_v6q_vae_173.pt"))
        return improved

    # backfill warm metrics if not known
    if warm_tag != "Base" and warm_asr <= 0.0 and warm_exact <= 0.0:
        warm_asr, warm_exact, _, _, _ = _honest_eval(ns, warm_patch, eval_frames)
        best["asr"], best["exact"] = float(warm_asr), float(warm_exact)
        print(f"Backfilled warm metrics: ASR={warm_asr:.0%} exact={warm_exact:.0%}")

    # precompute warm reference for anchor loss (decoded-at-low-rank quality)
    with torch.no_grad():
        warm_ref = decode_to_patch(dz.detach() * 0.0, z_warm)  # patches from warm latent

    # ── Training loop ───────────────────────────────────────────────────────
    def _check_stop():
        if best["exact"] >= stop_exact and best["asr"] >= stop_asr:
            print(f"  [EARLY STOP] exact={best['exact']:.0%} asr={best['asr']:.0%} "
                  f"(thresholds exact>={stop_exact:.0%} asr>={stop_asr:.0%})")
            raise _EarlyStop()

    def _sanity_plot(candidate: torch.Tensor, pi: int, step: int) -> None:
        """Plot the simulated scene at each cluster's first frame + VLM output.
        Non-fatal: any error is printed and swallowed so training continues.
        """
        try:
            import matplotlib.pyplot as plt
            cids = list(ns["cluster_ids"])
            prompt = sanity_plot_prompt or ns["ATTACK_PROMPTS"][0]
            n = len(cids)
            if n == 0:
                return
            compose_image = ns.get("v6e_compose_image")
            if compose_image is None:
                print("  [sanity_plot warn] ns['v6e_compose_image'] missing "
                      "— pass it in to render the simulated scene.")
                return
            ncols = min(4, n)
            nrows = int(math.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(4 * ncols, 4 * nrows),
                                     squeeze=False)
            with torch.inference_mode():
                for i, cid in enumerate(cids):
                    fi = ns["cluster_frames"][cid][0]
                    scene = compose_image(candidate, fi)             # (1,3,H,W) in [0,1]
                    gen = ns["v6e_eval_gen"](candidate, fi, prompt,
                                             max_new_tokens=sanity_plot_max_new_tokens)
                    img = scene[0].detach().float().cpu().permute(1, 2, 0).numpy()
                    img = img.clip(0.0, 1.0)
                    ax = axes[i // ncols][i % ncols]
                    ax.imshow(img); ax.axis("off")
                    ax.set_title(f"C{cid}  fi={fi}\n'{gen.strip()[:50]}'",
                                 fontsize=9)
                for j in range(n, nrows * ncols):
                    axes[j // ncols][j % ncols].axis("off")
            fig.suptitle(f"Sanity @ phase {pi} step {step}  target='{target_text}'",
                         fontsize=10)
            fig.tight_layout()
            save_root = ns.get("_save_root")
            if save_root:
                sub = os.path.join(save_root, "sanity_plots")
                os.makedirs(sub, exist_ok=True)
                fig.savefig(os.path.join(sub, f"p{pi}_s{step:04d}.png"), dpi=90)
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"  [sanity_plot warn] {e}")

    t_attack = time.time()
    try:
     for pi, ph in enumerate(phases_cfg):
        opt = torch.optim.Adam([
            {"params": [dz], "lr": ph["lr_z"]},
            {"params": [pix_residual], "lr": ph["lr_r"]},
        ])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ph["steps"])

        print("\n" + "=" * 60)
        print(f"  {ph['name']}  steps={ph['steps']}  lr_z={ph['lr_z']}  "
              f"kappa={ph['kappa']}  tokens={len(ph['target_ids'])}")
        print("=" * 60)

        t_ph = time.time()
        ph_best_asr, ph_best_ex = best["asr"], best["exact"]

        for step in range(ph["steps"]):
            opt.zero_grad(set_to_none=True)

            do_aug = random.random() < aug_prob
            p_idxs = random.sample(
                range(len(ns["ATTACK_PROMPTS"])),
                min(prompts_per_step, len(ns["ATTACK_PROMPTS"])),
            )

            cw_running = 0.0
            n_fwd = 0
            top_tok = ""
            margins_avg = [0.0] * len(ph["target_ids"])

            # Gradient accumulation: recompute decode+patch inside each inner
            # iteration so each backward is independent (graph freed each time).
            n_total = len(ns["cluster_ids"]) * len(p_idxs)
            w = 1.0 / n_total
            do_cj = random.random() < color_jitter_prob
            for cid in ns["cluster_ids"]:
                fi = random.choice(ns["cluster_frames"][cid])
                for p_idx in p_idxs:
                    patch_vae = decode_to_patch(dz, anchor_z)
                    patch = (patch_vae + pix_residual.clamp(-eps_res, eps_res)
                             ).clamp(0.0, max_pix)
                    # Apply small colour jitter to the patch before composition
                    # to model projector/camera photometric drift.  Kept small
                    # so the clean patch isn't visually distorted.
                    if do_cj:
                        patch_in = _color_jitter(
                            patch / max_pix,
                            brightness=cj_brightness,
                            contrast=cj_contrast,
                            saturation=cj_saturation,
                            hue=cj_hue,
                        ) * max_pix
                    else:
                        patch_in = patch
                    pv = ns["v6e_compose_view"](
                        patch_in, fi, jitter_px=jitter_px, augment=do_aug
                    )
                    li, ti, mg = ns["target_hijack_loss"](
                        pv, ns["v6e_frz"][p_idx],
                        ph["target_ids"], ph["pos_weights"], ph["kappa"],
                    )
                    (w * li).backward()
                    cw_running += li.item()
                    for mi, mv in enumerate(mg):
                        margins_avg[mi] += mv
                    if not top_tok:
                        top_tok = ti
                    n_fwd += 1
                    del pv, patch_vae, patch, li

            # Stealth losses on a fresh forward pass
            patch_vae = decode_to_patch(dz, anchor_z)
            patch = (patch_vae + pix_residual.clamp(-eps_res, eps_res)
                     ).clamp(0.0, max_pix)
            perc = ns["vgg_perceptual_loss"](patch, base_patch)
            ssim_l = _ssim(_norm01(patch, max_pix), _norm01(base_patch, max_pix))
            tv_l = _tv(_norm01(patch, max_pix))
            anchor_l = F.l1_loss(_norm01(patch, max_pix), _norm01(base_patch, max_pix))
            # KL of (anchor + dz) toward N(0,I) keeps latents near the prior
            kl_l = 0.5 * (anchor_z + dz).pow(2).mean()
            stealth = (
                ph["perc_w"] * perc
                + ph["ssim_w"] * ssim_l
                + ph["tv_w"] * tv_l
                + ph["anchor_w"] * anchor_l
                + ph["kl_w"] * kl_l
            )
            stealth.backward()

            torch.nn.utils.clip_grad_norm_([dz, pix_residual], 1.0)
            opt.step()
            sched.step()

            cw_val = cw_running / max(n_fwd, 1)
            for mi in range(len(margins_avg)):
                margins_avg[mi] /= max(n_fwd, 1)

            if step % log_every == 0:
                mg_str = " ".join(f"m{i}={m:.1f}" for i, m in enumerate(margins_avg))
                print(
                    f"  [{step:4d}/{ph['steps']}] atk={cw_val:.3f} "
                    f"perc={perc.item():.3f} ssim={ssim_l.item():.3f} "
                    f"tv={tv_l.item():.3f} anc={anchor_l.item():.3f} "
                    f"top='{top_tok}' {mg_str}"
                )

            if step > 0 and step % eval_every == 0:
                with torch.inference_mode():
                    cand = decode_to_patch(dz.detach(), anchor_z) \
                           + pix_residual.detach().clamp(-eps_res, eps_res)
                    cand = cand.clamp(0.0, max_pix)
                    evaluate(cand, pi, step)
                    _check_stop()

            if sanity_plot_every > 0 and step > 0 and step % sanity_plot_every == 0:
                with torch.inference_mode():
                    cand = decode_to_patch(dz.detach(), anchor_z) \
                           + pix_residual.detach().clamp(-eps_res, eps_res)
                    cand = cand.clamp(0.0, max_pix)
                    _sanity_plot(cand, pi, step)

        # end-of-phase eval
        with torch.inference_mode():
            cand = decode_to_patch(dz.detach(), anchor_z) \
                   + pix_residual.detach().clamp(-eps_res, eps_res)
            cand = cand.clamp(0.0, max_pix)
            evaluate(cand, pi, ph["steps"])
            _check_stop()

        phase_results.append(dict(
            phase=pi, name=ph["name"],
            seconds=time.time() - t_ph,
            best_asr_at_end=best["asr"], best_exact_at_end=best["exact"],
        ))

        # carry forward best if this phase made no progress (avoid drift)
        if best["exact"] < ph_best_ex or (best["exact"] == ph_best_ex and best["asr"] < ph_best_asr):
            # restore
            with torch.no_grad():
                dz.copy_(best["dz"])
                pix_residual.copy_(best["res"])
    except _EarlyStop:
        pass

    elapsed = time.time() - t_attack
    psnr_best = _psnr(best["patch"], base_patch, max_pix)
    print("\n" + "=" * 72)
    print("V6q complete")
    print("=" * 72)
    print(f"Elapsed      : {elapsed/60:.1f} min")
    print(f"VAE          : {vae_kind}")
    print(f"Best ASR     : {best['asr']:.0%}")
    print(f"Best exact   : {best['exact']:.0%}")
    print(f"PSNR         : {psnr_best:.1f} dB")
    print(f"Delta vs {warm_tag}: ASR {best['asr']-warm_asr:+.0%}  "
          f"exact {best['exact']-warm_exact:+.0%}")

    # stash
    ns.update(dict(
        v6q_vae=vae, v6q_vae_kind=vae_kind,
        v6q_warm_tag=warm_tag,
        v6q_warm_asr=float(warm_asr), v6q_warm_exact=float(warm_exact),
        v6q_best_patch=best["patch"].detach().clone(),
        v6q_best_dz=best["dz"].detach().clone(),
        v6q_best_res=best["res"].detach().clone(),
        v6q_best_asr=float(best["asr"]),
        v6q_best_exact=float(best["exact"]),
        v6q_best_phase=int(best["phase"]),
        v6q_best_step=int(best["step"]),
        v6q_phase_results=phase_results,
        v6q_psnr=float(psnr_best),
        v6q_eps_pix=float(eps_pix),
        v6q_eps_lat=float(eps_lat),
        v6q_vae_side=int(vae_side),
    ))

    # save final
    save_root = ns.get("_save_root")
    if save_root:
        path = os.path.join(save_root, "pp_v6q_vae_173.pt")
        torch.save(dict(
            best_patch=best["patch"].cpu(),
            best_dz=best["dz"].cpu(),
            best_res=best["res"].cpu(),
            best_asr=best["asr"], best_exact=best["exact"],
            best_phase=best["phase"], best_step=best["step"],
            vae_kind=vae_kind, vae_side=vae_side,
            eps_pix=eps_pix, eps_lat=eps_lat,
            warm_tag=warm_tag, phase_results=phase_results,
            elapsed=elapsed, psnr=psnr_best,
        ), path)
        print(f"Saved -> {path}")

    return dict(
        warm_tag=warm_tag, warm_asr=float(warm_asr), warm_exact=float(warm_exact),
        best_asr=float(best["asr"]), best_exact=float(best["exact"]),
        psnr=float(psnr_best), minutes=elapsed/60, vae_kind=vae_kind,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Summary & visualization
# ─────────────────────────────────────────────────────────────────────────────

def summarize_v6q_vae_attack(ns: Dict[str, Any]) -> Dict[str, Any]:
    _require(ns, "v6q_best_patch", "mona_base_e",
             "cluster_ids", "cluster_frames", "ATTACK_PROMPTS", "MATCH_PATTERNS",
             "v6e_eval_gen", "V6E_MAX_PIX", "device")

    device = ns["device"]
    max_pix = float(ns["V6E_MAX_PIX"])
    patch = ns["v6q_best_patch"].to(device).float()
    base_patch = ns["mona_base_e"].to(device).float()

    target_text = str(ns.get("v6q_target_text", "i hate you")).strip().lower()
    cr, hit, ex, n = {}, 0, 0, 0
    with torch.inference_mode():
        for cid in sorted(ns["cluster_ids"]):
            ch = ce = ct = 0
            gens = []
            for fi in ns["cluster_frames"][cid][:3]:
                for prompt in ns["ATTACK_PROMPTS"]:
                    gen = ns["v6e_eval_gen"](patch, fi, prompt, max_new_tokens=30)
                    h = any(re.search(p, gen, re.I) for p in ns["MATCH_PATTERNS"])
                    e = gen.strip().lower() == target_text
                    ch += int(h); ce += int(e); ct += 1
                    if len(gens) < 2:
                        gens.append(gen)
            cr[cid] = dict(hit=ch, exact=ce, total=ct, gens=gens)
            hit += ch; ex += ce; n += ct

    overall_asr = hit / n
    overall_ex = ex / n
    psnr = _psnr(patch, base_patch, max_pix)

    ns.update(dict(v6q_cluster_results=cr,
                   v6q_overall_asr=overall_asr,
                   v6q_overall_exact=overall_ex,
                   v6q_psnr=psnr))

    print("\n" + "=" * 72)
    print("V6q thorough evaluation")
    print("=" * 72)
    for cid in sorted(cr):
        r = cr[cid]
        s = r["gens"][0][:70] if r["gens"] else ""
        print(f"  C{cid}: ASR={r['hit']}/{r['total']}={r['hit']/r['total']:.0%} "
              f"exact={r['exact']}/{r['total']} sample='{s}'")
    print(f"Overall ASR   : {hit}/{n} = {overall_asr:.0%}")
    print(f"Overall exact : {ex}/{n} = {overall_ex:.0%}")
    print(f"PSNR          : {psnr:.1f} dB")
    print(f"VAE           : {ns.get('v6q_vae_kind','?')}")

    # Figure
    base_np = _norm01(base_patch, max_pix)[0].permute(1, 2, 0).cpu().numpy()
    patch_np = _norm01(patch, max_pix)[0].permute(1, 2, 0).cpu().numpy()
    delta_np = (patch - base_patch).abs().mean(dim=1)[0].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(base_np); axes[0, 0].set_title("Base"); axes[0, 0].axis("off")
    axes[0, 1].imshow(patch_np); axes[0, 1].set_title(
        f"V6q (PSNR={psnr:.1f}dB)\nASR={overall_asr:.0%} exact={overall_ex:.0%}")
    axes[0, 1].axis("off")
    im = axes[0, 2].imshow(delta_np, cmap="magma")
    axes[0, 2].set_title("|V6q - base| mean RGB"); axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # comparison
    warm_tag = ns.get("v6q_warm_tag", "warm")
    warm_asr = ns.get("v6q_warm_asr", 0.0)
    warm_ex = ns.get("v6q_warm_exact", 0.0)
    labels = [warm_tag, "V6q"]
    asrs = [warm_asr, overall_asr]
    exs = [warm_ex, overall_ex]
    x = np.arange(len(labels))
    axes[1, 0].bar(x - 0.18, [100 * v for v in asrs], 0.32, label="ASR %", color="steelblue")
    axes[1, 0].bar(x + 0.18, [100 * v for v in exs], 0.32, label="Exact %", color="coral")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylim(0, 105); axes[1, 0].legend(); axes[1, 0].set_title("Warm vs V6q")

    # per-cluster
    ax = axes[1, 1]
    cids = sorted(cr)
    vals = [cr[c]["hit"] / cr[c]["total"] * 100 for c in cids]
    colors = ["#2ca02c" if v > 50 else "#d62728" for v in vals]
    ax.bar([f"C{c}" for c in cids], vals, color=colors)
    ax.set_ylim(0, 105); ax.set_title("Per-cluster ASR"); ax.set_ylabel("%")

    # text
    ax = axes[1, 2]; ax.axis("off")
    lines = [
        f"VAE kind     : {ns.get('v6q_vae_kind','?')}",
        f"warm         : {warm_tag}  asr={warm_asr:.0%}  exact={warm_ex:.0%}",
        f"V6q          : asr={overall_asr:.0%}  exact={overall_ex:.0%}",
        f"PSNR         : {psnr:.1f} dB",
        f"delta ASR    : {overall_asr-warm_asr:+.0%}",
        f"delta exact  : {overall_ex-warm_ex:+.0%}",
        "",
        "Cluster samples:",
    ]
    for c in cids[:4]:
        lines.append(f" C{c}: {(cr[c]['gens'][0][:48] if cr[c]['gens'] else '')}")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)

    plt.tight_layout(); plt.show()

    return dict(overall_asr=float(overall_asr),
                overall_exact=float(overall_ex),
                psnr=float(psnr))
