"""
augment_deca.py
───────────────
Generate synthetic training data from single portraits using DECA.

Fixes over the original script:
  1. Expression noise no longer accumulates — original codedict is cloned each iteration
  2. Pose varies across yaw, pitch, AND roll for richer viewpoint diversity
  3. Texture rendering (when available) instead of shape-only renders
  4. torch.amp.autocast uses the modern API
  5. Output images are properly channel-ordered for cv2
"""

import os
import copy
import cv2
import torch
import numpy as np
import sys

# ── DECA ──────────────────────────────────────────────────────────────────────
sys.path.append("DECA")
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.renderer import SRenderY

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RAW_DIR = "data/raw"          # one sub-folder per person, each with 1+ portrait
OUT_DIR = "data/train"        # augmented outputs land here
NUM_VARIATIONS = 15           # synthetic views per source image

# Augmentation ranges
YAW_RANGE = 0.15              # radians per step from centre (≈ ±60° total span)
PITCH_RANGE = 0.08            # subtle up/down tilt
ROLL_RANGE = 0.05             # subtle head roll
EXPR_NOISE_STD = 0.03         # expression perturbation strength

# ── INIT ──────────────────────────────────────────────────────────────────────
deca_cfg.model.use_tex = True  # enable texture for more realistic renders
try:
    deca = DECA(config=deca_cfg, device=DEVICE)
except Exception:
    # Fall back to shape-only if texture model isn't available
    print("[WARN] Texture model unavailable — falling back to shape-only renders.")
    deca_cfg.model.use_tex = False
    deca = DECA(config=deca_cfg, device=DEVICE)

renderer = SRenderY(224, obj_filename=deca_cfg.model.topology_path).to(DEVICE)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_and_preprocess(path: str) -> torch.Tensor | None:
    """Read an image, resize to 224×224, return a (1,3,H,W) float tensor."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # DECA expects RGB
    img = cv2.resize(img, (224, 224))
    tensor = (
        torch.tensor(img / 255.0, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )
    return tensor


def render_variation(codedict: dict, use_texture: bool) -> np.ndarray:
    """Decode a codedict and render; return an (H,W,3) uint8 BGR image."""
    with torch.no_grad():
        opdict, _ = deca.decode(codedict)

    if use_texture and "uv_texture_gt" in opdict:
        # Textured render
        rendered = renderer(
            opdict["verts"],
            opdict["trans_verts"],
            opdict["uv_texture_gt"],
        )
        out = rendered[0]
    else:
        # Shape-only fallback
        rendered = renderer.render_shape(
            opdict["verts"],
            opdict["trans_verts"],
        )
        out = rendered[0]

    out = out.cpu().numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out


# ── PER-PERSON AUGMENTATION ──────────────────────────────────────────────────
def generate_for_person(person: str) -> None:
    input_dir = os.path.join(RAW_DIR, person)
    output_dir = os.path.join(OUT_DIR, person)
    os.makedirs(output_dir, exist_ok=True)

    use_texture = deca_cfg.model.use_tex

    for img_name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, img_name)
        img_tensor = load_and_preprocess(path)
        if img_tensor is None:
            continue

        # Encode once
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                codedict_orig = deca.encode(img_tensor)

        stem = os.path.splitext(img_name)[0]

        for i in range(NUM_VARIATIONS):
            # ── Deep-copy so each variation starts from the original ──
            codedict = {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                        for k, v in codedict_orig.items()}

            # ── Pose: vary yaw, pitch, roll ──
            yaw   = (i - NUM_VARIATIONS // 2) * YAW_RANGE
            pitch = (np.random.rand() - 0.5) * 2 * PITCH_RANGE
            roll  = (np.random.rand() - 0.5) * 2 * ROLL_RANGE

            codedict["pose"][:, 3] = yaw    # yaw  (left ↔ right)
            codedict["pose"][:, 4] = pitch   # pitch (up ↔ down)
            codedict["pose"][:, 5] = roll    # roll  (tilt)

            # ── Expression: add noise to a FRESH copy each time ──
            codedict["exp"] += torch.randn_like(codedict["exp"]) * EXPR_NOISE_STD

            out = render_variation(codedict, use_texture)

            save_path = os.path.join(output_dir, f"{stem}_var{i:02d}.jpg")
            cv2.imwrite(save_path, out)

    print(f"[OK] {person}: {len(os.listdir(output_dir))} images generated")


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isdir(RAW_DIR):
        print(f"[ERR] Raw directory not found: {RAW_DIR}")
        print("       Place one folder per person inside data/raw/")
        sys.exit(1)

    persons = sorted(os.listdir(RAW_DIR))
    print(f"Found {len(persons)} person(s) in {RAW_DIR}")

    for person in persons:
        if os.path.isdir(os.path.join(RAW_DIR, person)):
            generate_for_person(person)

    print("\nAugmentation complete.")


if __name__ == "__main__":
    main()
 
        
