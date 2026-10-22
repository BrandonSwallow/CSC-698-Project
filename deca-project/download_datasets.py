"""
download_datasets.py
────────────────────
Download, organise, and split face-recognition datasets for training
and testing a ResNet18 attendance classifier.

Datasets handled
────────────────
1. LFW   — 13 233 images / 5 749 identities  (auto-download via sklearn)
2. VGGFace2 — 3.31 M images / 9 131 identities (manual download required)

Usage
─────
    python download_datasets.py --dataset lfw          # quick start
    python download_datasets.py --dataset vggface2 \\
        --vggface2_train_tar /path/to/vggface2_train.tar.gz \\
        --vggface2_test_tar  /path/to/vggface2_test.tar.gz
    python download_datasets.py --dataset both ...

The script creates:
    data/
    ├── lfw/
    │   ├── train/   (80 %)
    │   └── test/    (20 %)
    ├── vggface2/
    │   ├── train/
    │   └── test/
    └── train/       ← your DECA-augmented data (from augment_deca.py)
"""

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path("data")
LFW_DIR = BASE_DIR / "lfw"
VGG_DIR = BASE_DIR / "vggface2"

MIN_IMAGES_PER_PERSON = 5     # skip identities with fewer photos
TRAIN_RATIO = 0.80            # 80/20 train-test split
SEED = 42


# ──────────────────────────────────────────────────────────────────────────────
# LFW  (auto-download via scikit-learn)
# ──────────────────────────────────────────────────────────────────────────────
def download_lfw(min_faces: int = MIN_IMAGES_PER_PERSON,
                 image_size: int = 224) -> None:
    """Download LFW via sklearn and split into train/test folders."""
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image

    print("\n══════════════════════════════════════")
    print("  Downloading LFW via scikit-learn")
    print("══════════════════════════════════════\n")

    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=1.0,         # keep original resolution
        color=True,
        download_if_missing=True,
    )

    images = lfw.images          # (N, H, W, 3)  float64 0-255
    labels = lfw.target
    names = lfw.target_names

    print(f"Loaded {len(images)} images across {len(names)} identities "
          f"(min {min_faces} images each)\n")

    train_dir = LFW_DIR / "train"
    test_dir = LFW_DIR / "test"

    rng = np.random.RandomState(SEED)

    for idx, name in enumerate(names):
        mask = labels == idx
        person_imgs = images[mask]
        n = len(person_imgs)

        # Shuffle deterministically
        order = rng.permutation(n)
        person_imgs = person_imgs[order]

        split = int(n * TRAIN_RATIO)
        splits = {"train": person_imgs[:split], "test": person_imgs[split:]}

        for split_name, split_imgs in splits.items():
            out = (train_dir if split_name == "train" else test_dir) / name.replace(" ", "_")
            out.mkdir(parents=True, exist_ok=True)

            for i, img_arr in enumerate(split_imgs):
                img = Image.fromarray(img_arr.astype(np.uint8))
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(out / f"{i:04d}.jpg", quality=95)

    _print_split_stats("LFW", train_dir, test_dir)


# ──────────────────────────────────────────────────────────────────────────────
# VGGFace2  (requires manual download — files are ~36 GB)
# ──────────────────────────────────────────────────────────────────────────────
def setup_vggface2(train_tar: str | None, test_tar: str | None,
                   max_identities: int = 100,
                   image_size: int = 224) -> None:
    """
    Extract and organise a subset of VGGFace2.

    Because VGGFace2 is enormous (3.3 M images), we extract a manageable
    subset of `max_identities` people with the most images.  Adjust
    max_identities upward if you have the disk space and GPU hours.
    """
    from PIL import Image

    print("\n══════════════════════════════════════")
    print("  Setting up VGGFace2")
    print("══════════════════════════════════════\n")

    if not train_tar or not os.path.isfile(train_tar):
        print("[INFO] VGGFace2 requires a manual download (≈36 GB).")
        print("       1. Go to  https://github.com/ox-vgg/vgg_face2")
        print("          or     https://academictorrents.com  (search 'VGGFace2')")
        print("       2. Download vggface2_train.tar.gz and vggface2_test.tar.gz")
        print("       3. Re-run with:")
        print("            python download_datasets.py --dataset vggface2 \\")
        print("                --vggface2_train_tar /path/to/train.tar.gz \\")
        print("                --vggface2_test_tar  /path/to/test.tar.gz")
        return

    # ── Extract to a temp location ────────────────────────────────────────
    raw_dir = VGG_DIR / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for label, tar_path in [("train", train_tar), ("test", test_tar)]:
        if tar_path and os.path.isfile(tar_path):
            print(f"Extracting {label} archive — this may take a while …")
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(raw_dir)

    # ── Find identity folders ─────────────────────────────────────────────
    # VGGFace2 structure: train/<identity_id>/xxxx.jpg
    search_root = None
    for candidate in [raw_dir / "train", raw_dir / "data" / "train", raw_dir]:
        if candidate.is_dir() and any(candidate.iterdir()):
            search_root = candidate
            break

    if search_root is None:
        print("[ERR] Could not locate identity folders after extraction.")
        return

    # Count images per identity, pick the top N
    id_counts = {}
    for d in sorted(search_root.iterdir()):
        if d.is_dir():
            n_imgs = sum(1 for f in d.iterdir()
                         if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if n_imgs >= MIN_IMAGES_PER_PERSON:
                id_counts[d] = n_imgs

    if not id_counts:
        print("[ERR] No identities found with enough images.")
        return

    # Sort by image count descending, take the top subset
    selected = sorted(id_counts, key=id_counts.get, reverse=True)[:max_identities]
    print(f"Selected {len(selected)} identities out of {len(id_counts)} available")

    train_dir = VGG_DIR / "train"
    test_dir = VGG_DIR / "test"
    rng = np.random.RandomState(SEED)

    for id_dir in selected:
        person_name = id_dir.name
        img_files = sorted([
            f for f in id_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        order = rng.permutation(len(img_files))
        img_files = [img_files[i] for i in order]
        split = int(len(img_files) * TRAIN_RATIO)

        for split_name, subset in [("train", img_files[:split]),
                                    ("test", img_files[split:])]:
            out = (train_dir if split_name == "train" else test_dir) / person_name
            out.mkdir(parents=True, exist_ok=True)

            for i, src in enumerate(subset):
                try:
                    img = Image.open(src).convert("RGB")
                    img = img.resize((image_size, image_size), Image.LANCZOS)
                    img.save(out / f"{i:04d}.jpg", quality=95)
                except Exception as e:
                    print(f"  [WARN] Skipped {src.name}: {e}")

    # Optionally clean up raw extraction
    shutil.rmtree(raw_dir, ignore_errors=True)

    _print_split_stats("VGGFace2", train_dir, test_dir)


# ──────────────────────────────────────────────────────────────────────────────
# COMBINE: merge synthetic DECA data with real data
# ──────────────────────────────────────────────────────────────────────────────
def combine_datasets(real_dir: Path, synthetic_dir: Path, out_dir: Path) -> None:
    """
    Merge real and DECA-synthesised images into a single training folder.

    Structure:  out_dir/<person>/<img>.jpg
    Files are copied (not moved) so originals stay intact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for src in [real_dir, synthetic_dir]:
        if not src.is_dir():
            continue
        for person_dir in sorted(src.iterdir()):
            if not person_dir.is_dir():
                continue
            dest = out_dir / person_dir.name
            dest.mkdir(parents=True, exist_ok=True)

            prefix = "real_" if src == real_dir else "syn_"
            for img_file in person_dir.iterdir():
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    shutil.copy2(img_file, dest / f"{prefix}{img_file.name}")

    total = sum(1 for _ in out_dir.rglob("*.jpg"))
    persons = sum(1 for d in out_dir.iterdir() if d.is_dir())
    print(f"\n[COMBINED] {total} images across {persons} identities → {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def _print_split_stats(name: str, train_dir: Path, test_dir: Path) -> None:
    """Print a summary of train/test split."""
    def _count(d: Path):
        persons = [p for p in d.iterdir() if p.is_dir()] if d.exists() else []
        imgs = sum(sum(1 for f in p.iterdir() if f.suffix == ".jpg") for p in persons)
        return len(persons), imgs

    tr_p, tr_i = _count(train_dir)
    te_p, te_i = _count(test_dir)

    print(f"\n{'─'*45}")
    print(f"  {name} dataset ready")
    print(f"{'─'*45}")
    print(f"  Train : {tr_p:>5} identities │ {tr_i:>7} images")
    print(f"  Test  : {te_p:>5} identities │ {te_i:>7} images")
    print(f"  Paths : {train_dir}")
    print(f"          {test_dir}")
    print(f"{'─'*45}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download & prepare face-recognition datasets for ResNet18 training."
    )
    parser.add_argument(
        "--dataset", choices=["lfw", "vggface2", "both"], default="lfw",
        help="Which dataset(s) to set up (default: lfw)"
    )
    parser.add_argument(
        "--vggface2_train_tar", type=str, default=None,
        help="Path to vggface2_train.tar.gz"
    )
    parser.add_argument(
        "--vggface2_test_tar", type=str, default=None,
        help="Path to vggface2_test.tar.gz"
    )
    parser.add_argument(
        "--max_identities", type=int, default=100,
        help="Max identities to extract from VGGFace2 (default: 100)"
    )
    parser.add_argument(
        "--min_faces", type=int, default=MIN_IMAGES_PER_PERSON,
        help="Min images per person to include (default: 5)"
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Resize all images to this square size (default: 224)"
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Also merge real + synthetic (data/train) into data/combined/"
    )

    args = parser.parse_args()

    if args.dataset in ("lfw", "both"):
        download_lfw(min_faces=args.min_faces, image_size=args.image_size)

    if args.dataset in ("vggface2", "both"):
        setup_vggface2(
            train_tar=args.vggface2_train_tar,
            test_tar=args.vggface2_test_tar,
            max_identities=args.max_identities,
            image_size=args.image_size,
        )

    if args.combine:
        # Merge whichever real dataset exists with DECA synthetic data
        real = LFW_DIR / "train" if (LFW_DIR / "train").exists() else VGG_DIR / "train"
        synthetic = BASE_DIR / "train"
        combine_datasets(real, synthetic, BASE_DIR / "combined")


if __name__ == "__main__":
    main()
