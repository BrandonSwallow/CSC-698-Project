"""
run_pipeline.py
───────────────
End-to-end orchestrator for the classroom attendance system.

This script runs the full pipeline described in the project write-up:

  Step 1 │ DECA + FLAME  → Reconstruct 3D face from single portrait
  Step 2 │ Render        → Generate multi-view synthetic training images
  Step 3 │ Download      → Fetch real-world face datasets (LFW)
  Step 4 │ Combine       → Merge synthetic + real images
  Step 5 │ Train         → Train ResNet18 classifier with MTCNN cropping
  Step 6 │ Evaluate      → Per-person accuracy report

Usage:
  # Full pipeline (after DECA is set up & portraits placed in data/raw/)
  python run_pipeline.py

  # Skip DECA augmentation (already done), just train
  python run_pipeline.py --skip_augment --skip_download

  # Just run inference on a classroom photo
  python run_pipeline.py --predict classroom.jpg
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: str, desc: str) -> bool:
    """Run a shell command, printing a header."""
    print(f"\n{'━'*60}")
    print(f"  ▸ {desc}")
    print(f"{'━'*60}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {desc}")
        print(f"    Command: {cmd}")
        print(f"    Exit code: {result.returncode}")
        return False
    return True


def check_prereqs():
    """Verify the environment is ready."""
    errors = []

    # Python imports
    for pkg, name in [("torch", "PyTorch"), ("cv2", "OpenCV"),
                      ("PIL", "Pillow"), ("sklearn", "scikit-learn")]:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"  ✗ {name} ({pkg}) not installed")

    # CUDA
    import torch
    if not torch.cuda.is_available():
        errors.append("  ⚠ CUDA not available — will run on CPU (very slow)")

    # facenet-pytorch (MTCNN)
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        errors.append("  ✗ facenet-pytorch not installed (needed for MTCNN)")
        errors.append("    Fix: pip install facenet-pytorch")

    if errors:
        print("\nPre-flight check:")
        for e in errors:
            print(e)
        print()
        if any("✗" in e for e in errors):
            return False
    else:
        print("  ✓ All prerequisites OK")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run the full attendance recognition pipeline."
    )
    parser.add_argument("--skip_augment", action="store_true",
                        help="Skip DECA augmentation step")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip dataset download step")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training step")
    parser.add_argument("--predict", type=str, default=None,
                        help="Run inference on a classroom photo")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     Classroom Attendance System — Full Pipeline Runner      ║")
    print("║                                                             ║")
    print("║     DECA → FLAME renders → MTCNN → ResNet18 classifier     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Inference-only mode ───────────────────────────────────────────────
    if args.predict:
        run(f"python train_resnet18.py --predict {args.predict}",
            "Running attendance inference")
        return

    # ── Pre-flight ────────────────────────────────────────────────────────
    if not check_prereqs():
        print("Fix the issues above and re-run.")
        sys.exit(1)

    # ── Step 1+2: DECA augmentation ───────────────────────────────────────
    if not args.skip_augment:
        raw_dir = Path("data/raw")
        if not raw_dir.is_dir() or not any(raw_dir.iterdir()):
            print(f"\n  ⚠ No portraits found in {raw_dir}")
            print(f"    Create subfolders per person with 1+ photo each:")
            print(f"      data/raw/John_Smith/portrait.jpg")
            print(f"      data/raw/Jane_Doe/photo.jpg")
            print(f"\n    Skipping DECA augmentation.")
        else:
            ok = run("python augment_deca.py",
                     "Step 1-2: DECA 3D reconstruction + multi-view rendering")
            if not ok:
                print("  Continuing without synthetic data …")

    # ── Step 3: Download real datasets ────────────────────────────────────
    if not args.skip_download:
        run("python download_datasets.py --dataset lfw --min_faces 10",
            "Step 3: Downloading LFW dataset")

    # ── Step 4: Combine synthetic + real ──────────────────────────────────
    if not args.skip_download:
        run("python download_datasets.py --dataset lfw --combine",
            "Step 4: Merging synthetic + real data")

    # ── Step 5+6: Train + evaluate ────────────────────────────────────────
    if not args.skip_train:
        # Determine the best data directory
        if Path("data/combined").is_dir():
            data_dir = "data/combined"
        elif Path("data/train").is_dir():
            data_dir = "data/train"
        else:
            print("\n  ✗ No training data found.")
            print("    Run DECA augmentation or dataset download first.")
            sys.exit(1)

        test_dir = "data/test" if Path("data/test").is_dir() else data_dir

        cmd = (f"python train_resnet18.py "
               f"--data_dir {data_dir} "
               f"--test_dir {test_dir} "
               f"--epochs {args.epochs} "
               f"--batch_size {args.batch_size}")

        ok = run(cmd, "Step 5-6: Training ResNet18 + evaluation")
        if not ok:
            sys.exit(1)

    # ── Done ──────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✓ PIPELINE COMPLETE")
    print(f"{'═'*60}")
    print(f"  Model saved in: checkpoints/")
    print(f"  Run inference:  python run_pipeline.py --predict classroom.jpg")
    print()


if __name__ == "__main__":
    main()
