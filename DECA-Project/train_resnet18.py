"""
train_resnet18.py
─────────────────
Train a ResNet18-based face classifier for classroom attendance.

Pipeline (matches the project write-up):
  1. Load mixed dataset (DECA-synthetic + real photos)
  2. MTCNN crops & aligns every face at load time
  3. ResNet18 (ImageNet-pretrained backbone) with a custom classification head
  4. Training with mixed-precision, cosine LR schedule, class-weighted loss
  5. Evaluation with per-person accuracy, confusion matrix, and saved model

Hardware targets:
  GPU  : RTX 5060 Ti 16 GB (Blackwell sm_120)
  RAM  : 16 GB system
  Safe batch size: 64 @ 224×224  (~4-6 GB VRAM)

Usage:
  python train_resnet18.py                          # defaults
  python train_resnet18.py --epochs 50 --batch_size 64 --lr 1e-3
  python train_resnet18.py --data_dir data/combined --test_dir data/test
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models

from PIL import Image

# ── Optional: MTCNN from facenet-pytorch ──────────────────────────────────────
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    print("[WARN] facenet-pytorch not installed. Install with:")
    print("       pip install facenet-pytorch")
    print("       Face detection/cropping will be skipped.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224                 # ResNet18 input size
NUM_WORKERS = 4                # DataLoader workers (safe for 6 CPU cores)
PIN_MEMORY = True if DEVICE == "cuda" else False


# ══════════════════════════════════════════════════════════════════════════════
#  MTCNN FACE CROPPER
# ══════════════════════════════════════════════════════════════════════════════
class FaceCropper:
    """
    Detect + crop + align a single face from an image using MTCNN.

    If no face is detected, returns the original image (center-cropped).
    This mirrors the write-up's pipeline: MTCNN locates faces before
    the classifier sees them.
    """

    def __init__(self, image_size: int = IMG_SIZE, margin: int = 40,
                 device: str = DEVICE):
        if HAS_MTCNN:
            self.mtcnn = MTCNN(
                image_size=image_size,
                margin=margin,
                min_face_size=40,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,       # return 0-255 tensor, not normalised
                device=device,
            )
        else:
            self.mtcnn = None

        self.fallback = transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
        ])

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        if self.mtcnn is not None:
            try:
                # MTCNN returns a (3, H, W) tensor or None
                cropped = self.mtcnn(pil_img)
                if cropped is not None:
                    # Convert back to PIL for downstream transforms
                    arr = cropped.permute(1, 2, 0).byte().numpy()
                    return Image.fromarray(arr)
            except Exception:
                pass  # fall through to centre crop

        return self.fallback(pil_img)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════
class FaceDataset(Dataset):
    """
    Reads a directory of structure:
        root/<person_name>/<image>.jpg

    Each person_name becomes a class label.
    Images are cropped via MTCNN, then augmented.
    """

    def __init__(self, root: str, transform=None, face_cropper=None,
                 cache_crops: bool = True):
        self.root = Path(root)
        self.transform = transform
        self.face_cropper = face_cropper
        self.cache_crops = cache_crops
        self._crop_cache: dict = {}

        # Build class map
        self.classes = sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and any(d.iterdir())
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

        # Collect (path, label) pairs
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")

        print(f"  Dataset: {len(self.samples)} images, "
              f"{len(self.classes)} classes from {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ── Load & crop face ──────────────────────────────────────────────
        if self.cache_crops and path in self._crop_cache:
            img = self._crop_cache[path].copy()
        else:
            img = Image.open(path).convert("RGB")
            if self.face_cropper is not None:
                img = self.face_cropper(img)
            if self.cache_crops:
                self._crop_cache[path] = img.copy()

        # ── Augment & normalise ───────────────────────────────────────────
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for handling class imbalance."""
        counts = np.zeros(len(self.classes))
        for _, label in self.samples:
            counts[label] += 1
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.sum() * len(self.classes)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> list[float]:
        """Per-sample weight for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        return [class_weights[label] for _, label in self.samples]


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════════
def get_train_transforms(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_test_transforms(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL
# ══════════════════════════════════════════════════════════════════════════════
class AttendanceResNet18(nn.Module):
    """
    ResNet18 with ImageNet-pretrained backbone.
    The final FC layer is replaced with a dropout + classification head.
    """

    def __init__(self, num_classes: int, dropout: float = 0.4,
                 freeze_backbone: bool = False):
        super().__init__()

        # Load pretrained backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features  # 512

        # Optionally freeze early layers for fine-tuning
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze layer4 + bn + fc
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True

        # Replace classifier head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 20 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  acc={100.*correct/total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (running_loss / total,
            100.0 * correct / total,
            np.array(all_preds),
            np.array(all_labels))


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTING
# ══════════════════════════════════════════════════════════════════════════════
def print_per_class_report(preds, labels, class_names):
    """Print per-person accuracy like a classification report."""
    print(f"\n  {'Person':<25s} {'Correct':>7s} {'Total':>7s} {'Accuracy':>9s}")
    print(f"  {'─'*50}")

    for idx, name in enumerate(class_names):
        mask = labels == idx
        if mask.sum() == 0:
            continue
        correct = (preds[mask] == idx).sum()
        total = mask.sum()
        acc = 100.0 * correct / total
        print(f"  {name:<25s} {correct:>7d} {total:>7d} {acc:>8.1f}%")

    overall = 100.0 * (preds == labels).sum() / len(labels)
    print(f"  {'─'*50}")
    print(f"  {'OVERALL':<25s} {(preds==labels).sum():>7d} "
          f"{len(labels):>7d} {overall:>8.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE — run on a new classroom photo
# ══════════════════════════════════════════════════════════════════════════════
def predict_classroom(model_path: str, image_path: str,
                      class_names: list[str], device: str = DEVICE):
    """
    Full inference pipeline:
      1. MTCNN detects all faces in the classroom image
      2. Each face is cropped, normalised, and classified by ResNet18
      3. Returns list of (name, confidence, bounding_box)
    """
    if not HAS_MTCNN:
        print("[ERR] facenet-pytorch required for classroom inference")
        return []

    # Load model
    num_classes = len(class_names)
    model = AttendanceResNet18(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    # Detect faces
    mtcnn_detect = MTCNN(
        keep_all=True,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        device=device,
    )

    img = Image.open(image_path).convert("RGB")
    boxes, probs = mtcnn_detect.detect(img)

    if boxes is None:
        print("No faces detected.")
        return []

    # Crop each face and classify
    transform = get_test_transforms()
    results = []

    for box, det_prob in zip(boxes, probs):
        if det_prob < 0.9:
            continue

        # Crop with margin
        x1, y1, x2, y2 = [int(b) for b in box]
        margin = int(max(x2 - x1, y2 - y1) * 0.2)
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img.width, x2 + margin)
        y2 = min(img.height, y2 + margin)

        face_crop = img.crop((x1, y1, x2, y2))
        face_tensor = transform(face_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(face_tensor)
            probs_cls = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs_cls.max(1)

        name = class_names[pred_idx.item()]
        conf = confidence.item()
        results.append({
            "name": name,
            "confidence": conf,
            "box": [x1, y1, x2, y2],
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 face classifier for classroom attendance."
    )

    # ── Data ──
    parser.add_argument("--data_dir", type=str, default="data/combined",
                        help="Training images root (person subfolders)")
    parser.add_argument("--test_dir", type=str, default="data/test",
                        help="Test images root (person subfolders)")

    # ── Training ──
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (64 safe for 16GB VRAM @ 224px)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze ResNet layers except layer4+fc")

    # ── Output ──
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--model_name", type=str, default="attendance_resnet18")

    # ── Inference mode ──
    parser.add_argument("--predict", type=str, default=None,
                        help="Path to classroom image for inference (skips training)")

    args = parser.parse_args()

    # ── Inference-only mode ───────────────────────────────────────────────
    if args.predict:
        meta_path = os.path.join(args.save_dir, f"{args.model_name}_meta.json")
        if not os.path.exists(meta_path):
            print(f"[ERR] Metadata not found: {meta_path}")
            print("      Train the model first.")
            sys.exit(1)

        with open(meta_path) as f:
            meta = json.load(f)

        model_path = os.path.join(args.save_dir, f"{args.model_name}_best.pth")
        results = predict_classroom(model_path, args.predict, meta["classes"])

        print(f"\nAttendance results for: {args.predict}")
        print(f"{'─'*50}")
        for r in results:
            status = "✓ PRESENT" if r["confidence"] > 0.7 else "? UNCERTAIN"
            print(f"  {status}  {r['name']:<20s}  "
                  f"conf={r['confidence']:.2%}  box={r['box']}")
        return

    # ── Setup ─────────────────────────────────────────────────────────────
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ResNet18 Attendance Classifier — Training Pipeline    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Device       : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  VRAM         : {vram:.1f} GB")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print()

    # ── Face cropper ──────────────────────────────────────────────────────
    print("▸ Initialising MTCNN face cropper …")
    cropper = FaceCropper(image_size=IMG_SIZE, device=DEVICE)
    if cropper.mtcnn:
        print("  ✓ MTCNN ready")
    else:
        print("  ⚠ MTCNN unavailable — using centre crop fallback")

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\n▸ Loading datasets …")

    # If test_dir doesn't exist but data_dir does, auto-split
    if not os.path.isdir(args.test_dir) and os.path.isdir(args.data_dir):
        print(f"  [INFO] No test directory found at {args.test_dir}")
        print(f"         Using 80/20 random split from {args.data_dir}")
        # We'll handle this via a random split below
        full_dataset = FaceDataset(
            args.data_dir,
            transform=None,  # transforms applied after split
            face_cropper=cropper,
        )
        n_total = len(full_dataset)
        n_test = max(1, int(n_total * 0.2))
        n_train = n_total - n_test

        gen = torch.Generator().manual_seed(42)
        train_subset, test_subset = torch.utils.data.random_split(
            full_dataset, [n_train, n_test], generator=gen
        )

        # Wrap with transforms
        class TransformedSubset(Dataset):
            def __init__(self, subset, transform):
                self.subset = subset
                self.transform = transform
            def __len__(self):
                return len(self.subset)
            def __getitem__(self, idx):
                img, label = self.subset[idx]
                if self.transform and isinstance(img, Image.Image):
                    img = self.transform(img)
                return img, label

        train_dataset = TransformedSubset(train_subset, get_train_transforms())
        test_dataset = TransformedSubset(test_subset, get_test_transforms())
        class_names = full_dataset.classes
        class_weights = full_dataset.get_class_weights()

    else:
        train_dataset = FaceDataset(
            args.data_dir,
            transform=get_train_transforms(),
            face_cropper=cropper,
        )
        test_dataset = FaceDataset(
            args.test_dir,
            transform=get_test_transforms(),
            face_cropper=cropper,
        )
        class_names = train_dataset.classes
        class_weights = train_dataset.get_class_weights()

        # Verify class consistency
        if hasattr(test_dataset, 'classes') and test_dataset.classes != class_names:
            print("  [WARN] Train/test class lists differ!")
            print(f"    Train: {class_names}")
            print(f"    Test : {test_dataset.classes}")

    num_classes = len(class_names)
    print(f"\n  Classes ({num_classes}): {', '.join(class_names)}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    # Use weighted sampler to handle synthetic/real imbalance
    if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'subset'):
        # random_split path — can't easily do weighted sampling
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
        )
    else:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
        )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\n▸ Building ResNet18 ({num_classes} classes, "
          f"dropout={args.dropout}, freeze={args.freeze_backbone}) …")
    model = AttendanceResNet18(
        num_classes=num_classes,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {train_params:,}")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE),
        label_smoothing=0.1,
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # ── Training loop ─────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    best_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    print(f"\n{'═'*60}")
    print(f"  TRAINING — {args.epochs} epochs")
    print(f"{'═'*60}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch}/{args.epochs}  (lr={lr:.2e})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE, epoch
        )

        test_loss, test_acc, preds, labels = evaluate(
            model, test_loader, criterion, DEVICE
        )

        scheduler.step()

        elapsed = time.time() - t0
        print(f"    ► train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%")
        print(f"    ► test_loss ={test_loss:.4f}   test_acc ={test_acc:.1f}%  "
              f"({elapsed:.1f}s)")

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_path = os.path.join(
                args.save_dir, f"{args.model_name}_best.pth"
            )
            torch.save(model.state_dict(), save_path)
            print(f"    ★ New best! Saved → {save_path}")

        print()

    # ── Save last model + metadata ────────────────────────────────────────
    last_path = os.path.join(args.save_dir, f"{args.model_name}_last.pth")
    torch.save(model.state_dict(), last_path)

    meta = {
        "classes": class_names,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "args": vars(args),
        "history": history,
    }
    meta_path = os.path.join(args.save_dir, f"{args.model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ── Final evaluation ──────────────────────────────────────────────────
    print(f"{'═'*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'═'*60}")
    print(f"  Best test accuracy : {best_acc:.1f}%  (epoch {best_epoch})")
    print(f"  Model saved to     : {args.save_dir}/")
    print(f"  Metadata           : {meta_path}")

    # Load best model for final report
    model.load_state_dict(torch.load(
        os.path.join(args.save_dir, f"{args.model_name}_best.pth"),
        map_location=DEVICE, weights_only=True,
    ))
    _, _, preds, labels = evaluate(model, test_loader, criterion, DEVICE)
    print_per_class_report(preds, labels, class_names)

    # ── Inference example ─────────────────────────────────────────────────
    print(f"\n  To run attendance on a classroom photo:")
    print(f"    python train_resnet18.py --predict classroom.jpg")
    print()


if __name__ == "__main__":
    main()
