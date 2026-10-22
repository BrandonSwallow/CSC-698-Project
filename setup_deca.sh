#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DECA Environment Setup — RTX 5060 Ti (Blackwell sm_120) / Ubuntu 22.04    ║
# ║                                                                            ║
# ║  This script sets up DECA in a Python venv that PRESERVES your existing    ║
# ║  custom-compiled PyTorch (CUDA 12.8 / sm_120).                            ║
# ║                                                                            ║
# ║  Hardware assumptions:                                                     ║
# ║    GPU  — NVIDIA RTX 5060 Ti, 16 GB VRAM, Blackwell (sm_120)              ║
# ║    CPU  — 6 cores (Ryzen 3600)                                            ║
# ║    RAM  — 16 GB system                                                     ║
# ║    Disk — 100 GB available                                                 ║
# ║                                                                            ║
# ║  Run:  bash setup_deca.sh                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION — edit these if your paths differ
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/deca-project"
VENV_DIR="$PROJECT_DIR/venv"
DECA_DIR="$PROJECT_DIR/DECA"
DATA_DIR="$DECA_DIR/data"
PYTHON_BIN="python3"           # change to python3.10 etc. if needed

# Where is your pre-built PyTorch?
# If it's a system-wide install or in another venv, we'll link it.
EXISTING_TORCH_PATH=""         # leave empty to auto-detect

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         DECA Setup for Blackwell sm_120 / CUDA 12.8        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────
echo "▸ [1/8] Pre-flight checks …"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "  ✗ nvidia-smi not found. Is the GPU passthrough working?"
    exit 1
fi
echo "  ✓ GPU detected:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'

# Check Python
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "  ✗ $PYTHON_BIN not found"
    exit 1
fi
PY_VER=$($PYTHON_BIN --version 2>&1)
echo "  ✓ $PY_VER"

# Check that PyTorch is already installed and has CUDA
TORCH_CHECK=$($PYTHON_BIN -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Arch list: {torch.cuda.get_arch_list()}')
" 2>&1) || true

echo "  ─ PyTorch status:"
echo "$TORCH_CHECK" | sed 's/^/    /'

if echo "$TORCH_CHECK" | grep -q "CUDA available: False"; then
    echo ""
    echo "  ⚠  WARNING: PyTorch reports CUDA unavailable."
    echo "     Your custom sm_120 build may not be on the current PATH."
    echo "     This script will continue, but DECA will fall back to CPU."
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CREATE PROJECT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
echo "▸ [2/8] Creating project structure …"

mkdir -p "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/data/raw"       # your student portraits go here
mkdir -p "$PROJECT_DIR/data/train"     # DECA-augmented outputs
mkdir -p "$PROJECT_DIR/data/test"      # held-out real photos

echo "  ✓ $PROJECT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CREATE VIRTUAL ENVIRONMENT (inheriting system PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
echo "▸ [3/8] Creating Python venv (--system-site-packages to keep PyTorch) …"

$PYTHON_BIN -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip inside venv
pip install --upgrade pip wheel 2>/dev/null

# Verify torch is still visible
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null \
    && echo "  ✓ PyTorch with CUDA confirmed inside venv" \
    || echo "  ⚠ PyTorch CUDA check failed — will continue anyway"

# ─────────────────────────────────────────────────────────────────────────────
# 4.  CLONE DECA
# ─────────────────────────────────────────────────────────────────────────────
echo "▸ [4/8] Cloning DECA repository …"

if [ -d "$DECA_DIR" ]; then
    echo "  ✓ DECA already cloned at $DECA_DIR"
else
    git clone https://github.com/yfeng95/DECA.git "$DECA_DIR"
    echo "  ✓ Cloned"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5.  INSTALL DEPENDENCIES (without overwriting PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
echo "▸ [5/8] Installing DECA dependencies …"
echo "         (skipping torch/torchvision — using your sm_120 build)"

# Core scientific deps — pinned loosely to avoid conflicts with modern Python
pip install \
    "numpy>=1.23,<2.0" \
    "scipy>=1.4.1" \
    "scikit-image>=0.19" \
    "opencv-python>=4.1.1" \
    "PyYAML>=5.1" \
    "yacs>=0.1.8" \
    "kornia>=0.6" \
    "ninja" \
    "fvcore" \
    "chumpy>=0.69" \
    "face-alignment>=1.3.4" \
    "Pillow>=9.0" \
    "scikit-learn" \
    2>&1 | tail -1

echo "  ✓ Core deps installed"

# ── PyTorch3D (needed for the rasterizer) ────────────────────────────────────
# PyTorch3D must be built from source for sm_120 / CUDA 12.8.
# Pre-built wheels only cover up to sm_90 (Ada Lovelace).
echo ""
echo "▸ [5b/8] Building PyTorch3D from source (for sm_120) …"
echo "         This compiles CUDA kernels — expect 5-15 minutes."

PYTORCH3D_DIR="$PROJECT_DIR/pytorch3d"
if python -c "import pytorch3d" 2>/dev/null; then
    echo "  ✓ PyTorch3D already installed"
else
    if [ ! -d "$PYTORCH3D_DIR" ]; then
        git clone https://github.com/facebookresearch/pytorch3d.git "$PYTORCH3D_DIR"
    fi
    cd "$PYTORCH3D_DIR"

    # Force compilation for Blackwell
    export TORCH_CUDA_ARCH_LIST="12.0"
    export FORCE_CUDA=1

    # Clean any stale build artifacts
    rm -rf build/ **/*.so 2>/dev/null || true

    pip install -e . --no-build-isolation --no-build-isolation 2>&1 | tail -5
    echo "  ✓ PyTorch3D built and installed"
    cd "$PROJECT_DIR"
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6.  DOWNLOAD DECA PRETRAINED MODEL
# ─────────────────────────────────────────────────────────────────────────────
echo "▸ [6/8] Downloading DECA pretrained model …"

mkdir -p "$DATA_DIR"

# DECA checkpoint
DECA_CKPT="$DATA_DIR/deca_model.tar"
if [ -f "$DECA_CKPT" ]; then
    echo "  ✓ DECA checkpoint already present"
else
    echo "  ↓ Downloading deca_model.tar …"
    # The model is hosted on the DECA project page
    # If this URL stops working, download manually from:
    #   https://deca.is.tue.mpg.de/
    pip install gdown 2>/dev/null
    gdown --fuzzy "https://drive.google.com/file/d/1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje/view" \
        -O "$DECA_CKPT" 2>/dev/null \
    && echo "  ✓ Downloaded" \
    || echo "  ⚠ Auto-download failed. Please download manually — see instructions below."
fi

# ─────────────────────────────────────────────────────────────────────────────
# 7.  FLAME MODEL WEIGHTS (manual registration required)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "▸ [7/8] FLAME model setup"
echo ""
echo "  ┌─────────────────────────────────────────────────────────────────┐"
echo "  │  FLAME requires free registration at:                          │"
echo "  │                                                                │"
echo "  │    https://flame.is.tue.mpg.de/register.php                    │"
echo "  │                                                                │"
echo "  │  After registering and logging in, download these files        │"
echo "  │  from the Downloads page:                                      │"
echo "  │                                                                │"
echo "  │    1. generic_model.pkl          → FLAME parametric model      │"
echo "  │    2. FLAME_texture.npz          → UV texture space            │"
echo "  │    3. FLAME_albedo_from_BFM.npz  → (optional) albedo model     │"
echo "  │                                                                │"
echo "  │  Place them in:                                                │"
echo "  │    $DATA_DIR/"
echo "  │                                                                │"
echo "  │  The FLAME 2023 model is under CC BY 4.0 (open for research). │"
echo "  └─────────────────────────────────────────────────────────────────┘"
echo ""

# Check if files are already present
FLAME_FILES=("generic_model.pkl" "FLAME_texture.npz")
ALL_PRESENT=true
for f in "${FLAME_FILES[@]}"; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✓ $f found"
    else
        echo "  ✗ $f MISSING — download from FLAME website"
        ALL_PRESENT=false
    fi
done

# Also check for DECA's other required data files
OTHER_FILES=(
    "head_template.obj"
    "landmark_embedding.npy"
    "mean_texture.jpg"
    "uv_face_eye_mask.png"
    "uv_face_mask.png"
    "fixed_displacement_256.npy"
)
echo ""
echo "  Other DECA data files (should come with the repo):"
for f in "${OTHER_FILES[@]}"; do
    if [ -f "$DATA_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ? $f — may need to be downloaded with the DECA model"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# 8.  VERIFY & PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "▸ [8/8] Verification …"
echo ""

cd "$DECA_DIR"
python -c "
import sys, torch
print(f'Python      : {sys.version.split()[0]}')
print(f'PyTorch     : {torch.__version__}')
print(f'CUDA avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'VRAM        : {mem:.1f} GB')

try:
    import pytorch3d
    print(f'PyTorch3D   : {pytorch3d.__version__}')
except ImportError:
    print('PyTorch3D   : NOT INSTALLED')

try:
    import cv2
    print(f'OpenCV      : {cv2.__version__}')
except ImportError:
    print('OpenCV      : NOT INSTALLED')

try:
    import kornia
    print(f'Kornia      : {kornia.__version__}')
except ImportError:
    print('Kornia      : NOT INSTALLED')

try:
    from face_alignment import FaceAlignment
    print(f'FaceAlign   : OK')
except ImportError:
    print('FaceAlign   : NOT INSTALLED')
"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                        SETUP COMPLETE                              ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║                                                                    ║"
echo "║  Project root: $PROJECT_DIR"
echo "║                                                                    ║"
echo "║  To activate:                                                      ║"
echo "║    source $VENV_DIR/bin/activate"
echo "║                                                                    ║"
echo "║  Directory layout:                                                 ║"
echo "║    DECA/                    ← DECA source code                     ║"
echo "║    DECA/data/               ← model weights go here               ║"
echo "║      ├── deca_model.tar     ← DECA checkpoint                     ║"
echo "║      ├── generic_model.pkl  ← FLAME model (download manually)     ║"
echo "║      ├── FLAME_texture.npz  ← FLAME textures (download manually)  ║"
echo "║      └── head_template.obj  ← mesh topology (comes with repo)     ║"
echo "║    data/                                                           ║"
echo "║      ├── raw/<person>/      ← put source portraits here           ║"
echo "║      ├── train/<person>/    ← DECA-augmented output               ║"
echo "║      └── test/<person>/     ← held-out real photos                ║"
echo "║                                                                    ║"
echo "║  Next steps:                                                       ║"
echo "║    1. Download FLAME weights from flame.is.tue.mpg.de             ║"
echo "║    2. Place portraits in data/raw/<person_name>/                   ║"
echo "║    3. Run:  python augment_deca.py                                 ║"
echo "║    4. Run:  python download_datasets.py --dataset lfw              ║"
echo "║                                                                    ║"
echo "║  VRAM budget (16 GB RTX 5060 Ti):                                 ║"
echo "║    DECA encode/decode  : ~2-3 GB  (one image at a time is safe)   ║"
echo "║    ResNet18 training   : ~4-6 GB  (batch_size=64 @ 224×224)       ║"
echo "║    Max safe batch size : 128      (for ResNet18 @ 224×224)        ║"
echo "║    Leave ~2 GB headroom for PyTorch caching allocator             ║"
echo "║                                                                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
