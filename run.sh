#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# VoxDetect — run.sh
# Convenience script: activate venv and run the full pipeline.
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# Activate virtual environment
if [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
    echo "✅ Virtual environment activated: $VENV"
else
    echo "⚠️  Virtual environment not found. Creating..."
    python3 -m venv "$VENV"
    source "$VENV/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          VoxDetect — Audio Forgery Detection System          ║"
echo "║   Based on: Mubeen et al. (2021)                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
STEP="${1:-all}"

case "$STEP" in
    "1" | "preprocess")
        echo "→ Running Step 1: Preprocessing..."
        python "$SCRIPT_DIR/scripts/01_preprocess.py" "${@:2}"
        ;;
    "2" | "train")
        echo "→ Running Step 2: Model Training..."
        python "$SCRIPT_DIR/scripts/02_train_models.py" "${@:2}"
        ;;
    "3" | "forgeries")
        echo "→ Running Step 3: Forgery Generation..."
        python "$SCRIPT_DIR/scripts/03_generate_forgeries.py" "${@:2}"
        ;;
    "4" | "evaluate")
        echo "→ Running Step 4: Evaluation..."
        python "$SCRIPT_DIR/scripts/04_evaluate.py" "${@:2}"
        ;;
    "demo")
        echo "→ Running Demo..."
        python "$SCRIPT_DIR/scripts/05_demo.py" "${@:2}"
        ;;
    "all")
        echo "→ Running FULL pipeline (Steps 1–4)..."
        echo ""
        python "$SCRIPT_DIR/scripts/01_preprocess.py"
        echo ""
        python "$SCRIPT_DIR/scripts/02_train_models.py"
        echo ""
        python "$SCRIPT_DIR/scripts/03_generate_forgeries.py"
        echo ""
        python "$SCRIPT_DIR/scripts/04_evaluate.py" --mixture-analysis
        ;;
    "test")
        echo "→ Running smoke tests..."
        python -c "
from src.config import ensure_dirs; ensure_dirs()
from src.vad import extract_word_regions
from src.features import extract_mfcc_matrix
from src.gmm_models import train_gmm, log_likelihood
import numpy as np

sr = 16000
# 4 distinct word bursts separated by silence
words = []
for freq in [300, 450, 600, 800]:
    w = np.sin(2*np.pi*freq*np.arange(sr//4)/sr).astype(np.float32) * 0.9
    words.append(w)
    words.append(np.zeros(sr//8, dtype=np.float32))
sig = np.concatenate(words)
regions = extract_word_regions(sig)
print(f'VAD: {len(regions)} words detected')
feat = extract_mfcc_matrix(words[0])
print(f'MFCC: shape = {feat.shape}')
gmm = train_gmm(np.random.randn(300, 39).astype(np.float32), 4)
print(f'GMM: LLH = {log_likelihood(gmm, feat):.2f}')
print('ALL TESTS PASSED ✅')
"
        ;;
    "notebook")
        echo "→ Launching Jupyter notebooks..."
        jupyter notebook "$SCRIPT_DIR/notebooks/"
        ;;
    *)
        echo "Usage: $0 [step]"
        echo ""
        echo "Steps:"
        echo "  all         Run complete pipeline (default)"
        echo "  1|preprocess  Step 1: VAD + Segmentation"
        echo "  2|train       Step 2: Train GMM models"
        echo "  3|forgeries   Step 3: Generate forged audio"
        echo "  4|evaluate    Step 4: Full evaluation + plots"
        echo "  demo         Interactive demo (pass --audio and --speaker)"
        echo "  test         Run smoke tests"
        echo "  notebook     Launch Jupyter notebooks"
        echo ""
        echo "Examples:"
        echo "  ./run.sh all"
        echo "  ./run.sh demo --audio data/forged/genuine/S1/S01_genuine.wav --speaker S01"
        echo "  ./run.sh 4 --mixture-analysis --no-plots"
        ;;
esac
