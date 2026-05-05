"""
VoxDetect — src/config.py
Central configuration: all hyperparameters, paths, and constants.
Based on: Mubeen et al. (2021), Computers and Electrical Engineering, 91, 107122
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────
# Root is always the vox_detect directory (two levels up from this file)
ROOT_DIR       = Path(__file__).resolve().parent.parent

DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"           # Place Kaggle dataset here
PROCESSED_DIR  = DATA_DIR / "processed"    # Segmented per-word WAVs
FORGED_DIR     = DATA_DIR / "forged"       # Spliced forged sentences
SPLITS_DIR     = DATA_DIR / "splits"       # train/test JSON splits

MODELS_DIR     = ROOT_DIR / "models"
SPEECH_MODELS_DIR  = MODELS_DIR / "speech"    # One GMM per digit word
SPEAKER_MODELS_DIR = MODELS_DIR / "speakers"  # One GMM per (speaker, word)

RESULTS_DIR    = ROOT_DIR / "results"
FIGURES_DIR    = RESULTS_DIR / "figures"
REPORTS_DIR    = RESULTS_DIR / "reports"

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# Paper uses 16 kHz; dataset is recorded at 44.1 kHz → we resample
TARGET_SR      = 16_000          # Target sample rate (Hz)
RAW_SR         = 44_100          # Original dataset sample rate

# ─────────────────────────────────────────────────────────────────────────────
# VAD (VOICE ACTIVITY DETECTION) PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# From paper Section 2.2: "a VAD method based on volume and zero crossing"
VAD_FRAME_MS       = 20          # Frame duration in milliseconds
VAD_FRAME_SAMPLES  = int(TARGET_SR * VAD_FRAME_MS / 1000)  # 320 @ 16kHz
VAD_THRESHOLD_PCT  = 0.01        # 1% adaptive threshold — tuned for this dataset
                                 # (paper uses 3% on KSUD; Kaggle corpus is quieter)
VAD_MIN_VOICED_FRAMES = 5        # ≥5 consecutive voiced frames = one word
                                 # (paper: 10; lowered because words are shorter here)
VAD_PAD_FRAMES     = 2           # Padding frames around voiced regions
VAD_ZCR_THRESHOLD  = 1.0         # ZCR disabled (Arabic has high ZCR ≥0.10 for
                                 # fricatives/stops; energy-only VAD works better here)

# ─────────────────────────────────────────────────────────────────────────────
# MFCC FEATURE EXTRACTION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
N_MFCC        = 13               # Number of MFCC coefficients (paper standard)
N_FFT         = 512              # FFT size
HOP_LENGTH    = 160              # Hop length = 10ms @ 16kHz
WIN_LENGTH    = 400              # Window length = 25ms @ 16kHz
N_MELS        = 26               # Number of Mel filter banks
FMIN          = 0                # Minimum frequency for Mel filters
FMAX          = TARGET_SR // 2   # Maximum frequency (Nyquist)
USE_DELTA     = True             # Append delta features → 26 total
USE_DELTA2    = True             # Append delta-delta features → 39 total

# Total feature dimension: 13 * 3 = 39 (if both deltas used)
N_FEATURES    = N_MFCC * (1 + int(USE_DELTA) + int(USE_DELTA2))  # 39

# ─────────────────────────────────────────────────────────────────────────────
# GMM HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# From Table 3: 256 mixtures = best accuracy (paper); 64 is much faster and still good.
# Use --speech-mixtures 256 on 02_train_models.py if you want full paper accuracy.
SPEECH_GMM_COMPONENTS  = 64    # Fast default; paper optimal = 256

# From Table 4: 32 mixtures = best speaker accuracy; 16 is fast and still strong.
SPEECH_GMM_COMPONENTS_PAPER   = 256   # Paper optimal (slow)
SPEAKER_GMM_COMPONENTS_PAPER  = 32    # Paper optimal (slow)

SPEAKER_GMM_COMPONENTS = 16    # Fast default; paper optimal = 32

GMM_MAX_ITER           = 100    # Max EM iterations (paper doesn't specify; 100 is enough)
GMM_N_INIT             = 1      # k-means restarts (1 = fastest; paper method = 1)
GMM_COVARIANCE_TYPE    = "diag" # Diagonal covariance (efficient for speech)
GMM_RANDOM_STATE       = 42     # For reproducibility
GMM_TOL                = 1e-3   # Convergence tolerance (relaxed for speed)
GMM_REG_COVAR          = 1e-3   # Regularization — prevents ill-defined covariance

# Mixture counts to evaluate during optimization experiments
SPEECH_GMM_EVAL_MIXTURES  = [8, 16, 32, 64, 128, 256]
SPEAKER_GMM_EVAL_MIXTURES = [4, 8, 16, 32]

# ─────────────────────────────────────────────────────────────────────────────
# DATASET / SPEAKER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Dataset has speakers S01–S50
N_SPEAKERS     = 50
SPEAKER_IDS    = [f"S{i:02d}" for i in range(1, N_SPEAKERS + 1)]

# Arabic digits — dictionary from paper Table 1
# 0-indexed → digit label 0 = Arabic "zero", 1 = "one", etc.
DIGIT_NAMES    = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]
# The paper uses digits 1–9 (not zero) for sentence generation
ACTIVE_DIGITS  = list(range(1, 10))   # 1 through 9

# Four test sentences (digit index sequences, 1-based)
# Sentence 1: digits 1,2,3,4,5,6
# Sentence 2: digits 2,3,4,5,6,7
# Sentence 3: digits 3,4,5,6,7,8
# Sentence 4: digits 4,5,6,7,8,9
SENTENCES = {
    "S1": [1, 2, 3, 4, 5, 6],
    "S2": [2, 3, 4, 5, 6, 7],
    "S3": [3, 4, 5, 6, 7, 8],
    "S4": [4, 5, 6, 7, 8, 9],
}

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
# Use ~70% of sessions per speaker for training, ~30% for testing
TRAIN_SESSION_FRACTION = 0.7

# ─────────────────────────────────────────────────────────────────────────────
# FORGERY GENERATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# In forgery: alternate digits come from an impostor speaker
# Pattern: [genuine, impostor, impostor, genuine, genuine, impostor]
#           digits  [0,        1,        2,        3,       4,       5] in sentence
FORGED_DIGIT_POSITIONS = [1, 2, 5]   # 0-indexed positions within sentence that are impostor
N_IMPOSTOR_COMBINATIONS = -1          # -1 = use all possible impostor pairs

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
# Log-likelihood threshold: if best speaker LLH < REJECT_THRESHOLD,
# reject regardless (optional, paper uses argmax only)
LLH_REJECT_THRESHOLD = None  # None = argmax decision only (paper method)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Ensure all output directories exist
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [
        PROCESSED_DIR, FORGED_DIR, SPLITS_DIR,
        SPEECH_MODELS_DIR, SPEAKER_MODELS_DIR,
        FIGURES_DIR, REPORTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("VoxDetect Configuration")
    print("=" * 50)
    print(f"Root:              {ROOT_DIR}")
    print(f"Target SR:         {TARGET_SR} Hz")
    print(f"MFCC coefficients: {N_MFCC} (+ delta + delta2 = {N_FEATURES} total)")
    print(f"Speech GMM:        {SPEECH_GMM_COMPONENTS} mixtures")
    print(f"Speaker GMM:       {SPEAKER_GMM_COMPONENTS} mixtures")
    print(f"Speakers:          {N_SPEAKERS} (S01–S50)")
    print(f"Active digits:     {ACTIVE_DIGITS}")
