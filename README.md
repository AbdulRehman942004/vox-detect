# VoxDetect — Audio Forgery Detection System

> **Based on:** "Detection of impostor and tampered segments in audio by using an intelligent system"  
> Zeshan Mubeen, Mehtab Afzal, Zulfiqar Ali et al. — *Computers and Electrical Engineering*, 91 (2021) 107122

---

## Overview

VoxDetect is an intelligent audio forensics system that detects **audio splicing forgery** — the tampering of an audio recording by mixing speech from different speakers. The system:

1. Segments input audio into individual spoken words using **Voice Activity Detection (VAD)**
2. Extracts **MFCC** (Mel-Frequency Cepstral Coefficient) features from each word segment
3. Trains **Gaussian Mixture Models (GMM)** for both word recognition and speaker identification
4. Authenticates audio by comparing each segment's **log-likelihood** against registered speaker models
5. Flags and highlights any segments that do **not** belong to the claimed speaker

---

## Dataset

**Merged Arabic Corpus of Isolated Words** ([Kaggle](https://www.kaggle.com/datasets/mohamedanwarvic/merged-arabic-corpus-of-isolated-words))

- **50** Native-Arabic speakers (S01–S50)
- **~10 sessions** per speaker, 2 WAV files per session
  - `{speaker_id}.{session_id}.digits.wav` — Arabic digits 0–9
  - `{speaker_id}.{session_id}.words.wav` — Arabic words
- 44100 Hz, 16-bit resolution (resampled to 16kHz in pipeline)

### Download the Dataset

```bash
# Option 1: Using Kaggle CLI
pip install kaggle
kaggle datasets download -d mohamedanwarvic/merged-arabic-corpus-of-isolated-words
unzip merged-arabic-corpus-of-isolated-words.zip -d data/raw/

# Option 2: Manual download from Kaggle → place speaker folders in data/raw/
# data/raw/S01/, data/raw/S02/, ..., data/raw/S50/
```

---

## Project Structure

```
vox_detect/
├── article.pdf                    # Research paper
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/
│   ├── raw/                       # ← PUT DATASET HERE (S01...S50 folders)
│   ├── processed/                 # Segmented word WAV files (auto-generated)
│   ├── forged/                    # Generated forged sentences (auto-generated)
│   └── splits/                    # Train/test splits JSON (auto-generated)
│
├── src/
│   ├── config.py                  # Hyperparameters & paths
│   ├── vad.py                     # Voice Activity Detection
│   ├── segmentation.py            # Word boundary detection
│   ├── features.py                # MFCC feature extraction
│   ├── gmm_models.py              # GMM training & inference
│   ├── forgery_generator.py       # Forged audio creation
│   ├── authenticator.py           # Audio authentication
│   └── evaluator.py               # Metrics & visualization
│
├── models/                        # Saved GMM models (auto-generated)
│   ├── speech/                    # Word recognition models (256 mixtures)
│   └── speakers/                  # Speaker models (32 mixtures per speaker/word)
│
├── scripts/                       # Run in order
│   ├── 01_preprocess.py           # Step 1: VAD + segmentation
│   ├── 02_train_models.py         # Step 2: Train GMMs
│   ├── 03_generate_forgeries.py   # Step 3: Create forged dataset
│   ├── 04_evaluate.py             # Step 4: Full evaluation
│   └── 05_demo.py                 # Demo: test a single audio file
│
├── notebooks/                     # Jupyter notebooks for analysis
│
└── results/                       # Output plots & CSV tables
```

---

## Quick Start

### 1. Set up the environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download dataset → place in `data/raw/`

### 3. Run the pipeline

```bash
# Step 1: Preprocess all audio (VAD + segmentation)
python scripts/01_preprocess.py

# Step 2: Train speech and speaker GMM models
python scripts/02_train_models.py

# Step 3: Generate forged audio dataset
python scripts/03_generate_forgeries.py

# Step 4: Full evaluation (TP/FP/TN/FN, plots)
python scripts/04_evaluate.py

# Optional: Test a single audio file interactively
python scripts/05_demo.py --audio path/to/audio.wav --speaker S01
```

### 4. Launch Jupyter notebooks

```bash
jupyter notebook notebooks/
```

---

## Pipeline Architecture

```
Input Audio
    │
    ▼ VAD (Volume + Zero-Crossing Rate)
[Silence Removal & Word Boundary Detection]
    │
    ▼ MFCC Extraction (13 coeffs + Δ + ΔΔ = 39 features)
[Feature Vectors per Word Segment]
    │
    ├──▶ Speech GMM (256 mixtures) → Word Identity (digit 1–9)
    │
    └──▶ Speaker GMMs (32 mixtures) → Log-Likelihood per speaker
              │
              ▼
    [Compare LLH: claimed speaker vs. all others]
              │
    ┌─────────┴─────────┐
    │                   │
  GENUINE             FORGED
(all segments       (any segment ≠
= claimed speaker)   claimed speaker)
```

---

## Key Parameters (from paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| Sample rate | 16,000 Hz | Paper |
| Frame size | 20 ms (320 samples) | Paper |
| MFCC coefficients | 13 | Paper |
| Speech GMM mixtures | 256 | Table 3 |
| Speaker GMM mixtures | 32 | Table 4 |
| VAD threshold | 3% of (Vmax – Vmin) + Vmin | Eq. 2 |
| Min voiced frames/word | 10 | Paper |

---

## Results (Paper Benchmark)

| Metric | Genuine Audio | Forged Audio |
|--------|-------------|--------------|
| Best accuracy | 100% | 100% |
| Worst accuracy | 92.50% | 99.90% |
| Average | 96.6% | 99.98% |

---

## Theory

### Voice Activity Detection (VAD)
- **Volume** of a frame: V = Σ|aᵢ| for all samples aᵢ
- **Adaptive threshold**: adaptiveTh = 3%(Vmax − Vmin) + Vmin
- ≥10 consecutive voiced frames → one word

### MFCC (Mel-Frequency Cepstral Coefficients)
Simulates human auditory perception:
1. Pre-emphasis filter
2. Framing + Hamming window
3. FFT → power spectrum
4. Mel filter bank (26 filters)
5. Log → DCT → 13 MFCCs
6. Delta and delta-delta features appended

### GMM (Gaussian Mixture Model)
- Models distribution of MFCC features as a weighted sum of Gaussians
- p(X|Θ) = Σ wᵢ · g(X|μᵢ, Σᵢ)
- Parameters estimated via **EM (Expectation-Maximization)** algorithm
- **Log-likelihood** score: log p(X|Θ_speaker) used for decision

### Forgery Detection Decision
```
For each word segment X in test audio:
    1. Identify word w = argmax_word log p(X | Θ_word)
    2. Identify speaker s = argmax_speaker log p(X | Θ_{w,speaker})
    3. If s ≠ claimed_speaker → TAMPERED SEGMENT

If any segment is TAMPERED → audio is FORGED
```

---

## License

Dataset: Open Database License (ODbL) — University of Stirling  
Research Paper: © 2021 Elsevier Ltd.  
This implementation: For academic/educational purposes only.
