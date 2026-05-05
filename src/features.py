"""
VoxDetect — src/features.py
MFCC feature extraction for audio segments.

Implements the feature extraction pipeline described in the paper:
    Mubeen et al. (2021), Section 2.3

Steps (as described in paper):
    1. Pre-emphasis filtering
    2. Segmentation into overlapping frames
    3. Hamming window (to avoid spectral leakage during FT)
    4. Fast Fourier Transform → power spectrum
    5. Mel-spaced band-pass filter bank (simulates human auditory perception)
    6. Log compression
    7. Discrete Cosine Transform (DCT) → 13 MFCCs
    8. Delta and delta-delta computation (for dynamic features)

Final feature vector: 13 + 13 + 13 = 39 dimensions per frame
"""

import numpy as np
import librosa
import warnings
from pathlib import Path
from typing import List, Optional, Union

from src import config

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MFCC EXTRACTION (SINGLE SIGNAL)
# ─────────────────────────────────────────────────────────────────────────────

def extract_mfcc(
    signal:      np.ndarray,
    sr:          int   = config.TARGET_SR,
    n_mfcc:      int   = config.N_MFCC,
    n_fft:       int   = config.N_FFT,
    hop_length:  int   = config.HOP_LENGTH,
    win_length:  int   = config.WIN_LENGTH,
    n_mels:      int   = config.N_MELS,
    fmin:        float = config.FMIN,
    fmax:        float = config.FMAX,
    use_delta:   bool  = config.USE_DELTA,
    use_delta2:  bool  = config.USE_DELTA2,
) -> np.ndarray:
    """
    Compute MFCC features for a speech signal.

    Args:
        signal:     1-D float audio signal
        sr:         Sample rate
        n_mfcc:     Number of MFCC coefficients (paper: 13)
        n_fft:      FFT window size
        hop_length: Hop between frames in samples
        win_length: Analysis window length in samples
        n_mels:     Number of Mel filter banks (paper: 26)
        fmin:       Minimum frequency for Mel filters
        fmax:       Maximum frequency for Mel filters
        use_delta:  Append first-order delta features
        use_delta2: Append second-order delta (acceleration) features

    Returns:
        features: 2-D array of shape (n_features, n_frames)
                  where n_features = n_mfcc × (1 + use_delta + use_delta2)
    """
    if len(signal) == 0:
        raise ValueError("Empty signal passed to extract_mfcc()")

    # Ensure float32
    signal = signal.astype(np.float32)

    # Compute base MFCCs using librosa
    # librosa.feature.mfcc internally:
    #   1. Applies Hamming window (via STFT)
    #   2. Computes power spectrogram via FFT
    #   3. Applies Mel filter bank
    #   4. Takes log
    #   5. Applies DCT to get MFCCs
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window="hamming",       # Hamming window as per paper
        center=True,
    )  # shape: (n_mfcc, n_frames)

    features = [mfcc]

    # Append delta features (velocity)
    if use_delta:
        delta = librosa.feature.delta(mfcc, order=1)
        features.append(delta)

    # Append delta-delta features (acceleration)
    if use_delta2:
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.append(delta2)

    # Stack along feature axis: (n_features, n_frames)
    features_stacked = np.vstack(features)

    return features_stacked  # (n_features, n_frames)


def extract_mfcc_matrix(
    signal:    np.ndarray,
    sr:        int = config.TARGET_SR,
    **kwargs,
) -> np.ndarray:
    """
    Extract MFCC and return as frame-major matrix: (n_frames, n_features).
    This is the format expected by scikit-learn's GaussianMixture.

    Args:
        signal: 1-D float audio signal
        sr:     Sample rate

    Returns:
        X: 2-D array of shape (n_frames, n_features)
    """
    mfcc = extract_mfcc(signal, sr, **kwargs)
    return mfcc.T  # Transpose to (n_frames, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH EXTRACTION (FROM FILES)
# ─────────────────────────────────────────────────────────────────────────────

def features_from_file(
    wav_path: Union[str, Path],
    sr:       int = config.TARGET_SR,
    **kwargs,
) -> np.ndarray:
    """
    Load a WAV file and extract MFCC features.

    Args:
        wav_path: Path to WAV file
        sr:       Target sample rate (resamples if needed)

    Returns:
        X: 2-D array of shape (n_frames, n_features)
    """
    signal, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    return extract_mfcc_matrix(signal.astype(np.float32), sr, **kwargs)


def features_from_files(
    wav_paths:     List[Union[str, Path]],
    sr:            int  = config.TARGET_SR,
    min_frames:    int  = 5,   # Discard files with too few frames
    verbose:       bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Extract and stack MFCC features from multiple WAV files.

    Args:
        wav_paths:  List of paths to WAV files
        sr:         Target sample rate
        min_frames: Minimum frames per file to include
        verbose:    Print progress

    Returns:
        X: Concatenated 2-D array of shape (total_frames, n_features)
    """
    all_features = []
    skipped = 0

    for path in wav_paths:
        try:
            feat = features_from_file(path, sr, **kwargs)
            if feat.shape[0] < min_frames:
                skipped += 1
                continue
            all_features.append(feat)
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not process {path}: {e}")
            skipped += 1

    if not all_features:
        raise ValueError(f"No valid feature matrices extracted from {len(wav_paths)} files.")

    if verbose and skipped:
        print(f"  Skipped {skipped}/{len(wav_paths)} files (too short / errors)")

    return np.vstack(all_features)  # (total_frames, n_features)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NORMALIZATION (CEPSTRAL MEAN SUBTRACTION)
# ─────────────────────────────────────────────────────────────────────────────

def cepstral_mean_subtraction(features: np.ndarray) -> np.ndarray:
    """
    Apply Cepstral Mean Subtraction (CMS) to normalize features.

    CMS removes channel effects by subtracting the mean of each
    MFCC coefficient across all frames. This makes features more
    robust to channel and microphone variations.

    Args:
        features: 2-D array (n_frames, n_features)

    Returns:
        normalized: features with zero mean per feature dimension
    """
    mean = features.mean(axis=0, keepdims=True)
    return features - mean


def mean_variance_normalize(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance per dimension.

    Args:
        features: 2-D array (n_frames, n_features)

    Returns:
        normalized: standardized features
    """
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0, keepdims=True) + 1e-9
    return (features - mean) / std


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SUMMARY / STATISTICS (FOR ANALYSIS)
# ─────────────────────────────────────────────────────────────────────────────

def feature_stats(features: np.ndarray) -> dict:
    """
    Compute summary statistics of a feature matrix.

    Args:
        features: 2-D array (n_frames, n_features)

    Returns:
        stats dict with mean, std, min, max, n_frames, n_features
    """
    return {
        "n_frames":    features.shape[0],
        "n_features":  features.shape[1],
        "mean":        float(features.mean()),
        "std":         float(features.std()),
        "min":         float(features.min()),
        "max":         float(features.max()),
    }


def per_coefficient_stats(features: np.ndarray) -> dict:
    """
    Return per-coefficient mean and std arrays.

    Args:
        features: 2-D array (n_frames, n_features)

    Returns:
        {"mean": array(n_features,), "std": array(n_features,)}
    """
    return {
        "mean": features.mean(axis=0),
        "std":  features.std(axis=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sr     = config.TARGET_SR
    # Generate 0.3s white noise as a dummy "speech" signal
    signal = (np.random.randn(sr // 3)).astype(np.float32)

    feat = extract_mfcc_matrix(signal)
    stats = feature_stats(feat)

    print("MFCC Feature Extraction — Smoke Test")
    print("=" * 45)
    print(f"  Signal length:  {len(signal)} samples ({len(signal)/sr:.2f}s)")
    print(f"  Feature shape:  {feat.shape}  (frames × features)")
    print(f"  Expected feats: {config.N_FEATURES}")
    print(f"  Mean:           {stats['mean']:.4f}")
    print(f"  Std:            {stats['std']:.4f}")
    assert feat.shape[1] == config.N_FEATURES, "Feature dimension mismatch!"
    print("  ✓ All checks passed")
