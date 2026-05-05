"""
VoxDetect — src/vad.py
Voice Activity Detection (VAD) module.

Implements the volume + zero-crossing-rate based VAD described in the paper:
    Mubeen et al. (2021), Section 2.2

Key equations:
    V = Σ|aᵢ|                                  (Eq. 1 — frame volume)
    adaptiveTh = 3%(Vmax − Vmin) + Vmin        (Eq. 2 — adaptive threshold)

A segment is voiced if V > adaptiveTh. ≥10 consecutive voiced frames = one word.
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src import config


@dataclass
class VoicedRegion:
    """Represents a voiced (speech) region in a signal."""
    start_sample: int
    end_sample:   int
    start_frame:  int
    end_frame:    int
    word_label:   Optional[int] = None   # 0-9 if known

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def duration_sec(self) -> float:
        return self.duration_samples / config.TARGET_SR


# ─────────────────────────────────────────────────────────────────────────────
# FRAME-LEVEL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_frame_volume(signal: np.ndarray, frame_size: int) -> np.ndarray:
    """
    Compute the volume (sum of absolute amplitudes) for each frame.

    V = Σ|aᵢ|  (Eq. 1 in paper)

    Args:
        signal:     1-D audio signal (float32, normalized to [-1, 1])
        frame_size: Number of samples per frame

    Returns:
        volumes: 1-D array of volumes, one per frame
    """
    n_frames = len(signal) // frame_size
    volumes  = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame       = signal[i * frame_size : (i + 1) * frame_size]
        volumes[i]  = np.sum(np.abs(frame))
    return volumes


def compute_frame_zcr(signal: np.ndarray, frame_size: int) -> np.ndarray:
    """
    Compute the zero-crossing rate (ZCR) for each frame.

    ZCR counts how many times the signal crosses zero amplitude per frame.
    High ZCR indicates unvoiced/noisy frames; low ZCR indicates voiced speech.

    Args:
        signal:     1-D audio signal
        frame_size: Number of samples per frame

    Returns:
        zcr_values: Normalized ZCR (crossings / frame_size) per frame
    """
    n_frames   = len(signal) // frame_size
    zcr_values = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame          = signal[i * frame_size : (i + 1) * frame_size]
        crossings      = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
        zcr_values[i]  = crossings / frame_size
    return zcr_values


def compute_adaptive_threshold(volumes: np.ndarray,
                                pct: float = config.VAD_THRESHOLD_PCT) -> float:
    """
    Compute the adaptive volume threshold.

    adaptiveTh = pct × (Vmax − Vmin) + Vmin   (Eq. 2 in paper)

    Args:
        volumes: Array of frame volumes
        pct:     Percentage (default 3% = 0.03 as per paper)

    Returns:
        threshold: Scalar adaptive threshold value
    """
    v_max = np.max(volumes)
    v_min = np.min(volumes)
    return pct * (v_max - v_min) + v_min


# ─────────────────────────────────────────────────────────────────────────────
# VOICED FRAME DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_voiced_frames(signal: np.ndarray,
                          frame_size:    int   = config.VAD_FRAME_SAMPLES,
                          threshold_pct: float = config.VAD_THRESHOLD_PCT,
                          zcr_threshold: float = config.VAD_ZCR_THRESHOLD
                          ) -> np.ndarray:
    """
    Detect voiced frames using volume + ZCR (as described in paper).

    A frame is voiced if:
        volume > adaptive_threshold  AND  zcr < zcr_threshold

    Args:
        signal:        1-D audio signal (float, normalized)
        frame_size:    Samples per frame
        threshold_pct: Adaptive threshold percentage (paper uses 3%)
        zcr_threshold: ZCR upper bound for voiced frames

    Returns:
        voiced_mask: Boolean array, True = voiced frame
    """
    volumes   = compute_frame_volume(signal, frame_size)
    zcr_vals  = compute_frame_zcr(signal, frame_size)
    adap_th   = compute_adaptive_threshold(volumes, threshold_pct)

    voiced_mask = (volumes > adap_th) & (zcr_vals < zcr_threshold)
    return voiced_mask


# ─────────────────────────────────────────────────────────────────────────────
# WORD (VOICED REGION) EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _smooth_voiced_mask(voiced_mask: np.ndarray,
                         min_voiced_frames: int = config.VAD_MIN_VOICED_FRAMES,
                         pad_frames:        int = config.VAD_PAD_FRAMES
                         ) -> np.ndarray:
    """
    Post-process the voiced mask:
    1. Remove short voiced runs (< min_voiced_frames) — likely noise
    2. Fill short silence gaps between voiced runs (merge nearby words)
    3. Optionally pad each voiced region by pad_frames on each side

    Args:
        voiced_mask:        Raw boolean frame mask
        min_voiced_frames:  Minimum consecutive voiced frames for a word
        pad_frames:         Frames to add on each side of voiced regions

    Returns:
        smoothed: Processed boolean frame mask
    """
    mask = voiced_mask.copy()
    n    = len(mask)

    # --- Pass 1: Remove short voiced runs ---
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            run_len = j - i
            if run_len < min_voiced_frames:
                mask[i:j] = False
            i = j
        else:
            i += 1

    # --- Pass 2: Fill short silence gaps (≤ min_voiced_frames / 2) ---
    gap_fill = max(1, min_voiced_frames // 2)
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            gap_len = j - i
            # Check if there is a voiced run on both sides
            left_ok  = i > 0 and mask[i - 1]
            right_ok = j < n and mask[j]
            if left_ok and right_ok and gap_len <= gap_fill:
                mask[i:j] = True
            i = j
        else:
            i += 1

    # --- Pass 3: Pad voiced regions ---
    if pad_frames > 0:
        padded = mask.copy()
        for i in range(n):
            if mask[i]:
                lo = max(0, i - pad_frames)
                hi = min(n, i + pad_frames + 1)
                padded[lo:hi] = True
        mask = padded

    return mask


def extract_word_regions(signal: np.ndarray,
                          frame_size:        int   = config.VAD_FRAME_SAMPLES,
                          threshold_pct:     float = config.VAD_THRESHOLD_PCT,
                          zcr_threshold:     float = config.VAD_ZCR_THRESHOLD,
                          min_voiced_frames: int   = config.VAD_MIN_VOICED_FRAMES,
                          pad_frames:        int   = config.VAD_PAD_FRAMES
                          ) -> List[VoicedRegion]:
    """
    Full VAD pipeline: detect and return all voiced word regions.

    Pipeline:
        signal → voiced frame mask → smooth → extract contiguous runs → VoicedRegion list

    Args:
        signal:            1-D float audio signal at TARGET_SR
        frame_size:        Samples per analysis frame
        threshold_pct:     Adaptive threshold percentage
        zcr_threshold:     ZCR upper bound for voiced
        min_voiced_frames: Minimum frames for a word
        pad_frames:        Padding around each word

    Returns:
        regions: List of VoicedRegion objects sorted by start position
    """
    if len(signal) == 0:
        return []

    voiced_mask = detect_voiced_frames(
        signal, frame_size, threshold_pct, zcr_threshold
    )
    smoothed    = _smooth_voiced_mask(voiced_mask, min_voiced_frames, pad_frames)

    regions: List[VoicedRegion] = []
    n = len(smoothed)
    i = 0
    while i < n:
        if smoothed[i]:
            j = i
            while j < n and smoothed[j]:
                j += 1
            # Convert frame indices to sample indices
            start_sample = i * frame_size
            end_sample   = min(j * frame_size, len(signal))
            regions.append(VoicedRegion(
                start_sample=start_sample,
                end_sample=end_sample,
                start_frame=i,
                end_frame=j,
            ))
            i = j
        else:
            i += 1

    return regions


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL API
# ─────────────────────────────────────────────────────────────────────────────

def load_and_vad(audio_path: str,
                 target_sr: int = config.TARGET_SR
                 ) -> Tuple[np.ndarray, List[VoicedRegion]]:
    """
    Load an audio file, resample if needed, and run VAD.

    Args:
        audio_path: Path to .wav file
        target_sr:  Target sample rate in Hz

    Returns:
        signal:  Resampled 1-D float signal
        regions: List of VoicedRegion word segments
    """
    signal, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    signal     = signal.astype(np.float32)
    regions    = extract_word_regions(signal)
    return signal, regions


def extract_word_signals(signal: np.ndarray,
                          regions: List[VoicedRegion]
                          ) -> List[np.ndarray]:
    """
    Slice signal into individual word signals.

    Args:
        signal:  Full audio signal
        regions: Voiced regions from VAD

    Returns:
        word_signals: List of 1-D arrays, one per detected word
    """
    return [signal[r.start_sample:r.end_sample] for r in regions]


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def vad_summary(signal: np.ndarray, regions: List[VoicedRegion]) -> dict:
    """Return a summary dict of VAD results for logging/debugging."""
    total_dur  = len(signal) / config.TARGET_SR
    voiced_dur = sum(r.duration_sec for r in regions)
    return {
        "total_duration_sec":  round(total_dur, 3),
        "voiced_duration_sec": round(voiced_dur, 3),
        "silence_fraction":    round(1 - voiced_dur / max(total_dur, 1e-9), 3),
        "n_words_detected":    len(regions),
        "word_durations_ms":   [round(r.duration_sec * 1000, 1) for r in regions],
    }


if __name__ == "__main__":
    # Quick smoke test with a synthetic sine + silence signal
    import sys

    sr = config.TARGET_SR
    silence = np.zeros(sr // 2, dtype=np.float32)           # 0.5s silence
    word1   = np.sin(2 * np.pi * 440 * np.arange(sr // 4) / sr).astype(np.float32)  # 0.25s tone
    word2   = np.sin(2 * np.pi * 880 * np.arange(sr // 4) / sr).astype(np.float32)
    signal  = np.concatenate([silence, word1, silence, word2, silence])

    regions = extract_word_regions(signal)
    summary = vad_summary(signal, regions)

    print("VAD Smoke Test")
    print("=" * 40)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  Expected 2 words, found: {len(regions)}")
