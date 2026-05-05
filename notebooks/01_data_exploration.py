#!/usr/bin/env python3
"""
VoxDetect — notebooks/01_data_exploration.py
Run as a script or open as a notebook (nbformat compatible)

Exploratory Data Analysis of the Merged Arabic Corpus of Isolated Words dataset.
Explores:
    - Dataset structure and speaker distribution
    - Audio waveforms and spectrograms
    - Duration statistics per digit per speaker
    - VAD visualization on sample recordings
"""

# %% [markdown]
# # VoxDetect: Data Exploration
# ## Merged Arabic Corpus of Isolated Words
# ### Based on: Mubeen et al. (2021) — Audio Forgery Detection System

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import librosa
import librosa.display
import pandas as pd

from src import config
from src.vad import load_and_vad, extract_word_signals, vad_summary, compute_frame_volume, compute_adaptive_threshold, VAD_FRAME_SAMPLES
from src.segmentation import discover_dataset, get_digit_paths

sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 110

print("✅ Imports OK")
print(f"Raw data directory: {config.RAW_DIR}")

# %% [markdown]
# ## 1. Dataset Overview

# %%
print("=" * 55)
print("  Dataset: Merged Arabic Corpus of Isolated Words")
print("=" * 55)
print(f"  Expected speakers : 50 (S01–S50)")
print(f"  Sessions/speaker  : ~10")
print(f"  File type         : {{speaker_id}}.{{session_id}}.digits.wav")
print(f"  Sampling rate     : 44,100 Hz (resampled to {config.TARGET_SR:,} Hz)")
print(f"  Language          : Arabic (digits 0–9)")
print(f"  Total size        : ~1 GB")
print()

if config.RAW_DIR.exists():
    speaker_files = discover_dataset(config.RAW_DIR)
    print(f"  Found {len(speaker_files)} speakers in {config.RAW_DIR}")
    
    # Count total files
    total_files = sum(len(v) for v in speaker_files.items())
    
    # Speaker gender breakdown (from dataset docs)
    females = ["S11", "S36", "S44"]
    males   = [s for s in speaker_files.keys() if s not in females]
    
    print(f"  Male speakers   : {len(males)}")
    print(f"  Female speakers : {len(females)} ({', '.join(females)})")
else:
    print(f"  ⚠️  Dataset not found at {config.RAW_DIR}")
    print("  Please download from Kaggle first.")

# %% [markdown]
# ## 2. Waveform + Spectrogram Visualization

# %%
def plot_waveform_and_spectrogram(audio_path, title="Audio Sample"):
    """Plot waveform, spectrogram, and Mel spectrogram for an audio file."""
    signal, sr = librosa.load(str(audio_path), sr=config.TARGET_SR, mono=True)
    
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(3, 1, hspace=0.4)
    
    t = np.linspace(0, len(signal) / sr, len(signal))
    
    # ── Waveform ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, signal, color="#2196F3", linewidth=0.5, alpha=0.8)
    ax1.set_title(f"Waveform — {title}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_xlim([0, t[-1]])
    
    # ── STFT Spectrogram ──
    ax2 = fig.add_subplot(gs[1])
    D    = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=512)), ref=np.max)
    img  = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz",
                                     hop_length=160, ax=ax2, cmap="magma")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB", label="dB")
    ax2.set_title("STFT Spectrogram")
    
    # ── Mel Spectrogram ──
    ax3 = fig.add_subplot(gs[2])
    M    = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=26, hop_length=160)
    M_db = librosa.power_to_db(M, ref=np.max)
    img2 = librosa.display.specshow(M_db, sr=sr, x_axis="time", y_axis="mel",
                                     hop_length=160, ax=ax3, cmap="viridis")
    fig.colorbar(img2, ax=ax3, format="%+2.0f dB", label="dB")
    ax3.set_title("Mel Spectrogram (26 filters)")
    
    plt.suptitle(title, fontsize=14, y=1.01)
    return fig


# Try with actual data or show a synthetic example
if config.RAW_DIR.exists():
    # Find first available digits file
    first_spk = sorted(discover_dataset(config.RAW_DIR).items())[0]
    sample_path = first_spk[1][0]
    fig = plot_waveform_and_spectrogram(sample_path, title=f"Speaker {first_spk[0]} — Digits Recording")
else:
    # Synthetic example
    sr     = config.TARGET_SR
    t      = np.linspace(0, 2.5, sr * 2 + sr // 2)
    signal = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)
    
    # Fake a WAV file for demo
    import soundfile as sf, tempfile, os
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, signal.astype(np.float32), sr)
    fig = plot_waveform_and_spectrogram(tmp, title="Synthetic Signal (demo)")
    os.unlink(tmp)

config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(config.FIGURES_DIR / "01_waveform_spectrogram.png", bbox_inches="tight")
plt.show()
print("Figure saved.")

# %% [markdown]
# ## 3. VAD Visualization

# %%
def plot_vad_result(audio_path, title="VAD Result"):
    """Show the VAD frame mask overlaid on the waveform."""
    from src.vad import detect_voiced_frames, extract_word_regions
    
    signal, sr = librosa.load(str(audio_path), sr=config.TARGET_SR, mono=True)
    signal     = signal.astype(np.float32)
    
    frame_size   = config.VAD_FRAME_SAMPLES
    volumes      = compute_frame_volume(signal, frame_size)
    adap_th      = compute_adaptive_threshold(volumes)
    voiced_mask  = detect_voiced_frames(signal, frame_size)
    regions      = extract_word_regions(signal)
    words        = extract_word_signals(signal, regions)
    
    # Build frame-level coloring
    n_frames     = len(volumes)
    frame_times  = np.arange(n_frames) * frame_size / sr
    sample_times = np.arange(len(signal)) / sr
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    
    # Waveform + voiced regions highlighted
    ax = axes[0]
    ax.plot(sample_times, signal, color="#555", linewidth=0.4, alpha=0.7, label="Signal")
    for r in regions:
        s = r.start_sample / sr
        e = r.end_sample / sr
        ax.axvspan(s, e, alpha=0.25, color="#2196F3", label="Word region")
    ax.set_title(f"Waveform + Detected Word Regions (n={len(regions)})")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8)
    
    # Volume per frame + threshold
    ax2 = axes[1]
    ax2.bar(frame_times, volumes, width=frame_size/sr, color="#FF9800", alpha=0.7, label="Volume")
    ax2.axhline(adap_th, color="red", linestyle="--", linewidth=1.5,
                label=f"Adaptive threshold (3% rule) = {adap_th:.4f}")
    ax2.set_title("Frame Volume + Adaptive Threshold (Eq. 1, 2 from paper)")
    ax2.set_ylabel("Volume (Σ|aᵢ|)")
    ax2.legend(fontsize=9)
    
    # Voiced mask
    ax3 = axes[2]
    ax3.fill_between(frame_times, voiced_mask.astype(int), step="post",
                     alpha=0.7, color="#4CAF50", label="Voiced")
    ax3.fill_between(frame_times, (~voiced_mask).astype(int), step="post",
                     alpha=0.3, color="#F44336", label="Unvoiced/Silence")
    ax3.set_ylim([-0.1, 1.5])
    ax3.set_title("VAD Binary Mask (1=Voiced, 0=Silence)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Voiced")
    ax3.legend(fontsize=9)
    
    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig, words


if config.RAW_DIR.exists():
    fig, words = plot_vad_result(sample_path, title=f"VAD — {first_spk[0]}")
else:
    import soundfile as sf, tempfile, os
    # Create synthetic multi-word signal
    sr = config.TARGET_SR
    silence = np.zeros(sr // 4, dtype=np.float32)
    words_list = []
    for freq in [300, 500, 700, 400, 600]:
        dur = sr // 5
        w = (np.sin(2 * np.pi * freq * np.arange(dur) / sr) * 0.8).astype(np.float32)
        words_list.extend([w, silence])
    sig = np.concatenate(words_list)
    tmp = tempfile.mktemp(suffix=".wav")
    sf.write(tmp, sig, sr)
    fig, words = plot_vad_result(tmp, title="VAD on Synthetic Signal (demo)")
    os.unlink(tmp)

fig.savefig(config.FIGURES_DIR / "01_vad_visualization.png", bbox_inches="tight")
plt.show()
print(f"Detected {len(words)} word segments")

# %% [markdown]
# ## 4. MFCC Visualization

# %%
from src.features import extract_mfcc

def plot_mfcc(word_signal, sr=config.TARGET_SR, title="MFCC Features"):
    """Visualize MFCC + delta + delta-delta for one word segment."""
    mfcc = extract_mfcc(word_signal, sr, use_delta=True, use_delta2=True)
    # mfcc shape: (39, n_frames)
    
    n_coeff = config.N_MFCC  # 13
    base    = mfcc[:n_coeff]
    delta   = mfcc[n_coeff:2*n_coeff]
    delta2  = mfcc[2*n_coeff:]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    for ax, data, label, cmap in zip(
        axes,
        [base, delta, delta2],
        ["MFCC (13 coefficients)", "Delta (velocity)", "Delta-Delta (acceleration)"],
        ["viridis", "plasma", "inferno"],
    ):
        img = librosa.display.specshow(data, x_axis="frames", ax=ax, cmap=cmap)
        ax.set_title(label)
        ax.set_ylabel("Coefficient")
        fig.colorbar(img, ax=ax, label="Value")
    
    axes[-1].set_xlabel("Frame")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig

if words:
    # Use first detected word
    first_word = words[0]
else:
    first_word = (np.sin(2 * np.pi * 440 * np.arange(config.TARGET_SR // 5) /
                         config.TARGET_SR) * 0.8).astype(np.float32)

fig = plot_mfcc(first_word, title="MFCC Features for a Single Digit Word")
fig.savefig(config.FIGURES_DIR / "01_mfcc_visualization.png", bbox_inches="tight")
plt.show()
print(f"MFCC shape: {extract_mfcc(first_word).shape} (features × frames)")

# %% [markdown]
# ## 5. Duration Statistics

# %%
if config.PROCESSED_DIR.exists() and any(config.PROCESSED_DIR.iterdir()):
    durations = []
    speakers = list((config.PROCESSED_DIR).iterdir())[:10]  # First 10 speakers
    
    for spk_dir in speakers:
        if not spk_dir.is_dir():
            continue
        for digit_dir in sorted(spk_dir.iterdir()):
            if not digit_dir.is_dir():
                continue
            digit_idx = int(digit_dir.name.split("_")[1])
            for wav_path in sorted(digit_dir.glob("*.wav")):
                sig, _ = librosa.load(str(wav_path), sr=config.TARGET_SR)
                dur_ms  = len(sig) / config.TARGET_SR * 1000
                durations.append({
                    "speaker": spk_dir.name,
                    "digit":   config.DIGIT_NAMES[digit_idx].capitalize(),
                    "dur_ms":  dur_ms,
                })
    
    if durations:
        df = pd.DataFrame(durations)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot per digit
        sns.boxplot(data=df, x="digit", y="dur_ms", ax=axes[0],
                    palette="Set2", order=[d.capitalize() for d in config.DIGIT_NAMES[1:]])
        axes[0].set_title("Word Duration Distribution per Digit")
        axes[0].set_xlabel("Digit")
        axes[0].set_ylabel("Duration (ms)")
        axes[0].tick_params(axis="x", rotation=30)
        
        # Histogram of all durations
        axes[1].hist(df["dur_ms"], bins=40, color="#2196F3", alpha=0.8, edgecolor="white")
        axes[1].axvline(df["dur_ms"].mean(), color="red", linestyle="--",
                        label=f"Mean: {df['dur_ms'].mean():.0f} ms")
        axes[1].set_title("Overall Word Duration Distribution")
        axes[1].set_xlabel("Duration (ms)")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        
        plt.tight_layout()
        fig.savefig(config.FIGURES_DIR / "01_duration_stats.png", bbox_inches="tight")
        plt.show()
        print(f"\nDuration stats (ms):")
        print(df["dur_ms"].describe().round(1))
    else:
        print("No processed files found. Run 01_preprocess.py first.")
else:
    print(f"Processed directory empty. Run scripts/01_preprocess.py first.")
    print(f"Expected: {config.PROCESSED_DIR}")

print("\n✅ Notebook 1: Data Exploration complete!")
print(f"   Figures saved to: {config.FIGURES_DIR}")
