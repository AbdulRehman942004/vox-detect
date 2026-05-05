#!/usr/bin/env python3
"""
VoxDetect — notebooks/02_feature_analysis.py
Feature Analysis: MFCC properties across speakers and digits.

Explores:
    - MFCC coefficient distributions per digit
    - Inter-speaker MFCC variability
    - Effect of delta/delta-delta features
    - Cepstral mean subtraction effect
    - GMM density visualization on 2D MFCC projections
"""

# %% [markdown]
# # VoxDetect: Feature Analysis
# ## MFCC Feature Extraction and Properties

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import (
    features_from_file,
    features_from_files,
    extract_mfcc,
    extract_mfcc_matrix,
    cepstral_mean_subtraction,
    feature_stats,
)
from src.segmentation import get_digit_paths

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

print("✅ Feature Analysis Notebook Ready")

# %% [markdown]
# ## 1. MFCC Coefficient Statistics Across Speakers

# %%
def analyze_mfcc_per_digit(speaker_ids, digit_idx, n_speakers_to_plot=5):
    """Compute and plot average MFCC per speaker for a given digit."""
    results = []
    
    for spk in speaker_ids[:n_speakers_to_plot]:
        paths = get_digit_paths(spk, digit_idx)
        if not paths:
            continue
        try:
            feat  = features_from_files(paths[:3])   # max 3 files per speaker
            mean  = feat[:, :config.N_MFCC].mean(axis=0)
            std   = feat[:, :config.N_MFCC].std(axis=0)
            results.append((spk, mean, std))
        except Exception:
            pass
    
    return results


if config.PROCESSED_DIR.exists() and any(config.PROCESSED_DIR.iterdir()):
    speaker_dirs = sorted([d for d in config.PROCESSED_DIR.iterdir() if d.is_dir()])
    speaker_ids  = [d.name for d in speaker_dirs]

    # Analyze digit 1 ("one") across speakers
    digit_to_show = 1
    results = analyze_mfcc_per_digit(speaker_ids, digit_to_show, n_speakers_to_plot=8)

    if results:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        colors = cm.Set1(np.linspace(0, 1, len(results)))
        
        # Mean MFCC coefficients
        ax = axes[0]
        for (spk, mean, std), color in zip(results, colors):
            coeff_idxs = np.arange(config.N_MFCC)
            ax.plot(coeff_idxs, mean, marker="o", markersize=4, label=spk,
                    color=color, linewidth=1.5)
            ax.fill_between(coeff_idxs, mean - std, mean + std, alpha=0.1, color=color)
        
        ax.set_title(f"Mean MFCC Coefficients per Speaker — Digit '{config.DIGIT_NAMES[digit_to_show].upper()}'")
        ax.set_xlabel("MFCC Coefficient Index")
        ax.set_ylabel("Mean Value")
        ax.legend(ncol=4, fontsize=8)
        ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        
        # Std of MFCC coefficients  
        ax2 = axes[1]
        all_stds = np.array([std for _, _, std in results])
        avg_std  = all_stds.mean(axis=0)
        
        ax2.bar(np.arange(config.N_MFCC), avg_std, color="#FF9800", alpha=0.8)
        ax2.set_title("Average MFCC Coefficient Std (across speakers) — Intra-speaker variability")
        ax2.set_xlabel("MFCC Coefficient Index")
        ax2.set_ylabel("Std Deviation")
        
        plt.tight_layout()
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / "02_mfcc_per_speaker.png", bbox_inches="tight")
        plt.show()
    else:
        print("No processed data found. Run scripts/01_preprocess.py first.")
else:
    print(f"No processed data at {config.PROCESSED_DIR}")
    speaker_ids = []

# %% [markdown]
# ## 2. PCA Visualization: MFCC Clusters per Digit

# %%
def pca_mfcc_digits(speaker_ids_subset, digits_subset, max_frames_per_class=500):
    """
    Project MFCC features to 2D using PCA to visualize digit separability.
    Each digit should form a distinct cluster if MFCCs are discriminative.
    """
    all_features = []
    all_labels   = []
    
    for digit_idx in digits_subset:
        frames_collected = 0
        for spk in speaker_ids_subset:
            paths = get_digit_paths(spk, digit_idx)
            if not paths:
                continue
            try:
                feat = features_from_files(paths[:2])
                n    = min(len(feat), max_frames_per_class - frames_collected)
                if n <= 0:
                    break
                all_features.append(feat[:n, :config.N_MFCC])
                all_labels.extend([digit_idx] * n)
                frames_collected += n
            except Exception:
                pass
        
        if frames_collected >= max_frames_per_class:
            break
    
    if not all_features:
        return None, None, None
    
    X = np.vstack(all_features)
    y = np.array(all_labels)
    
    # Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    
    # PCA to 2D
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X_norm)
    
    return X_2d, y, pca


if speaker_ids:
    digits_to_show = [1, 2, 3, 4, 5]
    X_2d, y, pca = pca_mfcc_digits(speaker_ids[:20], digits_to_show)
    
    if X_2d is not None:
        fig, ax = plt.subplots(figsize=(9, 7))
        colors_map = cm.tab10(np.linspace(0, 1, len(digits_to_show)))
        
        for digit_idx, color in zip(digits_to_show, colors_map):
            mask = y == digit_idx
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       alpha=0.3, s=8, color=color,
                       label=f"'{config.DIGIT_NAMES[digit_idx].capitalize()}'")
        
        ax.set_title("PCA of MFCC Features: Digit Separability\n"
                     f"(Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.legend(title="Digit", markerscale=3, fontsize=9)
        
        plt.tight_layout()
        fig.savefig(config.FIGURES_DIR / "02_pca_mfcc_digits.png", bbox_inches="tight")
        plt.show()
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    else:
        print("Insufficient data for PCA visualization.")
else:
    print("Skipping PCA (no processed data).")

# %% [markdown]
# ## 3. Effect of Cepstral Mean Subtraction (CMS)

# %%
def demo_cms_effect():
    """Demonstrate how CMS normalizes MFCC features."""
    # Use real or synthetic data
    sr = config.TARGET_SR
    
    if speaker_ids:
        paths = get_digit_paths(speaker_ids[0], 1)
        if paths:
            signal, _ = __import__("librosa").load(str(paths[0]), sr=sr)
        else:
            signal = np.random.randn(sr // 3).astype(np.float32)
    else:
        # Simulate channel effect: add a DC offset + low-freq drift
        t      = np.arange(sr // 3) / sr
        signal = (np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 5 * t) + 0.1).astype(np.float32)
    
    feat_raw = extract_mfcc_matrix(signal)
    feat_cms = cepstral_mean_subtraction(feat_raw)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    
    # Raw MFCC histograms
    axes[0, 0].hist(feat_raw[:, 0], bins=40, color="#2196F3", alpha=0.8)
    axes[0, 0].set_title("Raw MFCC-1 Distribution")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].axvline(feat_raw[:, 0].mean(), color="red", linestyle="--",
                       label=f"Mean={feat_raw[:, 0].mean():.2f}")
    axes[0, 0].legend()
    
    # CMS MFCC histograms
    axes[0, 1].hist(feat_cms[:, 0], bins=40, color="#4CAF50", alpha=0.8)
    axes[0, 1].set_title("CMS MFCC-1 Distribution")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].axvline(feat_cms[:, 0].mean(), color="red", linestyle="--",
                       label=f"Mean={feat_cms[:, 0].mean():.4f}")
    axes[0, 1].legend()
    
    # Coefficient means: raw vs cms
    raw_means = feat_raw[:, :config.N_MFCC].mean(axis=0)
    cms_means = feat_cms[:, :config.N_MFCC].mean(axis=0)
    
    x = np.arange(config.N_MFCC)
    axes[1, 0].bar(x - 0.2, raw_means, 0.35, label="Raw", color="#FF9800", alpha=0.8)
    axes[1, 0].bar(x + 0.2, cms_means, 0.35, label="After CMS", color="#9C27B0", alpha=0.8)
    axes[1, 0].set_title("Mean per MFCC Coefficient: Raw vs. CMS")
    axes[1, 0].set_xlabel("Coefficient Index")
    axes[1, 0].set_ylabel("Mean")
    axes[1, 0].legend()
    
    # Std comparison
    raw_stds = feat_raw[:, :config.N_MFCC].std(axis=0)
    cms_stds = feat_cms[:, :config.N_MFCC].std(axis=0)
    
    axes[1, 1].bar(x - 0.2, raw_stds, 0.35, label="Raw", color="#FF9800", alpha=0.8)
    axes[1, 1].bar(x + 0.2, cms_stds, 0.35, label="After CMS", color="#9C27B0", alpha=0.8)
    axes[1, 1].set_title("Std per MFCC Coefficient: Raw vs. CMS")
    axes[1, 1].set_xlabel("Coefficient Index")
    axes[1, 1].set_ylabel("Std")
    axes[1, 1].legend()
    
    plt.suptitle("Cepstral Mean Subtraction (CMS) Effect", fontsize=13)
    plt.tight_layout()
    return fig

fig = demo_cms_effect()
fig.savefig(config.FIGURES_DIR / "02_cms_effect.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Delta Feature Visualization

# %%
def plot_delta_comparison(signal, sr=config.TARGET_SR):
    """Show MFCC, Δ-MFCC, and ΔΔ-MFCC for a speech segment."""
    signal = signal.astype(np.float32)
    mfcc   = extract_mfcc(signal, sr, use_delta=True, use_delta2=True)
    
    n = config.N_MFCC
    base   = mfcc[:n]
    delta  = mfcc[n:2*n]
    delta2 = mfcc[2*n:]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    for ax, data, label, color in zip(
        axes,
        [base, delta, delta2],
        ["MFCC (static features)", "ΔMFCC (velocity / dynamic)", "ΔΔMFCC (acceleration)"],
        ["viridis", "plasma", "magma"],
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=color, interpolation="nearest")
        ax.set_title(label)
        ax.set_ylabel("Coefficient")
        plt.colorbar(im, ax=ax, fraction=0.02)
    
    axes[-1].set_xlabel("Frame Index")
    plt.suptitle("Static + Dynamic MFCC Features (39-dimensional vector)", fontsize=13)
    plt.tight_layout()
    return fig

if speaker_ids:
    paths = get_digit_paths(speaker_ids[0], 1)
    if paths:
        import librosa as _librosa
        sig, _ = _librosa.load(str(paths[0]), sr=config.TARGET_SR)
        sig = sig.astype(np.float32)
    else:
        sig = (np.sin(2*np.pi*440*np.arange(config.TARGET_SR//4)/config.TARGET_SR)).astype(np.float32)
else:
    sig = (np.sin(2*np.pi*440*np.arange(config.TARGET_SR//4)/config.TARGET_SR)).astype(np.float32)

fig = plot_delta_comparison(sig)
fig.savefig(config.FIGURES_DIR / "02_delta_features.png", bbox_inches="tight")
plt.show()

print("\n✅ Notebook 2: Feature Analysis complete!")
print(f"   All figures saved to: {config.FIGURES_DIR}")
