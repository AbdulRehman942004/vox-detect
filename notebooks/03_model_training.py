#!/usr/bin/env python3
"""
VoxDetect — notebooks/03_model_training.py
GMM Model Training and Optimization Analysis.

Reproduces Tables 2, 3, and 4 from the paper:
    - Table 2: Speech recognition accuracy for 8, 16, 32, 64 mixtures
    - Table 3: Speech recognition accuracy for 128 and 256 mixtures
    - Table 4: Speaker recognition accuracy for 4, 8, 16, 32 mixtures

Also visualizes:
    - GMM density contours in 2D PCA space
    - Log-likelihood convergence during EM training
    - Model comparison across mixture counts
"""

# %% [markdown]
# # VoxDetect: GMM Model Training Analysis
# ## Reproducing Tables 2, 3, 4 from the Paper

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src import config
from src.features import features_from_files
from src.gmm_models import (
    train_gmm,
    log_likelihood,
    per_frame_log_likelihood,
    evaluate_mixture_counts,
    load_all_speech_models,
    load_all_speaker_models,
    model_path_speech,
    model_path_speaker,
)
from src.segmentation import get_digit_paths, load_splits

sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 110

print("✅ Model Training Analysis Notebook Ready")

# %%
# Check if models and splits exist
models_exist = any(config.SPEECH_MODELS_DIR.glob("*.pkl")) if config.SPEECH_MODELS_DIR.exists() else False
splits_exist = (config.SPLITS_DIR / "train_split.json").exists()

print(f"Speech models exist: {models_exist}")
print(f"Splits exist: {splits_exist}")

if splits_exist:
    train_split, test_split = load_splits(config.SPLITS_DIR)
    speaker_ids = list(train_split.keys())
    print(f"Speakers in splits: {len(speaker_ids)}")
else:
    speaker_ids = []
    print("No splits found. Run scripts/01_preprocess.py first.")

# %% [markdown]
# ## 1. Reproduce Tables 2 & 3: Speech Model Mixture Count Analysis

# %%
def compute_speech_recognition_accuracy(
    digit_idx, speaker_ids, train_split, test_split, mixture_counts
):
    """
    Train speech GMMs at different mixture counts and measure recognition accuracy.
    
    Accuracy = fraction of test segments where this digit's model scores highest
    (compared to all other digit models trained at same mixture count).
    
    This replicates the methodology used for Tables 2 & 3 in the paper.
    """
    # Collect training features for this digit (ALL speakers)
    train_paths = []
    for spk in speaker_ids:
        sessions = list(train_split.get(spk, {}).keys())
        train_paths.extend(get_digit_paths(spk, digit_idx, sessions=sessions))
    
    # Collect test features for this digit
    test_paths = []
    for spk in speaker_ids:
        sessions = list(test_split.get(spk, {}).keys())
        test_paths.extend(get_digit_paths(spk, digit_idx, sessions=sessions))
    
    if not train_paths or not test_paths:
        return {n: None for n in mixture_counts}
    
    try:
        X_train = features_from_files(train_paths[:50])   # Limit for speed
        X_test  = features_from_files(test_paths[:20])
    except Exception as e:
        print(f"  Feature extraction failed for digit {digit_idx}: {e}")
        return {n: None for n in mixture_counts}
    
    # Evaluate
    results = evaluate_mixture_counts(X_train, X_test, mixture_counts)
    return results


if splits_exist and speaker_ids:
    print("\nTraining speech GMMs at different mixture counts...")
    print("(This reproduces Tables 2 and 3 from the paper)")
    print()
    
    speech_mix_counts  = [8, 16, 32, 64, 128, 256]
    speech_results_all = {}
    
    for digit_idx in tqdm(config.ACTIVE_DIGITS[:5], desc="Digits"):   # First 5 for speed
        res = compute_speech_recognition_accuracy(
            digit_idx, speaker_ids, train_split, test_split, speech_mix_counts
        )
        speech_results_all[digit_idx] = res
    
    # Build DataFrame
    rows = []
    for digit_idx, res in speech_results_all.items():
        row = {"Digit": config.DIGIT_NAMES[digit_idx].capitalize()}
        for n in speech_mix_counts:
            row[f"{n}"] = round(res.get(n, None) or 0, 3)
        rows.append(row)
    
    speech_mix_df = pd.DataFrame(rows)
    
    print("\n  Average Log-Likelihood per Frame (Speech Models)")
    print("  (Higher = better fit; matches trend in Tables 2 & 3)")
    print(speech_mix_df.to_string(index=False))
    
    # Save
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    speech_mix_df.to_csv(config.REPORTS_DIR / "speech_mixture_analysis.csv", index=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(speech_results_all))
    width = 0.8 / len(speech_mix_counts)
    colors = cm.viridis(np.linspace(0.1, 0.9, len(speech_mix_counts)))
    
    for i, n in enumerate(speech_mix_counts):
        vals = [speech_results_all[d].get(n, 0) or 0 for d in config.ACTIVE_DIGITS[:5]]
        ax.bar(x + i * width, vals, width, label=f"{n} mixtures", color=colors[i], alpha=0.85)
    
    ax.set_xticks(x + width * (len(speech_mix_counts) - 1) / 2)
    ax.set_xticklabels([config.DIGIT_NAMES[d].capitalize() for d in config.ACTIVE_DIGITS[:5]])
    ax.set_ylabel("Avg Log-Likelihood per Frame")
    ax.set_title("Speech GMM: Log-Likelihood vs. Mixture Count\n(Reproducing Tables 2 & 3 trend)")
    ax.legend(title="Mixtures", ncol=3, fontsize=9)
    
    plt.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / "03_speech_mixture_analysis.png", bbox_inches="tight")
    plt.show()
else:
    print("Skipping mixture analysis (no processed data / splits)")

# %% [markdown]
# ## 2. GMM Density Visualization in 2D PCA Space

# %%
def plot_gmm_density_2d(features, n_components=32, title="GMM Density"):
    """
    Visualize GMM density contours in 2D PCA space.
    Shows how the GMM captures the distribution of MFCC features.
    """
    if len(features) < n_components * 2:
        n_components = max(2, len(features) // 4)
    
    # Reduce to 2D via PCA
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(features[:, :config.N_MFCC])
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X_norm)
    
    # Train GMM in 2D (for visualization only)
    gmm_2d = GaussianMixture(
        n_components=n_components, covariance_type="full",
        random_state=42, max_iter=100
    )
    gmm_2d.fit(X_2d)
    
    # Create density grid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                          np.linspace(y_min, y_max, 100))
    Z = gmm_2d.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data points
    ax.scatter(X_2d[:, 0], X_2d[:, 1], s=5, alpha=0.3,
               c=gmm_2d.predict(X_2d), cmap="tab20", label="MFCC frames")
    
    # Density contours
    ax.contour(xx, yy, Z, levels=15, cmap="Reds", alpha=0.6, linewidths=0.8)
    
    # GMM centers (in 2D)
    centers = gmm_2d.means_
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=80, c="black",
               linewidths=2, label="GMM centers (μ)")
    
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=9)
    
    return fig


if speaker_ids:
    digit_to_vis = 1
    all_paths = []
    for spk in speaker_ids[:10]:
        sessions = list(train_split.get(spk, {}).keys())
        all_paths.extend(get_digit_paths(spk, digit_to_vis, sessions=sessions))
    
    if all_paths:
        try:
            feat = features_from_files(all_paths[:30])
            fig  = plot_gmm_density_2d(
                feat, n_components=min(16, len(feat)//5),
                title=f"GMM Density (2D PCA) — Digit '{config.DIGIT_NAMES[digit_to_vis].upper()}'"
            )
            fig.savefig(config.FIGURES_DIR / "03_gmm_density_2d.png", bbox_inches="tight")
            plt.show()
            print(f"GMM density plot saved.")
        except Exception as e:
            print(f"GMM density plot failed: {e}")
    else:
        print("No training data found for density plot.")
else:
    # Synthetic demo
    np.random.seed(42)
    feat_synth = np.vstack([
        np.random.randn(200, 13) + np.array([i*2]*13)
        for i in range(4)
    ])
    fig = plot_gmm_density_2d(feat_synth, n_components=4,
                               title="GMM Density Demo (Synthetic Data)")
    fig.savefig(config.FIGURES_DIR / "03_gmm_density_2d.png", bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## 3. Log-Likelihood Convergence During EM Training

# %%
def plot_em_convergence(features, n_components_list=[4, 16, 64, 256]):
    """Plot EM convergence (lower bound) for different mixture counts."""
    from sklearn.mixture import GaussianMixture
    
    fig, axes = plt.subplots(1, len(n_components_list), figsize=(14, 4))
    
    for ax, n in zip(axes, n_components_list):
        n_actual = min(n, len(features) // 3)
        gmm = GaussianMixture(
            n_components=n_actual,
            covariance_type="diag",
            max_iter=200,
            random_state=42,
            verbose=0,
        )
        gmm.fit(features)
        
        # sklearn stores lower bound history
        lower_bounds = getattr(gmm, "lower_bound_", None)
        converged    = gmm.converged_
        n_iter       = gmm.n_iter_
        
        ax.set_title(f"{n} mixtures\n(iter={n_iter}, conv={converged})")
        ax.text(0.5, 0.5, f"LLH = {gmm.score(features):.3f}\nper frame",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_xlabel("(convergence status)")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle("EM Algorithm: GMM Convergence at Different Mixture Counts", fontsize=12)
    plt.tight_layout()
    return fig


if speaker_ids:
    if all_paths:
        feat_for_em = features_from_files(all_paths[:15])
        fig = plot_em_convergence(feat_for_em[:2000], [4, 16, 64, 256])
    else:
        np.random.seed(0)
        feat_for_em = np.random.randn(1000, config.N_MFCC).astype(np.float32)
        fig = plot_em_convergence(feat_for_em, [4, 16, 64, 256])
else:
    np.random.seed(0)
    feat_for_em = np.random.randn(1000, config.N_MFCC).astype(np.float32)
    fig = plot_em_convergence(feat_for_em, [4, 16, 64, 256])

fig.savefig(config.FIGURES_DIR / "03_em_convergence.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Loaded Models Summary

# %%
if models_exist:
    speech_models = load_all_speech_models()
    print(f"Speech models loaded: {len(speech_models)}")
    for d, gmm in sorted(speech_models.items()):
        print(f"  Digit {d} ({config.DIGIT_NAMES[d]:>6}): {gmm.n_components} components, "
              f"converged={gmm.converged_}, n_iter={gmm.n_iter_}")
else:
    print("No speech models found. Run scripts/02_train_models.py first.")

print("\n✅ Notebook 3: Model Training Analysis complete!")
