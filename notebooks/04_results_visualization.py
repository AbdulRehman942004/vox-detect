#!/usr/bin/env python3
"""
VoxDetect — notebooks/04_results_visualization.py
Full Results Visualization — reproducing all paper figures.

Requires:
    - Trained models (scripts/02_train_models.py)
    - Generated forgeries (scripts/03_generate_forgeries.py)
    - Evaluation complete (scripts/04_evaluate.py)

Reproduces:
    - Fig. 4: LLH of digit across all speakers
    - Fig. 5: LLH values of all digits vs all impostors
    - Fig. 6: Distribution of LLH values (genuine vs impostor)
    - Fig. 7: PDF of LLH values for genuine/impostor
    - Tables 7–10: TP/FP/TN/FN per sentence
"""

# %% [markdown]
# # VoxDetect: Results Visualization
# ## Reproducing Figures and Tables from the Paper

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path("..").resolve()))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from scipy import stats

from src import config
from src.gmm_models import load_all_speech_models, load_all_speaker_models, log_likelihood
from src.segmentation import load_splits, get_digit_paths
from src.forgery_generator import load_dataset_info
from src.authenticator import authenticate_audio
from src.evaluator import (
    evaluate_dataset,
    plot_llh_distributions,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_tp_tn_summary,
    GENUINE_COLOR, IMPOSTOR_COLOR,
)
from src.features import features_from_files

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

FIGURES_DIR = config.FIGURES_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("✅ Results Visualization Notebook Ready")

# %%
# Load all components
def load_all():
    try:
        train_split, test_split = load_splits(config.SPLITS_DIR)
        speaker_ids = list(train_split.keys())
    except FileNotFoundError:
        print("⚠️  Splits not found. Run scripts/01_preprocess.py first.")
        return None, None, None, None, None
    
    speech_models = load_all_speech_models()
    if not speech_models:
        print("⚠️  No speech models. Run scripts/02_train_models.py first.")
        return speaker_ids, None, None, train_split, test_split
    
    speaker_models = load_all_speaker_models(speaker_ids)
    
    try:
        dataset_info = load_dataset_info()
    except FileNotFoundError:
        print("⚠️  No forged dataset. Run scripts/03_generate_forgeries.py first.")
        return speaker_ids, speech_models, None, train_split, test_split
    
    return speaker_ids, speech_models, speaker_models, dataset_info, train_split, test_split

try:
    speaker_ids, speech_models, speaker_models, dataset_info, train_split, test_split = load_all()
    has_models  = speech_models is not None
    has_dataset = dataset_info is not None
except Exception as e:
    print(f"Load failed: {e}")
    speaker_ids = None
    has_models = has_dataset = False

# %% [markdown]
# ## 1. Reproduce Fig. 4: LLH Scores Across All Speakers for One Digit

# %%
def reproduce_fig4(speaker_id, digit_idx, speech_models, speaker_models,
                   test_split, title=None):
    """
    Reproduce Figure 4 from the paper:
    'Computed LLH of digit2 of speaker NS15 with all the impostors'
    
    Shows the LLH of a test utterance from speaker_id against all
    registered speaker models for the given digit.
    """
    # Get a test utterance from the claimed speaker
    test_sessions = list(test_split.get(speaker_id, {}).keys())
    paths = get_digit_paths(speaker_id, digit_idx, sessions=test_sessions)
    
    if not paths:
        print(f"No test utterances found for {speaker_id} digit {digit_idx}")
        return None
    
    import librosa
    signal, _ = librosa.load(str(paths[0]), sr=config.TARGET_SR)
    from src.features import extract_mfcc_matrix
    features = extract_mfcc_matrix(signal.astype(np.float32))
    
    # Compute LLH for all speakers
    all_speakers = sorted(speaker_models.keys())
    llh_values   = []
    for spk in all_speakers:
        if digit_idx in speaker_models.get(spk, {}):
            gmm = speaker_models[spk][digit_idx]
            try:
                llh = log_likelihood(gmm, features)
            except Exception:
                llh = -999
        else:
            llh = -999
        llh_values.append(llh)
    
    # Identify genuine and impostors
    genuine_idx  = all_speakers.index(speaker_id) if speaker_id in all_speakers else -1
    colors = [GENUINE_COLOR if spk == speaker_id else IMPOSTOR_COLOR
              for spk in all_speakers]
    
    fig, ax = plt.subplots(figsize=(max(10, len(all_speakers) * 0.35), 5))
    bars = ax.bar(range(len(all_speakers)), llh_values, color=colors, width=0.7, alpha=0.85)
    
    ax.set_xticks(range(len(all_speakers)))
    ax.set_xticklabels(all_speakers, rotation=90, fontsize=6)
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title(title or f"LLH of Digit '{config.DIGIT_NAMES[digit_idx].capitalize()}' "
                          f"from {speaker_id} against all {len(all_speakers)} speakers")
    
    # Annotate the genuine speaker
    if genuine_idx >= 0:
        ax.annotate(f"← Claimed speaker\n   ({speaker_id})",
                    xy=(genuine_idx, llh_values[genuine_idx]),
                    xytext=(genuine_idx + 2, llh_values[genuine_idx] + 5),
                    arrowprops=dict(arrowstyle="->", color="black"),
                    fontsize=9, color="navy")
    
    genuine_patch  = mpatches.Patch(color=GENUINE_COLOR, label=f"Claimed ({speaker_id})")
    impostor_patch = mpatches.Patch(color=IMPOSTOR_COLOR, label="Other speakers (impostors)")
    ax.legend(handles=[genuine_patch, impostor_patch], fontsize=9)
    
    plt.tight_layout()
    return fig


if has_models and speaker_ids:
    # Use first speaker as the "genuine" speaker
    demo_speaker = speaker_ids[0]
    demo_digit   = 1  # Digit "one"
    
    fig = reproduce_fig4(
        demo_speaker, demo_digit,
        speech_models, speaker_models, test_split,
        title=f"Fig. 4 Reproduction: LLH of '{config.DIGIT_NAMES[demo_digit]}' — Claimed: {demo_speaker}"
    )
    if fig:
        fig.savefig(FIGURES_DIR / "04_fig4_llh_all_speakers.png", bbox_inches="tight")
        plt.show()
        print(f"Fig. 4 reproduction saved.")
else:
    print("Models not available — skipping Fig. 4 reproduction.")

# %% [markdown]
# ## 2. Reproduce Fig. 6: LLH Distribution — Genuine vs Impostor

# %%
if has_dataset and has_models:
    try:
        metrics, genuine_res, forged_res = evaluate_dataset(
            dataset_info, speech_models, speaker_models, "S1"
        )
        
        plot_llh_distributions(
            genuine_res, forged_res,
            digit_positions=[0, 1, 2],
            out_dir=FIGURES_DIR,
            filename="04_fig6_llh_distributions.png",
        )
        print("Fig. 6 reproduction saved.")
    except Exception as e:
        print(f"Could not reproduce Fig. 6: {e}")
else:
    print("Dataset/models not available — skipping Fig. 6.")

# %% [markdown]
# ## 3. Reproduce Fig. 7: Probability Density Functions

# %%
def reproduce_fig7(genuine_results, forged_results, position=0):
    """
    Reproduce Figure 7: PDF of LLH values for genuine vs. impostor.
    Shows the overlap between genuine and impostor distributions.
    """
    gen_llhs = [
        seg.llh_claimed
        for r in genuine_results
        for seg in r.segment_results
        if seg.position == position and seg.llh_claimed > -900
    ]
    imp_llhs = [
        seg.llh_claimed
        for r in forged_results
        for seg in r.segment_results
        if seg.position == position and seg.llh_claimed > -900
    ]
    
    if not gen_llhs or not imp_llhs:
        print("Insufficient data for PDF plot.")
        return None
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # KDE plots (Probability Density Functions)
    gen_llhs_arr = np.array(gen_llhs)
    imp_llhs_arr = np.array(imp_llhs)
    
    x_min = min(gen_llhs_arr.min(), imp_llhs_arr.min()) - 2
    x_max = max(gen_llhs_arr.max(), imp_llhs_arr.max()) + 2
    x     = np.linspace(x_min, x_max, 500)
    
    if len(gen_llhs_arr) > 5:
        gen_kde = stats.gaussian_kde(gen_llhs_arr, bw_method=0.3)
        ax.fill_between(x, gen_kde(x), alpha=0.5, color=GENUINE_COLOR,
                        label=f"Genuine (n={len(gen_llhs_arr)}, μ={gen_llhs_arr.mean():.1f})")
        ax.plot(x, gen_kde(x), color=GENUINE_COLOR, linewidth=2)
    
    if len(imp_llhs_arr) > 5:
        imp_kde = stats.gaussian_kde(imp_llhs_arr, bw_method=0.3)
        ax.fill_between(x, imp_kde(x), alpha=0.5, color=IMPOSTOR_COLOR,
                        label=f"Impostor (n={len(imp_llhs_arr)}, μ={imp_llhs_arr.mean():.1f})")
        ax.plot(x, imp_kde(x), color=IMPOSTOR_COLOR, linewidth=2)
    
    ax.set_xlabel("Log-Likelihood (LLH)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Fig. 7 Reproduction: PDF of LLH Values\n"
                 f"Genuine vs. Impostor — Segment Position {position + 1}")
    ax.legend(fontsize=10)
    ax.axvline(gen_llhs_arr.mean() if gen_llhs else 0, 
               color=GENUINE_COLOR, linestyle="--", alpha=0.7)
    ax.axvline(imp_llhs_arr.mean() if imp_llhs else 0,
               color=IMPOSTOR_COLOR, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    return fig


if has_dataset and has_models:
    try:
        fig = reproduce_fig7(genuine_res, forged_res, position=0)
        if fig:
            fig.savefig(FIGURES_DIR / "04_fig7_pdf_llh.png", bbox_inches="tight")
            plt.show()
            print("Fig. 7 reproduction saved.")
    except Exception as e:
        print(f"Could not reproduce Fig. 7: {e}")
else:
    print("Skipping Fig. 7.")

# %% [markdown]
# ## 4. Full Results Table (Tables 7–10)

# %%
from src.evaluator import evaluate_all_sentences

if has_dataset and has_models:
    try:
        summary_df = evaluate_all_sentences(
            dataset_info=dataset_info,
            speech_models=speech_models,
            speaker_models=speaker_models,
            sentences=list(config.SENTENCES.keys()),
            reports_dir=config.REPORTS_DIR,
        )
        
        print("\n📊 Full Results (matching Tables 7–10 in paper):")
        print(summary_df.to_string(index=False))
        
        # TP/TN Summary Plot
        plot_tp_tn_summary(summary_df, out_dir=FIGURES_DIR,
                            filename="04_tp_tn_summary.png")
        
        # Confusion Matrix
        plot_confusion_matrix(genuine_res, forged_res, out_dir=FIGURES_DIR,
                               filename="04_confusion_matrix.png")
        
        # ROC Curve
        plot_roc_curve(genuine_res, forged_res, out_dir=FIGURES_DIR,
                        filename="04_roc_curve.png")
        
        print("\nAll result plots saved!")
        
    except Exception as e:
        print(f"Full evaluation failed: {e}")
else:
    print("Models/dataset not ready. Run all scripts first.")

# %% [markdown]
# ## 5. Summary Comparison with Paper

# %%
paper_results = {
    "Sentence": ["S1", "S2", "S3", "S4"],
    "TP_paper": [98.80, 98.80, 100.00, 100.00],   # From Tables 7-10 (MC1 OF)
    "TN_paper": [99.90, 100.0, 100.0, 100.0],
}
paper_df = pd.DataFrame(paper_results)

print("\n📋 Paper Benchmark Results (MC1, Office Environment):")
print(paper_df.to_string(index=False))
print()
print("Note: Paper results are for KSUD dataset (different from Kaggle dataset)")
print("      Our implementation uses the same algorithm; results may vary by dataset.")

print("\n✅ Notebook 4: Results Visualization complete!")
print(f"   All figures: {FIGURES_DIR}")
print(f"   All reports: {config.REPORTS_DIR}")
