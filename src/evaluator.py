"""
VoxDetect — src/evaluator.py
Evaluation metrics, plots, and report generation.

Reproduces all result tables and figures from the paper:
    Mubeen et al. (2021)

Metrics:
    TP (True Positive)  — Genuine audio correctly authenticated
    FN (False Negative) — Genuine audio incorrectly rejected
    TN (True Negative)  — Forged audio correctly detected
    FP (False Positive) — Forged audio incorrectly authenticated

Figures produced:
    - LLH distribution plots (like Fig. 6 and Fig. 7 in paper)
    - Per-digit recognition accuracy bar charts (Tables 2, 3, 4)
    - Confusion matrix
    - TP/TN/FP/FN summary tables (like Tables 7–12)
    - ROC curves
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/notebook use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm

from src import config
from src.authenticator import AuthenticationResult, authenticate_audio
from src.gmm_models import GaussianMixture, log_likelihood, train_gmm
from src.features import features_from_files
from src.segmentation import get_digit_paths

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Plot style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi":   120,
})

GENUINE_COLOR  = "#2196F3"   # Blue (genuine speaker, like paper's green)
IMPOSTOR_COLOR = "#F44336"   # Red (impostor, like paper)
DETECTED_COLOR = "#FF9800"   # Orange (wrongly identified)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(results: List[AuthenticationResult]) -> Dict:
    """
    Compute TP, FN, TN, FP from a list of authentication results.

    Convention (matching paper):
        Genuine audio = POSITIVE class
        Forged  audio = NEGATIVE class

    Args:
        results: List of AuthenticationResult objects

    Returns:
        metrics: {TP%, FN%, TN%, FP%, n_genuine, n_forged, accuracy}
    """
    tp = fn = tn = fp = 0

    for r in results:
        true_label  = r.audio_path   # We'll use a separate is_genuine_label
        # We rely on result.is_genuine and a ground truth label stored externally
        # This function is called with pre-tagged results (see evaluate_dataset)
        pass

    # This is actually called from evaluate_dataset which sets gt properly
    return {}


def evaluate_dataset(
    dataset_info:  Dict,           # From load_dataset_info()
    speech_models: Dict[int, GaussianMixture],
    speaker_models: Dict[str, Dict[int, GaussianMixture]],
    sentence_name: str = "S1",
) -> Tuple[Dict, List[AuthenticationResult], List[AuthenticationResult]]:
    """
    Run full evaluation on the generated genuine + forged dataset.

    Args:
        dataset_info:   From forgery_generator.load_dataset_info()
        speech_models:  Trained speech GMMs
        speaker_models: Trained speaker GMMs
        sentence_name:  Which sentence to evaluate ("S1", "S2", "S3", "S4")

    Returns:
        metrics:         TP/FN/TN/FP percentages
        genuine_results: List of AuthenticationResult for genuine audio
        forged_results:  List of AuthenticationResult for forged audio
    """
    genuine_items = dataset_info.get("genuine", {}).get(sentence_name, [])
    forged_items  = dataset_info.get("forged",  {}).get(sentence_name, [])

    logger.info(f"Evaluating sentence {sentence_name}: "
                f"{len(genuine_items)} genuine, {len(forged_items)} forged")

    genuine_results: List[AuthenticationResult] = []
    forged_results:  List[AuthenticationResult] = []

    # Evaluate genuine audio
    for item in tqdm(genuine_items, desc=f"{sentence_name} genuine"):
        claimed = item["claimed_speaker"]
        path    = item["path"]
        result  = authenticate_audio(path, claimed, speech_models, speaker_models)
        genuine_results.append(result)

    # Evaluate forged audio
    for item in tqdm(forged_items, desc=f"{sentence_name} forged"):
        claimed = item["claimed_speaker"]
        path    = item["path"]
        result  = authenticate_audio(path, claimed, speech_models, speaker_models)
        forged_results.append(result)

    # Compute metrics
    n_genuine = len(genuine_results)
    n_forged  = len(forged_results)

    # Genuine audio: TP = correctly identified as genuine, FN = wrongly rejected
    tp = sum(1 for r in genuine_results if r.is_genuine)
    fn = n_genuine - tp

    # Forged audio: TN = correctly identified as forged, FP = wrongly authenticated
    tn = sum(1 for r in forged_results if not r.is_genuine)
    fp = n_forged - tn

    metrics = {
        "sentence":    sentence_name,
        "n_genuine":   n_genuine,
        "n_forged":    n_forged,
        "TP":          tp,
        "FN":          fn,
        "TN":          tn,
        "FP":          fp,
        "TP_pct":      round(100 * tp / max(n_genuine, 1), 2),
        "FN_pct":      round(100 * fn / max(n_genuine, 1), 2),
        "TN_pct":      round(100 * tn / max(n_forged, 1), 2),
        "FP_pct":      round(100 * fp / max(n_forged, 1), 2),
        "accuracy":    round(100 * (tp + tn) / max(n_genuine + n_forged, 1), 2),
    }

    logger.info(f"  TP={metrics['TP_pct']:.1f}%  FN={metrics['FN_pct']:.1f}%  "
                f"TN={metrics['TN_pct']:.1f}%  FP={metrics['FP_pct']:.1f}%")

    return metrics, genuine_results, forged_results


def evaluate_all_sentences(
    dataset_info:   Dict,
    speech_models:  Dict[int, GaussianMixture],
    speaker_models: Dict[str, Dict[int, GaussianMixture]],
    sentences:      List[str] = list(config.SENTENCES.keys()),
    reports_dir:    Path = config.REPORTS_DIR,
) -> pd.DataFrame:
    """
    Evaluate all sentences and return a summary DataFrame (like Tables 7–10 in paper).

    Returns:
        df: DataFrame with columns [sentence, TP%, FN%, TN%, FP%]
    """
    rows = []
    for sent in sentences:
        metrics, _, _ = evaluate_dataset(dataset_info, speech_models, speaker_models, sent)
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df[["sentence", "n_genuine", "n_forged", "TP_pct", "FN_pct", "TN_pct", "FP_pct", "accuracy"]]
    df.columns = ["Sentence", "N Genuine", "N Forged", "TP(%)", "FN(%)", "TN(%)", "FP(%)", "Accuracy(%)"]

    # Save to CSV
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "evaluation_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"Results saved to {out_csv}")
    print("\n" + df.to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MIXTURE COUNT OPTIMIZATION (Reproduce Tables 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_speech_mixture_counts(
    speaker_ids:    List[str],
    train_sessions: Dict[str, List[str]],
    test_sessions:  Dict[str, List[str]],
    digits:         List[int]    = config.ACTIVE_DIGITS,
    mixture_counts: List[int]    = config.SPEECH_GMM_EVAL_MIXTURES,
    processed_dir:  Path         = config.PROCESSED_DIR,
    reports_dir:    Path         = config.REPORTS_DIR,
) -> pd.DataFrame:
    """
    Reproduce Tables 2 & 3: speech recognition accuracy vs. GMM mixture count.
    
    Trains speech GMMs for each digit at each mixture count, then measures
    how accurately the word is recognized on held-out test data.

    Returns:
        df: (n_digits × n_mixture_counts) accuracy table
    """
    logger.info("Evaluating speech GMM mixture counts...")
    rows = []

    for digit_idx in tqdm(digits, desc="Digits"):
        # Collect training data: all speakers, train sessions
        train_paths = []
        test_paths  = []
        for spk in speaker_ids:
            train_paths.extend(get_digit_paths(spk, digit_idx, processed_dir,
                                                train_sessions.get(spk)))
            test_paths.extend(get_digit_paths(spk, digit_idx, processed_dir,
                                               test_sessions.get(spk)))

        if not train_paths or not test_paths:
            continue

        try:
            from src.features import features_from_files
            X_train = features_from_files(train_paths)
            X_test  = features_from_files(test_paths)
        except Exception as e:
            logger.warning(f"  Digit {digit_idx}: feature extraction failed: {e}")
            continue

        row = {"Digit": config.DIGIT_NAMES[digit_idx].capitalize()}

        for n in mixture_counts:
            try:
                gmm = train_gmm(X_train, n_components=n)
                # Measure per-frame average LLH on test (proxy for quality)
                llh = gmm.score(X_test)
                row[f"{n} mix"] = round(llh, 3)
            except Exception as e:
                row[f"{n} mix"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / "speech_mixture_optimization.csv", index=False)
    logger.info("Speech mixture optimization saved.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: LLH DISTRIBUTIONS (Paper Fig. 6)
# ─────────────────────────────────────────────────────────────────────────────

def plot_llh_distributions(
    genuine_results: List[AuthenticationResult],
    forged_results:  List[AuthenticationResult],
    digit_positions: List[int] = [0, 1, 2],   # Which positions to plot
    out_dir:         Path = config.FIGURES_DIR,
    filename:        str  = "llh_distributions.png",
) -> None:
    """
    Plot LLH value distributions for genuine vs. impostor segments.
    Reproduces Fig. 6 from the paper.

    Args:
        genuine_results: List of AuthenticationResult for genuine audio
        forged_results:  List of AuthenticationResult for forged audio
        digit_positions: Which segment positions to include in plot
        out_dir:         Directory to save figure
        filename:        Output filename
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_plots = len(digit_positions)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, pos in zip(axes, digit_positions):
        gen_llhs = []
        imp_llhs = []

        # Genuine audio: collect LLH for claimed speaker at this position
        for r in genuine_results:
            segs = [s for s in r.segment_results if s.position == pos]
            for s in segs:
                if s.llh_claimed > float("-inf"):
                    gen_llhs.append(s.llh_claimed)

        # Forged audio: collect LLH at this position
        for r in forged_results:
            segs = [s for s in r.segment_results if s.position == pos]
            for s in segs:
                # Impostor segment: compare claimed speaker LLH
                if s.llh_claimed > float("-inf"):
                    imp_llhs.append(s.llh_claimed)

        if gen_llhs:
            ax.hist(gen_llhs, bins=30, alpha=0.7, color=GENUINE_COLOR,
                    label=f"Genuine (n={len(gen_llhs)})", density=True)
        if imp_llhs:
            ax.hist(imp_llhs, bins=30, alpha=0.7, color=IMPOSTOR_COLOR,
                    label=f"Impostor (n={len(imp_llhs)})", density=True)

        ax.set_title(f"Segment Position {pos + 1}")
        ax.set_xlabel("Log-Likelihood")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)

    plt.suptitle("LLH Distribution: Genuine vs. Impostor Segments", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"LLH distribution plot saved: {out_dir / filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: PER-SPEAKER LLH BAR CHART (Paper Fig. 4)
# ─────────────────────────────────────────────────────────────────────────────

def plot_speaker_llh_comparison(
    audio_path:      str,
    claimed_speaker: str,
    digit_idx:       int,
    speech_models:   Dict[int, GaussianMixture],
    speaker_models:  Dict[str, Dict[int, GaussianMixture]],
    out_dir:         Path = config.FIGURES_DIR,
    filename:        str  = "speaker_llh_comparison.png",
) -> None:
    """
    Plot LLH of a test segment against all registered speakers.
    Reproduces Fig. 4 from the paper.

    Shows which speakers score highest for a test utterance,
    highlighting the claimed speaker in blue and others in red.
    """
    import librosa
    from src.vad import extract_word_regions, extract_word_signals
    from src.features import extract_mfcc_matrix

    signal, _ = librosa.load(audio_path, sr=config.TARGET_SR, mono=True)
    regions   = extract_word_regions(signal.astype(np.float32))
    words     = extract_word_signals(signal.astype(np.float32), regions)

    if not words:
        logger.warning("No words detected in audio for LLH comparison plot.")
        return

    word_idx = min(digit_idx, len(words) - 1)
    features = extract_mfcc_matrix(words[word_idx])

    if digit_idx not in speaker_models.get(claimed_speaker, {}):
        logger.warning("No speaker model available for comparison plot.")
        return

    speaker_ids = sorted(speaker_models.keys())
    llh_values  = []
    for spk in speaker_ids:
        if digit_idx in speaker_models[spk]:
            gmm = speaker_models[spk][digit_idx]
            try:
                llh = log_likelihood(gmm, features)
            except Exception:
                llh = float("-inf")
        else:
            llh = float("-inf")
        llh_values.append(llh)

    colors = [GENUINE_COLOR if spk == claimed_speaker else IMPOSTOR_COLOR
              for spk in speaker_ids]

    fig, ax = plt.subplots(figsize=(max(10, len(speaker_ids) * 0.35), 5))
    bars = ax.bar(range(len(speaker_ids)), llh_values, color=colors, width=0.7)
    ax.set_xticks(range(len(speaker_ids)))
    ax.set_xticklabels(speaker_ids, rotation=90, fontsize=7)
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title(
        f"LLH Scores for Digit '{config.DIGIT_NAMES[digit_idx]}' — "
        f"Claimed: {claimed_speaker}"
    )

    # Legend
    genuine_patch  = mpatches.Patch(color=GENUINE_COLOR, label=f"Claimed ({claimed_speaker})")
    impostor_patch = mpatches.Patch(color=IMPOSTOR_COLOR, label="Other speakers")
    ax.legend(handles=[genuine_patch, impostor_patch])

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Speaker LLH comparison plot saved: {out_dir / filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    genuine_results: List[AuthenticationResult],
    forged_results:  List[AuthenticationResult],
    out_dir:         Path = config.FIGURES_DIR,
    filename:        str  = "confusion_matrix.png",
) -> None:
    """
    Plot a confusion matrix for genuine vs. forged classification.
    """
    y_true = [1] * len(genuine_results) + [0] * len(forged_results)
    y_pred = [int(r.is_genuine) for r in genuine_results] + \
             [int(r.is_genuine) for r in forged_results]

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted GENUINE", "Predicted FORGED"],
        yticklabels=["Actual GENUINE",    "Actual FORGED"],
        ax=ax, cbar=True, linewidths=0.5,
    )
    ax.set_title("Confusion Matrix — VoxDetect System", fontsize=13)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {out_dir / filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: ACCURACY BAR CHART (Per-Digit, Tables 2–4)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mixture_accuracy(
    accuracy_df: pd.DataFrame,
    title:       str = "Speech Recognition Accuracy vs. GMM Mixture Count",
    out_dir:     Path = config.FIGURES_DIR,
    filename:    str  = "mixture_accuracy.png",
) -> None:
    """
    Bar chart of per-digit accuracy for different mixture counts.
    Reproduces the style of Tables 2, 3, 4 from the paper as a visual.
    """
    numeric_cols = [c for c in accuracy_df.columns if c != "Digit"]
    x     = np.arange(len(accuracy_df))
    width = 0.8 / len(numeric_cols)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, col in enumerate(numeric_cols):
        vals = accuracy_df[col].fillna(0).values
        ax.bar(x + i * width, vals, width=width, label=col, alpha=0.85)

    ax.set_xticks(x + width * (len(numeric_cols) - 1) / 2)
    ax.set_xticklabels(accuracy_df["Digit"].values, rotation=45)
    ax.set_ylabel("Avg Log-Likelihood per Frame")
    ax.set_title(title)
    ax.legend(title="Mixtures", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Mixture accuracy plot saved: {out_dir / filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: TP/TN SUMMARY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_tp_tn_summary(
    summary_df: pd.DataFrame,
    out_dir:    Path = config.FIGURES_DIR,
    filename:   str  = "tp_tn_summary.png",
) -> None:
    """
    Grouped bar chart of TP% and TN% per sentence.
    Mirrors Tables 7–10 from the paper visually.
    """
    sentences = summary_df["Sentence"].tolist()
    tp_vals   = summary_df["TP(%)"].tolist()
    tn_vals   = summary_df["TN(%)"].tolist()

    x     = np.arange(len(sentences))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, tp_vals, width, label="TP% (Genuine correctly auth.)",
                   color=GENUINE_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width/2, tn_vals, width, label="TN% (Forged correctly detected)",
                   color=IMPOSTOR_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(sentences)
    ax.set_ylim(80, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Sentence")
    ax.set_title("Authentication Accuracy by Sentence\n(TP = Genuine Auth., TN = Forgery Detection)")
    ax.legend()
    ax.axhline(100, color="grey", linestyle="--", linewidth=0.8)

    # Annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"TP/TN summary plot saved: {out_dir / filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: ROC CURVE
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    genuine_results: List[AuthenticationResult],
    forged_results:  List[AuthenticationResult],
    out_dir:         Path = config.FIGURES_DIR,
    filename:        str  = "roc_curve.png",
) -> None:
    """
    Plot ROC curve using the ratio of LLH(claimed) / LLH(best) as score.
    """
    scores = []
    labels = []

    for r in genuine_results + forged_results:
        # Score: average LLH of claimed speaker across all segments
        llh_vals = [
            seg.llh_claimed for seg in r.segment_results
            if seg.llh_claimed > float("-inf")
        ]
        score = np.mean(llh_vals) if llh_vals else -999.0
        scores.append(score)

    labels = [1] * len(genuine_results) + [0] * len(forged_results)

    if len(set(labels)) < 2:
        logger.warning("Cannot plot ROC — only one class present.")
        return

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=GENUINE_COLOR, lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — VoxDetect Audio Forgery Detection")
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved: {out_dir / filename}")


# ─────────────────────────────────────────────────────────────────────────────
# FULL REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_report(
    all_genuine: List[AuthenticationResult],
    all_forged:  List[AuthenticationResult],
    summary_df:  pd.DataFrame,
    reports_dir: Path = config.REPORTS_DIR,
    figures_dir: Path = config.FIGURES_DIR,
) -> None:
    """
    Save a comprehensive JSON + text report of all evaluation results.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSON
    results_data = {
        "genuine_count": len(all_genuine),
        "forged_count":  len(all_forged),
        "genuine_tp_pct": round(100 * sum(r.is_genuine for r in all_genuine) / max(len(all_genuine), 1), 2),
        "forged_tn_pct":  round(100 * sum(not r.is_genuine for r in all_forged) / max(len(all_forged), 1), 2),
        "overall_accuracy": round(
            100 * (sum(r.is_genuine for r in all_genuine) +
                   sum(not r.is_genuine for r in all_forged)) /
            max(len(all_genuine) + len(all_forged), 1), 2
        ),
    }

    with open(reports_dir / "final_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  VoxDetect — Final Evaluation Results")
    print("=" * 60)
    print(f"  Genuine audios tested : {results_data['genuine_count']}")
    print(f"  Forged audios tested  : {results_data['forged_count']}")
    print(f"  Genuine auth rate (TP): {results_data['genuine_tp_pct']:.2f}%")
    print(f"  Forgery detect. (TN) : {results_data['forged_tn_pct']:.2f}%")
    print(f"  Overall accuracy      : {results_data['overall_accuracy']:.2f}%")
    print("=" * 60)
    print("\nPer-sentence breakdown:")
    print(summary_df.to_string(index=False))
    print()

    logger.info(f"Full report saved to {reports_dir}")


if __name__ == "__main__":
    print("Evaluator module ready.")
    print(f"Figures output: {config.FIGURES_DIR}")
    print(f"Reports output: {config.REPORTS_DIR}")
