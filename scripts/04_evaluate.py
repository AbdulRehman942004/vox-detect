#!/usr/bin/env python3
"""
VoxDetect — scripts/04_evaluate.py
STEP 4: Full Evaluation (Optimised)

Key fixes:
  - Single-pass evaluation (no double-running)
  - Parallel audio authentication via joblib
  - --no-plots and --max-forged flags for speed control

Usage:
    python scripts/04_evaluate.py                      # fast (default)
    python scripts/04_evaluate.py --max-forged 100     # even faster (subset)
    python scripts/04_evaluate.py --mixture-analysis   # also reproduce Tables 2–4
"""

import sys
import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from src import config
from src.segmentation import load_splits
from src.gmm_models import load_all_speech_models, load_all_speaker_models
from src.forgery_generator import load_dataset_info
from src.authenticator import authenticate_audio, AuthenticationResult
from src.evaluator import (
    evaluate_all_sentences,
    evaluate_speech_mixture_counts,
    plot_llh_distributions,
    plot_confusion_matrix,
    plot_tp_tn_summary,
    plot_roc_curve,
    plot_mixture_accuracy,
    generate_full_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL EVALUATION (single pass — fixes the double-evaluation bug)
# ─────────────────────────────────────────────────────────────────────────────

def _auth_worker(args_tuple):
    """Top-level function for multiprocessing (must be picklable)."""
    path, claimed_speaker, speech_pkl, speaker_pkl = args_tuple
    speech_models  = pickle.loads(speech_pkl)
    speaker_models = pickle.loads(speaker_pkl)
    return authenticate_audio(path, claimed_speaker, speech_models, speaker_models)


def evaluate_sentence_parallel(
    genuine_items:  List[Dict],
    forged_items:   List[Dict],
    speech_models:  Dict,
    speaker_models: Dict,
    sentence_name:  str,
    n_jobs:         int = 4,
    max_forged:     int = -1,
) -> Tuple[Dict, List[AuthenticationResult], List[AuthenticationResult]]:
    """
    Authenticate all genuine + forged audio for one sentence.
    Uses joblib for parallel execution.

    Returns metrics dict + lists of AuthenticationResult.
    """
    from joblib import Parallel, delayed

    # Optionally limit forged items for speed
    if max_forged > 0 and len(forged_items) > max_forged:
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(forged_items), size=max_forged, replace=False)
        forged_items = [forged_items[i] for i in sorted(idxs)]

    def auth_one(path, claimed):
        return authenticate_audio(path, claimed, speech_models, speaker_models)

    logger.info(f"Sentence {sentence_name}: "
                f"{len(genuine_items)} genuine, {len(forged_items)} forged "
                f"(n_jobs={n_jobs})")

    # Authentic genuine audio
    genuine_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(auth_one)(item["path"], item["claimed_speaker"])
        for item in tqdm(genuine_items, desc=f"{sentence_name} genuine")
    )

    # Authenticate forged audio
    forged_results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(auth_one)(item["path"], item["claimed_speaker"])
        for item in tqdm(forged_items, desc=f"{sentence_name} forged ")
    )

    n_genuine = len(genuine_results)
    n_forged  = len(forged_results)

    tp = sum(1 for r in genuine_results if r.is_genuine)
    fn = n_genuine - tp
    tn = sum(1 for r in forged_results if not r.is_genuine)
    fp = n_forged - tn

    metrics = {
        "Sentence":    sentence_name,
        "N Genuine":   n_genuine,
        "N Forged":    n_forged,
        "TP(%)":       round(100 * tp / max(n_genuine, 1), 2),
        "FN(%)":       round(100 * fn / max(n_genuine, 1), 2),
        "TN(%)":       round(100 * tn / max(n_forged, 1), 2),
        "FP(%)":       round(100 * fp / max(n_forged, 1), 2),
        "Accuracy(%)": round(100 * (tp + tn) / max(n_genuine + n_forged, 1), 2),
    }

    print(f"  {sentence_name}: TP={metrics['TP(%)']:.1f}%  "
          f"FN={metrics['FN(%)']:.1f}%  "
          f"TN={metrics['TN(%)']:.1f}%  "
          f"FP={metrics['FP(%)']:.1f}%  "
          f"Acc={metrics['Accuracy(%)']:.1f}%")

    return metrics, genuine_results, forged_results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VoxDetect Step 4: Evaluation")
    p.add_argument("--sentences", nargs="+",
                   default=list(config.SENTENCES.keys()),
                   choices=list(config.SENTENCES.keys()),
                   help="Sentences to evaluate (default: all)")
    p.add_argument("--speech-mixtures", type=int,
                   default=config.SPEECH_GMM_COMPONENTS)
    p.add_argument("--speaker-mixtures", type=int,
                   default=config.SPEAKER_GMM_COMPONENTS)
    p.add_argument("--n-jobs", type=int, default=4,
                   help="Parallel workers for authentication (default: 4)")
    p.add_argument("--max-forged", type=int, default=200,
                   help="Max forged files per sentence to evaluate "
                        "(-1 = all 500, slow). Default: 200")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation")
    p.add_argument("--mixture-analysis", action="store_true",
                   help="Also reproduce Tables 2–4 mixture count analysis")
    return p.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()

    print("=" * 65)
    print("  VoxDetect — Step 4: Evaluation")
    print("=" * 65)
    print(f"  Sentences      : {args.sentences}")
    print(f"  Speech mixtures: {args.speech_mixtures}")
    print(f"  Speaker mixtures:{args.speaker_mixtures}")
    print(f"  Parallel jobs  : {args.n_jobs}")
    print(f"  Max forged/sent: {args.max_forged} (-1=all)")
    print()

    # ── Load models ───────────────────────────────────────────────────────
    logger.info("Loading speech models...")
    speech_models = load_all_speech_models(
        digits=config.ACTIVE_DIGITS,
        n_components=args.speech_mixtures,
        models_dir=config.SPEECH_MODELS_DIR,
    )
    if not speech_models:
        print("ERROR: No speech models found. Run scripts/02_train_models.py first.")
        sys.exit(1)
    print(f"✅ Loaded {len(speech_models)} speech models")

    logger.info("Loading speaker models...")
    train_split, test_split = load_splits(config.SPLITS_DIR)
    speaker_ids = list(train_split.keys())

    speaker_models = load_all_speaker_models(
        speaker_ids=speaker_ids,
        digits=config.ACTIVE_DIGITS,
        n_components=args.speaker_mixtures,
        models_dir=config.SPEAKER_MODELS_DIR,
    )
    total = sum(len(v) for v in speaker_models.values())
    print(f"✅ Loaded {total} speaker models ({len(speaker_models)} speakers)")

    # ── Load dataset ──────────────────────────────────────────────────────
    try:
        dataset_info = load_dataset_info(config.FORGED_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run scripts/03_generate_forgeries.py first.")
        sys.exit(1)

    print()
    print("─" * 65)
    print("  Running Authentication (single-pass, parallel)")
    print("─" * 65)

    # ── Single-pass evaluation — collect everything at once ───────────────
    import pandas as pd
    rows = []
    all_genuine: List[AuthenticationResult] = []
    all_forged:  List[AuthenticationResult] = []

    for sent in args.sentences:
        genuine_items = dataset_info.get("genuine", {}).get(sent, [])
        forged_items  = dataset_info.get("forged",  {}).get(sent, [])

        metrics, g_res, f_res = evaluate_sentence_parallel(
            genuine_items, forged_items,
            speech_models, speaker_models,
            sentence_name=sent,
            n_jobs=args.n_jobs,
            max_forged=args.max_forged,
        )
        rows.append(metrics)
        all_genuine.extend(g_res)
        all_forged.extend(f_res)

    # ── Summary table ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(rows)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.REPORTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(out_csv, index=False)

    print()
    print("─" * 65)
    print("  RESULTS SUMMARY (matching Tables 7–10 in paper)")
    print("─" * 65)
    print(summary_df.to_string(index=False))

    # ── Full report ───────────────────────────────────────────────────────
    generate_full_report(all_genuine, all_forged, summary_df,
                         config.REPORTS_DIR, config.FIGURES_DIR)

    # ── Plots ─────────────────────────────────────────────────────────────
    if not args.no_plots:
        print()
        print("─" * 65)
        print("  Generating Plots")
        print("─" * 65)

        for fn, kwargs in [
            (plot_llh_distributions,
             dict(genuine_results=all_genuine, forged_results=all_forged,
                  digit_positions=[0, 1, 2], out_dir=config.FIGURES_DIR,
                  filename="llh_distributions.png")),
            (plot_confusion_matrix,
             dict(genuine_results=all_genuine, forged_results=all_forged,
                  out_dir=config.FIGURES_DIR, filename="confusion_matrix.png")),
            (plot_tp_tn_summary,
             dict(summary_df=summary_df, out_dir=config.FIGURES_DIR,
                  filename="tp_tn_summary.png")),
            (plot_roc_curve,
             dict(genuine_results=all_genuine, forged_results=all_forged,
                  out_dir=config.FIGURES_DIR, filename="roc_curve.png")),
        ]:
            try:
                fn(**kwargs)
            except Exception as e:
                logger.warning(f"{fn.__name__} failed: {e}")

        print(f"✅ Plots saved to: {config.FIGURES_DIR}")

    # ── Optional: mixture analysis ────────────────────────────────────────
    if args.mixture_analysis:
        print()
        print("─" * 65)
        print("  GMM Mixture Count Analysis (Tables 2–4)")
        print("─" * 65)
        train_sessions_map = {s: list(d.keys()) for s, d in train_split.items()}
        test_sessions_map  = {s: list(d.keys()) for s, d in test_split.items()}
        try:
            mix_df = evaluate_speech_mixture_counts(
                speaker_ids=speaker_ids,
                train_sessions=train_sessions_map,
                test_sessions=test_sessions_map,
                digits=config.ACTIVE_DIGITS,
                mixture_counts=config.SPEECH_GMM_EVAL_MIXTURES,
                processed_dir=config.PROCESSED_DIR,
                reports_dir=config.REPORTS_DIR,
            )
            print(mix_df.to_string(index=False))
            if not args.no_plots:
                plot_mixture_accuracy(
                    mix_df,
                    title="Speech GMM: Avg LLH vs. Mixture Count (Tables 2 & 3)",
                    out_dir=config.FIGURES_DIR,
                    filename="mixture_analysis_speech.png",
                )
        except Exception as e:
            logger.warning(f"Mixture analysis failed: {e}")

    print()
    print("=" * 65)
    print("  Step 4 complete!")
    print(f"  Results → {config.REPORTS_DIR}")
    print(f"  Figures → {config.FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
