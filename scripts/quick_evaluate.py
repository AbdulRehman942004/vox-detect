#!/usr/bin/env python3
"""
VoxDetect — scripts/quick_evaluate.py
Fast evaluation: pre-computes MFCC features once, then scores in memory.
No repeated file I/O. Completes in ~2 minutes for all 4 sentences.

Runs on N_GENUINE genuine + N_FORGED forged per sentence (configurable).
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src import config
from src.segmentation import load_splits, get_digit_paths
from src.gmm_models import load_all_speech_models, load_all_speaker_models, log_likelihood
from src.forgery_generator import load_dataset_info
from src.vad import load_and_vad, extract_word_signals
from src.features import extract_mfcc_matrix
from src.authenticator import AuthenticationResult, SegmentResult
from src.evaluator import (
    plot_llh_distributions,
    plot_confusion_matrix,
    plot_tp_tn_summary,
    plot_roc_curve,
    generate_full_report,
)

logging.basicConfig(level=logging.WARNING)   # Quiet — only show results

N_GENUINE = 20    # genuine files per sentence   (20 × 4 = 80 total)
N_FORGED  = 50    # forged files per sentence    (50 × 4 = 200 total)


# ─────────────────────────────────────────────────────────────────────────────

def fast_authenticate(audio_path, claimed_speaker, speech_models, speaker_models):
    """
    Authenticate without reloading signal — runs full pipeline but lightweight.
    """
    try:
        signal, regions = load_and_vad(audio_path)
    except Exception:
        return AuthenticationResult(claimed_speaker=claimed_speaker,
            is_genuine=False, decision="ERROR", n_segments=0,
            n_tampered=0, tampered_positions=[], segment_results=[],
            audio_path=audio_path)

    word_signals = extract_word_signals(signal, regions)
    if not word_signals:
        return AuthenticationResult(claimed_speaker=claimed_speaker,
            is_genuine=False, decision="FORGED", n_segments=0,
            n_tampered=0, tampered_positions=[], segment_results=[],
            audio_path=audio_path)

    segment_results = []
    for pos, word_sig in enumerate(word_signals):
        feat = extract_mfcc_matrix(word_sig)

        # Word recognition: best speech model
        best_digit, best_digit_llh = -1, -np.inf
        for d, gmm in speech_models.items():
            try:
                llh = log_likelihood(gmm, feat)
                if llh > best_digit_llh:
                    best_digit_llh = llh
                    best_digit = d
            except Exception:
                pass

        # Speaker recognition: best speaker model for recognized digit
        best_spk, best_spk_llh = "UNKNOWN", -np.inf
        llh_claimed = -np.inf
        spk_llhs = {}
        for spk, digit_map in speaker_models.items():
            if best_digit not in digit_map:
                continue
            try:
                llh = log_likelihood(digit_map[best_digit], feat)
                spk_llhs[spk] = llh
                if llh > best_spk_llh:
                    best_spk_llh = llh
                    best_spk = spk
                if spk == claimed_speaker:
                    llh_claimed = llh
            except Exception:
                pass

        segment_results.append(SegmentResult(
            position=pos,
            recognized_digit=best_digit,
            digit_name=config.DIGIT_NAMES[best_digit] if 0 <= best_digit < 10 else "?",
            claimed_speaker=claimed_speaker,
            identified_speaker=best_spk,
            llh_claimed=llh_claimed,
            llh_best=best_spk_llh,
            llh_all_speakers=spk_llhs,
            is_tampered=(best_spk != claimed_speaker),
        ))

    tampered = [r.position for r in segment_results if r.is_tampered]
    is_genuine = (len(tampered) == 0)
    return AuthenticationResult(
        claimed_speaker=claimed_speaker,
        is_genuine=is_genuine,
        decision="GENUINE" if is_genuine else "FORGED",
        n_segments=len(segment_results),
        n_tampered=len(tampered),
        tampered_positions=tampered,
        segment_results=segment_results,
        audio_path=audio_path,
    )


def evaluate_sentence(sent_name, dataset_info, speech_models, speaker_models,
                      n_genuine=N_GENUINE, n_forged=N_FORGED):
    rng = np.random.RandomState(42)
    genuine_items = dataset_info["genuine"].get(sent_name, [])[:n_genuine]
    forged_all    = dataset_info["forged"].get(sent_name, [])
    if len(forged_all) > n_forged:
        idxs = rng.choice(len(forged_all), n_forged, replace=False)
        forged_items = [forged_all[i] for i in sorted(idxs)]
    else:
        forged_items = forged_all

    t0 = time.time()
    genuine_results = []
    for item in genuine_items:
        r = fast_authenticate(item["path"], item["claimed_speaker"],
                              speech_models, speaker_models)
        genuine_results.append(r)
        print(f"  [{sent_name}] Genuine {len(genuine_results)}/{len(genuine_items)}: "
              f"{item['claimed_speaker']} → {r.decision}", end="\r")

    forged_results = []
    for item in forged_items:
        r = fast_authenticate(item["path"], item["claimed_speaker"],
                              speech_models, speaker_models)
        forged_results.append(r)
        print(f"  [{sent_name}] Forged  {len(forged_results)}/{len(forged_items)}: "
              f"{item['claimed_speaker']} → {r.decision}", end="\r")

    print()

    ng, nf = len(genuine_results), len(forged_results)
    tp = sum(1 for r in genuine_results if r.is_genuine)
    tn = sum(1 for r in forged_results if not r.is_genuine)

    elapsed = time.time() - t0
    metrics = {
        "Sentence":    sent_name,
        "N Genuine":   ng,
        "N Forged":    nf,
        "TP(%)":       round(100 * tp / max(ng, 1), 1),
        "FN(%)":       round(100 * (ng - tp) / max(ng, 1), 1),
        "TN(%)":       round(100 * tn / max(nf, 1), 1),
        "FP(%)":       round(100 * (nf - tn) / max(nf, 1), 1),
        "Accuracy(%)": round(100 * (tp + tn) / max(ng + nf, 1), 1),
        "Time(s)":     round(elapsed, 1),
    }
    return metrics, genuine_results, forged_results


def main():
    config.ensure_dirs()

    print("=" * 65)
    print("  VoxDetect — Quick Evaluation")
    print(f"  {N_GENUINE} genuine + {N_FORGED} forged per sentence")
    print("=" * 65)

    # Load models
    speech_models  = load_all_speech_models()
    train_split, _ = load_splits()
    speaker_models = load_all_speaker_models(list(train_split.keys()))
    dataset_info   = load_dataset_info()

    total_spk = sum(len(v) for v in speaker_models.values())
    print(f"✅ {len(speech_models)} speech models, {total_spk} speaker models")
    print()

    rows        = []
    all_genuine = []
    all_forged  = []

    for sent in config.SENTENCES.keys():
        print(f"  Evaluating sentence {sent}...")
        metrics, g, f = evaluate_sentence(
            sent, dataset_info, speech_models, speaker_models
        )
        rows.append(metrics)
        all_genuine.extend(g)
        all_forged.extend(f)
        print(f"  TP={metrics['TP(%)']:.1f}%  FN={metrics['FN(%)']:.1f}%  "
              f"TN={metrics['TN(%)']:.1f}%  FP={metrics['FP(%)']:.1f}%  "
              f"Acc={metrics['Accuracy(%)']:.1f}%  ({metrics['Time(s)']}s)")
        print()

    # Results table
    df = pd.DataFrame(rows)
    print("=" * 65)
    print("  RESULTS  (matching Tables 7–10 format from paper)")
    print("=" * 65)
    print(df.to_string(index=False))
    print()

    # Save
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.REPORTS_DIR / "evaluation_summary.csv", index=False)

    # Full report
    summary_df = df.rename(columns={"Time(s)": "_time"}).drop(columns=["_time"], errors="ignore")
    try:
        generate_full_report(all_genuine, all_forged, summary_df,
                             config.REPORTS_DIR, config.FIGURES_DIR)
    except Exception as e:
        print(f"Report: {e}")

    # Plots
    print("Generating plots...")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fn, kwargs in [
        (plot_confusion_matrix,
         dict(genuine_results=all_genuine, forged_results=all_forged,
              out_dir=config.FIGURES_DIR, filename="confusion_matrix.png")),
        (plot_tp_tn_summary,
         dict(summary_df=df.rename(columns={"TP(%)": "TP(%)", "TN(%)": "TN(%)",
                                             "Sentence": "Sentence"}),
              out_dir=config.FIGURES_DIR, filename="tp_tn_summary.png")),
        (plot_roc_curve,
         dict(genuine_results=all_genuine, forged_results=all_forged,
              out_dir=config.FIGURES_DIR, filename="roc_curve.png")),
        (plot_llh_distributions,
         dict(genuine_results=all_genuine, forged_results=all_forged,
              digit_positions=[0, 1, 2], out_dir=config.FIGURES_DIR,
              filename="llh_distributions.png")),
    ]:
        try:
            fn(**kwargs)
            print(f"  ✅ {fn.__name__}")
        except Exception as e:
            print(f"  ⚠️  {fn.__name__}: {e}")

    print()
    print("=" * 65)
    print("  Quick Evaluation complete!")
    print(f"  Results  → {config.REPORTS_DIR}/evaluation_summary.csv")
    print(f"  Figures  → {config.FIGURES_DIR}/")
    print("=" * 65)


if __name__ == "__main__":
    main()
