#!/usr/bin/env python3
"""
VoxDetect — scripts/02_train_models.py
STEP 2: Train GMM Speech and Speaker Models

Trains two sets of models as described in the paper:
    1. Speech Models (word recognition): 256 Gaussian mixtures per digit
       One model trained on ALL speakers' data for that digit.
    
    2. Speaker Models (speaker recognition): 32 Gaussian mixtures
       One model per (speaker, digit) pair.
       Trained ONLY on enrolled/training sessions.

Usage:
    python scripts/02_train_models.py [--speech-only] [--speaker-only] [--force]
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.segmentation import load_splits
from src.gmm_models import (
    train_all_speech_models,
    train_all_speaker_models,
    load_all_speech_models,
    load_all_speaker_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="VoxDetect Step 2: Train GMM Models"
    )
    p.add_argument(
        "--speech-only", action="store_true",
        help="Train only speech (word recognition) models."
    )
    p.add_argument(
        "--speaker-only", action="store_true",
        help="Train only speaker recognition models."
    )
    p.add_argument(
        "--speech-mixtures", type=int, default=config.SPEECH_GMM_COMPONENTS,
        help=f"GMM components for speech models. Default: {config.SPEECH_GMM_COMPONENTS}"
    )
    p.add_argument(
        "--speaker-mixtures", type=int, default=config.SPEAKER_GMM_COMPONENTS,
        help=f"GMM components for speaker models. Default: {config.SPEAKER_GMM_COMPONENTS}"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Force retraining even if models already exist."
    )
    p.add_argument(
        "--speakers", nargs="+", default=None,
        help="Train models only for specific speakers."
    )
    return p.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()

    print("=" * 65)
    print("  VoxDetect — Step 2: GMM Model Training")
    print("=" * 65)
    print(f"  Speech mixtures  : {args.speech_mixtures}")
    print(f"  Speaker mixtures : {args.speaker_mixtures}")
    print(f"  Force retrain    : {args.force}")
    print()

    # ── Load train/test splits ────────────────────────────────────────────
    try:
        train_split, test_split = load_splits(config.SPLITS_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run scripts/01_preprocess.py first.")
        sys.exit(1)

    # Build sessions_map: {speaker_id: [session_ids_for_training]}
    sessions_map: dict = {}
    for spk, sess_data in train_split.items():
        sessions_map[spk] = list(sess_data.keys())

    speaker_ids = list(train_split.keys())
    if args.speakers:
        speaker_ids = [s for s in args.speakers if s in speaker_ids]
        sessions_map = {s: sessions_map[s] for s in speaker_ids if s in sessions_map}

    print(f"  Speakers         : {len(speaker_ids)}")
    print(f"  Digits           : {config.ACTIVE_DIGITS}")
    print()

    # ── Train Speech Models ───────────────────────────────────────────────
    if not args.speaker_only:
        print("─" * 65)
        print("  Training Speech (Word Recognition) Models")
        print(f"  One GMM per digit, pooling ALL speakers, {args.speech_mixtures} components")
        print("─" * 65)

        speech_models = train_all_speech_models(
            speaker_ids=speaker_ids,
            sessions_map=sessions_map,
            processed_dir=config.PROCESSED_DIR,
            n_components=args.speech_mixtures,
            models_dir=config.SPEECH_MODELS_DIR,
            digits=config.ACTIVE_DIGITS,
            force_retrain=args.force,
        )
        print(f"\n✅ Speech models trained: {len(speech_models)} models")
        print(f"   Saved to: {config.SPEECH_MODELS_DIR}")

    # ── Train Speaker Models ──────────────────────────────────────────────
    if not args.speech_only:
        print()
        print("─" * 65)
        print("  Training Speaker (Speaker Recognition) Models")
        print(f"  One GMM per (speaker × digit), {args.speaker_mixtures} components")
        print(f"  Total models to train: {len(speaker_ids)} × {len(config.ACTIVE_DIGITS)} = "
              f"{len(speaker_ids) * len(config.ACTIVE_DIGITS)}")
        print("─" * 65)

        speaker_models = train_all_speaker_models(
            speaker_ids=speaker_ids,
            sessions_map=sessions_map,
            processed_dir=config.PROCESSED_DIR,
            n_components=args.speaker_mixtures,
            models_dir=config.SPEAKER_MODELS_DIR,
            digits=config.ACTIVE_DIGITS,
            force_retrain=args.force,
        )
        total = sum(len(v) for v in speaker_models.values())
        print(f"\n✅ Speaker models trained: {total} models")
        print(f"   Saved to: {config.SPEAKER_MODELS_DIR}")

    print()
    print("=" * 65)
    print("  Step 2 complete! Next: python scripts/03_generate_forgeries.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
