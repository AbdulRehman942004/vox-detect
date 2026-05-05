#!/usr/bin/env python3
"""
VoxDetect — scripts/01_preprocess.py
STEP 1: Voice Activity Detection + Word Segmentation

Reads all speaker audio files from data/raw/, applies VAD to segment
individual digit words, and saves them to data/processed/.
Also creates train/test session splits.

Usage:
    python scripts/01_preprocess.py [--speakers S01 S02 ...] [--force]
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.segmentation import (
    segment_all_speakers,
    create_train_test_splits,
    discover_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="VoxDetect Step 1: VAD + Word Segmentation"
    )
    p.add_argument(
        "--speakers", nargs="+", default=None,
        help="Specific speaker IDs to process (e.g., S01 S02). Default: all."
    )
    p.add_argument(
        "--raw-dir", type=Path, default=config.RAW_DIR,
        help=f"Raw dataset directory. Default: {config.RAW_DIR}"
    )
    p.add_argument(
        "--out-dir", type=Path, default=config.PROCESSED_DIR,
        help=f"Output directory for processed files. Default: {config.PROCESSED_DIR}"
    )
    p.add_argument(
        "--train-frac", type=float, default=config.TRAIN_SESSION_FRACTION,
        help=f"Fraction of sessions to use for training. Default: {config.TRAIN_SESSION_FRACTION}"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Force re-segmentation even if processed files exist."
    )
    return p.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()

    print("=" * 65)
    print("  VoxDetect — Step 1: Audio Preprocessing")
    print("=" * 65)
    print(f"  Raw data      : {args.raw_dir}")
    print(f"  Output        : {args.out_dir}")
    print(f"  Train fraction: {args.train_frac:.0%}")
    if args.speakers:
        print(f"  Speakers      : {args.speakers}")
    print()

    # ── Check dataset exists ──────────────────────────────────────────────
    if not args.raw_dir.exists():
        print(f"ERROR: Raw data directory not found: {args.raw_dir}")
        print()
        print("Please download the dataset:")
        print("  1. Go to: https://www.kaggle.com/datasets/mohamedanwarvic/merged-arabic-corpus-of-isolated-words")
        print("  2. Download and extract to: data/raw/")
        print("  3. Expected structure: data/raw/S01/, data/raw/S02/, ...")
        sys.exit(1)

    # ── Discover speakers ─────────────────────────────────────────────────
    try:
        speaker_files = discover_dataset(args.raw_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if args.speakers:
        missing = [s for s in args.speakers if s not in speaker_files]
        if missing:
            print(f"WARNING: Speakers not found in dataset: {missing}")
        speakers_to_process = [s for s in args.speakers if s in speaker_files]
    else:
        speakers_to_process = None   # All speakers

    print(f"Found {len(speaker_files)} speakers in dataset.")
    print(f"Processing: {'all' if speakers_to_process is None else len(speakers_to_process)} speakers")
    print()

    # ── VAD + Segmentation ────────────────────────────────────────────────
    logger.info("Starting VAD and word segmentation...")
    metadata = segment_all_speakers(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        speakers=speakers_to_process,
    )

    if not metadata:
        print("ERROR: No speakers were successfully segmented. Check your dataset.")
        sys.exit(1)

    total_files = sum(
        len(s_meta.get("saved_paths", []))
        for spk_data in metadata.values()
        for s_meta in spk_data.values()
    )
    print(f"\n✅ Segmentation complete: {total_files} word files saved")

    # ── Train/Test Split ──────────────────────────────────────────────────
    logger.info("Creating train/test splits...")
    train_split, test_split = create_train_test_splits(
        metadata=metadata,
        train_frac=args.train_frac,
        splits_dir=config.SPLITS_DIR,
    )

    n_train = sum(len(v) for v in train_split.values())
    n_test  = sum(len(v) for v in test_split.values())
    print(f"✅ Splits created: {n_train} train sessions, {n_test} test sessions")
    print(f"   Saved to: {config.SPLITS_DIR}")

    print()
    print("=" * 65)
    print("  Step 1 complete! Next: python scripts/02_train_models.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
