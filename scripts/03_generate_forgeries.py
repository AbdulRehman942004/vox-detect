#!/usr/bin/env python3
"""
VoxDetect — scripts/03_generate_forgeries.py
STEP 3: Generate Forged (Spliced) Audio Sentences

Creates the evaluation dataset of genuine and forged sentences.

For each genuine speaker:
    - Generate 1 genuine sentence (all digits from same speaker)
    - Generate N forged sentences (some digits replaced by impostors)

Forgery pattern (positions 1, 2, 5 replaced by impostor):
    [Genuine, IMPOSTOR, IMPOSTOR, Genuine, Genuine, IMPOSTOR]

For each of the 4 test sentences:
    - Sentence 1: digits [1,2,3,4,5,6]
    - Sentence 2: digits [2,3,4,5,6,7]
    - Sentence 3: digits [3,4,5,6,7,8]
    - Sentence 4: digits [4,5,6,7,8,9]

Usage:
    python scripts/03_generate_forgeries.py [--n-impostors 5] [--sentence S1]
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.segmentation import load_splits
from src.forgery_generator import generate_dataset, load_dataset_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="VoxDetect Step 3: Generate Forged Audio Dataset"
    )
    p.add_argument(
        "--n-impostors", type=int, default=10,
        help="Number of impostor speakers to mix per genuine speaker. "
             "-1 for all. Default: 10"
    )
    p.add_argument(
        "--sentences", nargs="+", default=list(config.SENTENCES.keys()),
        choices=list(config.SENTENCES.keys()),
        help="Which sentences to generate. Default: all (S1 S2 S3 S4)"
    )
    p.add_argument(
        "--out-dir", type=Path, default=config.FORGED_DIR,
        help=f"Output directory for forged audio. Default: {config.FORGED_DIR}"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42"
    )
    return p.parse_args()


def main():
    args = parse_args()
    config.ensure_dirs()

    print("=" * 65)
    print("  VoxDetect — Step 3: Forged Audio Generation")
    print("=" * 65)
    print(f"  Sentences        : {args.sentences}")
    print(f"  Impostors/speaker: {args.n_impostors} (-1 = all)")
    print(f"  Forged positions : {config.FORGED_DIGIT_POSITIONS} (0-indexed in sentence)")
    print(f"  Pattern          : [G, I, I, G, G, I] (G=genuine, I=impostor)")
    print(f"  Output           : {args.out_dir}")
    print()

    # ── Load test splits ──────────────────────────────────────────────────
    try:
        train_split, test_split = load_splits(config.SPLITS_DIR)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Run scripts/01_preprocess.py first.")
        sys.exit(1)

    # Build test sessions map: {speaker_id: [test_session_ids]}
    test_sessions_map = {
        spk: list(sess_data.keys())
        for spk, sess_data in test_split.items()
    }

    speaker_ids = list(test_split.keys())

    # Filter to requested sentences
    sentences = {k: v for k, v in config.SENTENCES.items() if k in args.sentences}

    print(f"  Speakers         : {len(speaker_ids)}")
    print(f"  Expected genuine : {len(speaker_ids) * len(sentences)}")
    print(f"  Expected forged  : ~{len(speaker_ids) * min(args.n_impostors, len(speaker_ids)-1) * len(sentences)}")
    print()

    # ── Generate dataset ──────────────────────────────────────────────────
    dataset_info = generate_dataset(
        speaker_ids=speaker_ids,
        test_sessions_map=test_sessions_map,
        out_dir=args.out_dir,
        sentences=sentences,
        forged_positions=config.FORGED_DIGIT_POSITIONS,
        processed_dir=config.PROCESSED_DIR,
        n_impostor_pairs=args.n_impostors,
        random_seed=args.seed,
    )

    # ── Summary ───────────────────────────────────────────────────────────
    total_gen   = sum(len(v) for v in dataset_info["genuine"].values())
    total_forge = sum(len(v) for v in dataset_info["forged"].values())

    print()
    print(f"✅ Generation complete!")
    print(f"   Genuine sentences : {total_gen}")
    print(f"   Forged sentences  : {total_forge}")
    print(f"   Total             : {total_gen + total_forge}")
    print()
    for s_name in args.sentences:
        n_g = len(dataset_info["genuine"].get(s_name, []))
        n_f = len(dataset_info["forged"].get(s_name, []))
        print(f"   {s_name}: {n_g} genuine, {n_f} forged")

    print()
    print("=" * 65)
    print("  Step 3 complete! Next: python scripts/04_evaluate.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
