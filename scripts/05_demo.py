#!/usr/bin/env python3
"""
VoxDetect — scripts/05_demo.py
STEP 5: Interactive Demo — Test a Single Audio File

Authenticate any WAV file against any registered speaker.
Shows detailed per-segment results with log-likelihood scores.

Usage:
    python scripts/05_demo.py --audio path/to/audio.wav --speaker S01
    python scripts/05_demo.py --audio path/to/audio.wav --speaker S01 --plot

The system will:
    1. Run VAD on the audio → extract word segments
    2. For each segment: recognize word → identify speaker via LLH
    3. Print authentication decision + per-segment breakdown
    4. Highlight tampered segments if forged

You can also test with a pre-built forged/genuine file from data/forged/
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.segmentation import load_splits
from src.gmm_models import load_all_speech_models, load_all_speaker_models
from src.authenticator import authenticate_audio, print_authentication_result

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(
        description="VoxDetect Demo: Authenticate a single audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a genuine audio:
  python scripts/05_demo.py --audio data/forged/genuine/S1/S01_genuine.wav --speaker S01

  # Test a forged audio:
  python scripts/05_demo.py --audio data/forged/forged/S1/S01_vs_S05_forged.wav --speaker S01

  # With LLH comparison plot:
  python scripts/05_demo.py --audio data/forged/genuine/S1/S01_genuine.wav --speaker S01 --plot
        """
    )
    p.add_argument(
        "--audio", type=str, required=True,
        help="Path to the WAV audio file to authenticate."
    )
    p.add_argument(
        "--speaker", type=str, required=True,
        help="Claimed speaker ID (e.g., S01). Must be a registered speaker."
    )
    p.add_argument(
        "--speech-mixtures", type=int, default=config.SPEECH_GMM_COMPONENTS,
        help=f"Speech GMM components. Default: {config.SPEECH_GMM_COMPONENTS}"
    )
    p.add_argument(
        "--speaker-mixtures", type=int, default=config.SPEAKER_GMM_COMPONENTS,
        help=f"Speaker GMM components. Default: {config.SPEAKER_GMM_COMPONENTS}"
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Generate LLH comparison plots for each segment."
    )
    return p.parse_args()


def main():
    args = parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          VoxDetect — Audio Forgery Detection System          ║")
    print("║   Based on: Mubeen et al. (2021), Computers & Elec. Eng.    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print(f"  Audio file     : {args.audio}")
    print(f"  Claimed speaker: {args.speaker}")
    print()

    # ── Validate audio file ───────────────────────────────────────────────
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"❌ ERROR: Audio file not found: {audio_path}")
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────
    print("Loading models...")

    speech_models = load_all_speech_models(
        digits=config.ACTIVE_DIGITS,
        n_components=args.speech_mixtures,
        models_dir=config.SPEECH_MODELS_DIR,
    )
    if not speech_models:
        print("❌ ERROR: No speech models found.")
        print("   Run: python scripts/02_train_models.py")
        sys.exit(1)

    try:
        train_split, _ = load_splits(config.SPLITS_DIR)
    except FileNotFoundError:
        print("❌ ERROR: Split files not found.")
        print("   Run: python scripts/01_preprocess.py")
        sys.exit(1)

    speaker_ids = list(train_split.keys())

    if args.speaker not in speaker_ids:
        print(f"❌ ERROR: Speaker '{args.speaker}' not found in registered speakers.")
        print(f"   Available: {speaker_ids[:10]}{'...' if len(speaker_ids) > 10 else ''}")
        sys.exit(1)

    speaker_models = load_all_speaker_models(
        speaker_ids=speaker_ids,
        digits=config.ACTIVE_DIGITS,
        n_components=args.speaker_mixtures,
        models_dir=config.SPEAKER_MODELS_DIR,
    )
    total_spk = sum(len(v) for v in speaker_models.values())
    print(f"✅ Loaded {len(speech_models)} speech models, {total_spk} speaker models")
    print()

    # ── Authenticate ──────────────────────────────────────────────────────
    print("Running authentication...")
    print()

    result = authenticate_audio(
        audio_path=str(audio_path),
        claimed_speaker=args.speaker,
        speech_models=speech_models,
        speaker_models=speaker_models,
    )

    # ── Print result ──────────────────────────────────────────────────────
    print_authentication_result(result)

    # ── Summary decision ──────────────────────────────────────────────────
    print()
    if result.is_genuine:
        print("  ╔══════════════════════════════════════════════╗")
        print(f"  ║  ✅  DECISION: GENUINE                       ║")
        print(f"  ║  All {result.n_segments} segments belong to {result.claimed_speaker}         ║")
        print("  ╚══════════════════════════════════════════════╝")
    else:
        print("  ╔══════════════════════════════════════════════╗")
        print(f"  ║  🚨  DECISION: FORGED / TAMPERED             ║")
        print(f"  ║  {result.n_tampered}/{result.n_segments} segments do NOT belong to {result.claimed_speaker}  ║")
        print(f"  ║  Tampered positions: {result.tampered_positions}         ║")
        print("  ╚══════════════════════════════════════════════╝")

    # ── Optional: Generate plot ───────────────────────────────────────────
    if args.plot and result.segment_results:
        from src.evaluator import plot_speaker_llh_comparison
        print()
        print("Generating LLH comparison plot...")

        # Plot for the first segment
        first_seg = result.segment_results[0]
        out_file  = f"demo_llh_{args.speaker}_{audio_path.stem}.png"

        try:
            plot_speaker_llh_comparison(
                audio_path=str(audio_path),
                claimed_speaker=args.speaker,
                digit_idx=first_seg.recognized_digit if first_seg.recognized_digit >= 0 else 1,
                speech_models=speech_models,
                speaker_models=speaker_models,
                out_dir=config.FIGURES_DIR,
                filename=out_file,
            )
            print(f"✅ Plot saved: {config.FIGURES_DIR / out_file}")
        except Exception as e:
            print(f"  Plot generation failed: {e}")

    print()


if __name__ == "__main__":
    main()
