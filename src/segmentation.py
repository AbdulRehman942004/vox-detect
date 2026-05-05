"""
VoxDetect — src/segmentation.py
Word segmentation: processes raw speaker audio files, applies VAD,
and saves individual digit word WAV files into data/processed/.

Dataset structure expected in data/raw/:
    data/raw/
        S01/
            S01.1.digits.wav
            S01.1.words.wav
            S01.2.digits.wav
            ...
        S02/
        ...
        S50/

Each digits.wav file contains Arabic digits spoken in sequence (0–9).
We use VAD to segment out each individual digit.
"""

import os
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from src import config
from src.vad import load_and_vad, extract_word_signals, VoicedRegion

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_dataset(raw_dir: Path = config.RAW_DIR) -> Dict[str, List[Path]]:
    """
    Walk raw_dir and collect all .digits.wav files per speaker.

    Returns:
        speaker_files: dict mapping speaker_id → list of Path objects
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Please download the dataset and place it in: {raw_dir}"
        )

    speaker_files: Dict[str, List[Path]] = {}

    for speaker_dir in sorted(raw_dir.iterdir()):
        if not speaker_dir.is_dir():
            continue
        speaker_id = speaker_dir.name
        if not (speaker_id.startswith("S") and speaker_id[1:].isdigit()):
            continue

        digit_wavs = sorted(speaker_dir.glob("*.digits.wav"))
        if digit_wavs:
            speaker_files[speaker_id] = digit_wavs

    logger.info(f"Discovered {len(speaker_files)} speakers in {raw_dir}")
    return speaker_files


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE FILE SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def segment_digits_file(
    wav_path:      Path,
    speaker_id:    str,
    session_id:    str,
    out_dir:       Path = config.PROCESSED_DIR,
    expected_digits: int = 10,          # digits 0–9
    min_words:     int = 5,             # accept sessions with at least this many words
) -> Optional[Dict]:
    """
    Load a digits.wav file, run VAD, and save one WAV per digit.

    Naming convention:
        {out_dir}/{speaker_id}/digit_{d}/{speaker_id}_{session_id}_digit{d}.wav

    Args:
        wav_path:        Path to the .digits.wav file
        speaker_id:      e.g. "S01"
        session_id:      e.g. "1" (from filename)
        out_dir:         Root processed directory
        expected_digits: Expected number of digits in file (usually 10)
        min_words:       Minimum acceptable detected words

    Returns:
        metadata dict with keys: speaker_id, session_id, n_detected, saved_paths
        or None if segmentation failed badly
    """
    try:
        signal, regions = load_and_vad(str(wav_path))
    except Exception as e:
        logger.warning(f"  Could not load {wav_path}: {e}")
        return None

    n_detected = len(regions)

    if n_detected < min_words:
        logger.warning(
            f"  {speaker_id}/{session_id}: Only {n_detected} words detected "
            f"(expected ~{expected_digits}). Skipping."
        )
        return None

    word_signals = extract_word_signals(signal, regions)

    # If we detect more words than expected, merge short adjacent segments
    if n_detected > expected_digits:
        word_signals, regions = _merge_over_segmented(
            word_signals, regions, target=expected_digits
        )
        n_detected = len(word_signals)

    saved_paths: List[str] = []
    metadata = {
        "speaker_id":  speaker_id,
        "session_id":  session_id,
        "source_file": str(wav_path),
        "n_detected":  n_detected,
        "saved_paths": [],
    }

    for idx, (word_sig, region) in enumerate(zip(word_signals, regions)):
        # digit index: 0-based, mapping to Arabic digits 0–9
        # The dataset records digits in order: 0,1,2,...,9
        digit_idx = idx  # 0-based (digit_0 = Arabic zero, digit_1 = Arabic one, etc.)

        digit_dir = out_dir / speaker_id / f"digit_{digit_idx}"
        digit_dir.mkdir(parents=True, exist_ok=True)

        # Use zero-padded session ID in filename for consistency
        out_name = f"{speaker_id}_{session_id.zfill(2)}_digit{digit_idx}.wav"
        out_path = digit_dir / out_name

        try:
            sf.write(str(out_path), word_sig, config.TARGET_SR, subtype="PCM_16")
            saved_paths.append(str(out_path))
        except Exception as e:
            logger.warning(f"  Could not write {out_path}: {e}")

    metadata["saved_paths"] = saved_paths
    return metadata


def _merge_over_segmented(
    word_signals: List[np.ndarray],
    regions:      List[VoicedRegion],
    target:       int,
) -> Tuple[List[np.ndarray], List[VoicedRegion]]:
    """
    If VAD detected more segments than expected, merge the shortest pairs
    until we reach the target count.
    """
    signals = list(word_signals)
    regs    = list(regions)

    while len(signals) > target:
        # Find shortest consecutive pair
        durations = [len(s) for s in signals]
        min_idx   = int(np.argmin([durations[i] + durations[i + 1]
                                    for i in range(len(durations) - 1)]))
        # Merge min_idx and min_idx+1
        merged_sig = np.concatenate([signals[min_idx], signals[min_idx + 1]])
        merged_reg = VoicedRegion(
            start_sample=regs[min_idx].start_sample,
            end_sample=regs[min_idx + 1].end_sample,
            start_frame=regs[min_idx].start_frame,
            end_frame=regs[min_idx + 1].end_frame,
        )
        signals = signals[:min_idx] + [merged_sig] + signals[min_idx + 2:]
        regs    = regs[:min_idx] + [merged_reg] + regs[min_idx + 2:]

    return signals, regs


# ─────────────────────────────────────────────────────────────────────────────
# FULL DATASET SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def segment_all_speakers(
    raw_dir:  Path = config.RAW_DIR,
    out_dir:  Path = config.PROCESSED_DIR,
    speakers: Optional[List[str]] = None,
) -> Dict:
    """
    Process the entire dataset: for every speaker's every session,
    run VAD and save individual digit WAV files.

    Args:
        raw_dir:  Directory containing speaker subdirectories
        out_dir:  Output directory for processed files
        speakers: Optional list of speaker IDs to process (None = all)

    Returns:
        full_metadata: nested dict {speaker_id: {session_id: metadata}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    speaker_files = discover_dataset(raw_dir)

    if speakers is not None:
        speaker_files = {k: v for k, v in speaker_files.items() if k in speakers}

    full_metadata: Dict = {}
    total_saved = 0
    total_failed = 0

    for speaker_id, wav_list in tqdm(speaker_files.items(), desc="Speakers"):
        full_metadata[speaker_id] = {}
        logger.info(f"Processing speaker {speaker_id} ({len(wav_list)} sessions)")

        for wav_path in wav_list:
            # Parse session ID from filename
            # Dataset format: S01.01.digits.wav → session = "01"
            #                  S01.1.digits.wav  → session = "01" (normalize)
            parts      = wav_path.stem.split(".")
            session_id = parts[1].zfill(2) if len(parts) >= 2 else "00"

            meta = segment_digits_file(wav_path, speaker_id, session_id, out_dir)
            if meta is not None:
                full_metadata[speaker_id][session_id] = meta
                total_saved += len(meta["saved_paths"])
            else:
                total_failed += 1

    logger.info(f"Segmentation complete. Saved: {total_saved} files, Failed: {total_failed}")

    # Save metadata JSON
    meta_path = out_dir / "segmentation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(full_metadata, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")

    return full_metadata


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def create_train_test_splits(
    metadata:      Dict,
    train_frac:    float = config.TRAIN_SESSION_FRACTION,
    splits_dir:    Path  = config.SPLITS_DIR,
    random_seed:   int   = 42,
) -> Tuple[Dict, Dict]:
    """
    Split sessions per speaker into train and test sets.

    Strategy: For each speaker, sort sessions, take first ~70% for train,
    rest for test. This simulates enrollment (train) vs. verification (test).

    Args:
        metadata:   Segmentation metadata from segment_all_speakers()
        train_frac: Fraction of sessions to use for training
        splits_dir: Directory to save split JSON files
        random_seed: For reproducibility (shuffling)

    Returns:
        train_split, test_split: dicts {speaker_id: {session_id: metadata}}
    """
    rng = np.random.RandomState(random_seed)
    train_split: Dict = {}
    test_split:  Dict = {}

    for speaker_id, sessions in metadata.items():
        session_ids = sorted(sessions.keys())
        rng.shuffle(session_ids)  # Shuffle for random split

        n_train = max(1, int(len(session_ids) * train_frac))
        train_sessions = session_ids[:n_train]
        test_sessions  = session_ids[n_train:]

        train_split[speaker_id] = {s: sessions[s] for s in train_sessions if s in sessions}
        test_split[speaker_id]  = {s: sessions[s] for s in test_sessions  if s in sessions}

    # Save splits
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "train_split.json", "w") as f:
        json.dump(train_split, f, indent=2)
    with open(splits_dir / "test_split.json", "w") as f:
        json.dump(test_split, f, indent=2)

    # Summary
    n_train_sessions = sum(len(v) for v in train_split.values())
    n_test_sessions  = sum(len(v) for v in test_split.values())
    logger.info(f"Train sessions: {n_train_sessions}, Test sessions: {n_test_sessions}")

    return train_split, test_split


def load_splits(splits_dir: Path = config.SPLITS_DIR) -> Tuple[Dict, Dict]:
    """Load previously saved train/test splits."""
    train_path = splits_dir / "train_split.json"
    test_path  = splits_dir / "test_split.json"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Split files not found in {splits_dir}. Run 01_preprocess.py first."
        )
    with open(train_path) as f:
        train_split = json.load(f)
    with open(test_path) as f:
        test_split = json.load(f)
    return train_split, test_split


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Collect processed WAV paths for a speaker/digit
# ─────────────────────────────────────────────────────────────────────────────

def get_digit_paths(
    speaker_id:   str,
    digit_idx:    int,
    processed_dir: Path = config.PROCESSED_DIR,
    sessions:     Optional[List[str]] = None,
) -> List[Path]:
    """
    Get all processed WAV paths for a specific speaker + digit.

    Args:
        speaker_id:    e.g. "S01"
        digit_idx:     0-based digit index (0=zero, 1=one, ..., 9=nine)
        processed_dir: Root of processed data
        sessions:      Optional list of session IDs to include (None = all)

    Returns:
        paths: Sorted list of Path objects
    """
    digit_dir = processed_dir / speaker_id / f"digit_{digit_idx}"
    if not digit_dir.exists():
        return []

    all_paths = sorted(digit_dir.glob(f"{speaker_id}_*.wav"))

    if sessions is not None:
        filtered = []
        for p in all_paths:
            # filename: S01_01_digit1.wav → session part = "01"
            # normalize to match whether sessions are stored as "1" or "01"
            parts = p.stem.split("_")
            if len(parts) >= 2:
                file_sess_raw = parts[1]                # e.g. "01"
                file_sess_norm = file_sess_raw.zfill(2) # normalize to "01"
                # Accept if any provided session matches (raw or zero-padded)
                for s in sessions:
                    if s.zfill(2) == file_sess_norm:
                        filtered.append(p)
                        break
        return filtered

    return all_paths


def get_all_digit_paths_for_speaker(
    speaker_id:    str,
    processed_dir: Path = config.PROCESSED_DIR,
    sessions:      Optional[List[str]] = None,
) -> Dict[int, List[Path]]:
    """
    Get all digit WAV paths for a speaker, organized by digit index.

    Returns:
        {digit_idx: [path1, path2, ...]}
    """
    result = {}
    for d in range(10):
        paths = get_digit_paths(speaker_id, d, processed_dir, sessions)
        if paths:
            result[d] = paths
    return result


if __name__ == "__main__":
    print("Segmentation module ready.")
    print(f"Data expected at: {config.RAW_DIR}")
