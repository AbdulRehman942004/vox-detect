"""
VoxDetect — src/forgery_generator.py
Generate forged (spliced) audio sentences by mixing digits from different speakers.

Implements the forgery generation method from Section 2.2 of the paper:
    Mubeen et al. (2021) — "Generation of a forged speech signal" (Fig. 3)

Forgery approach:
    - Two speakers: a GENUINE speaker and an IMPOSTOR speaker
    - Some digit words in a sentence are replaced by the impostor's voice
    - The impostor's segments are inserted seamlessly (silence padding)

Four test sentences (digits in parentheses):
    Sentence 1: [1, 2, 3, 4, 5, 6]
    Sentence 2: [2, 3, 4, 5, 6, 7]
    Sentence 3: [3, 4, 5, 6, 7, 8]
    Sentence 4: [4, 5, 6, 7, 8, 9]

Forgery positions (0-indexed within sentence): digits at index 1, 2, 5
    → pattern: [GENUINE, IMPOSTOR, IMPOSTOR, GENUINE, GENUINE, IMPOSTOR]
"""

import json
import logging
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

from src import config
from src.segmentation import get_digit_paths

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Silence padding between words in a sentence (samples @ 16kHz)
INTER_WORD_SILENCE_MS = 150
INTER_WORD_SILENCE    = int(config.TARGET_SR * INTER_WORD_SILENCE_MS / 1000)  # 2400 samples


# ─────────────────────────────────────────────────────────────────────────────
# CORE: Build a sentence from word audio clips
# ─────────────────────────────────────────────────────────────────────────────

def build_sentence_signal(
    word_signals: List[np.ndarray],
    silence_samples: int = INTER_WORD_SILENCE,
) -> np.ndarray:
    """
    Concatenate word signals into a sentence with silence padding between words.

    Args:
        word_signals:    List of 1-D float arrays (one per word)
        silence_samples: Number of silence samples to insert between words

    Returns:
        sentence: 1-D float array of the full sentence
    """
    silence = np.zeros(silence_samples, dtype=np.float32)
    parts   = []
    for i, sig in enumerate(word_signals):
        parts.append(sig.astype(np.float32))
        if i < len(word_signals) - 1:
            parts.append(silence)
    return np.concatenate(parts)


def load_digit_sample(
    speaker_id:    str,
    digit_idx:     int,
    processed_dir: Path = config.PROCESSED_DIR,
    sessions:      Optional[List[str]] = None,
    rng:           Optional[np.random.RandomState] = None,
) -> Optional[np.ndarray]:
    """
    Load a random sample of a digit from a speaker's processed files.

    Args:
        speaker_id:    e.g. "S01"
        digit_idx:     0-based digit index
        processed_dir: Root of processed data
        sessions:      Optional list of sessions to choose from
        rng:           Random state for reproducible sampling

    Returns:
        signal: 1-D float array, or None if no file found
    """
    paths = get_digit_paths(speaker_id, digit_idx, processed_dir, sessions)
    if not paths:
        return None

    if rng is not None:
        path = paths[rng.randint(0, len(paths))]
    else:
        path = random.choice(paths)

    try:
        signal, _ = librosa.load(str(path), sr=config.TARGET_SR, mono=True)
        return signal.astype(np.float32)
    except Exception as e:
        logger.warning(f"Could not load {path}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GENUINE SENTENCE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_genuine_sentence(
    speaker_id:      str,
    sentence_digits: List[int],          # e.g. [1, 2, 3, 4, 5, 6]
    processed_dir:   Path = config.PROCESSED_DIR,
    sessions:        Optional[List[str]] = None,
    rng:             Optional[np.random.RandomState] = None,
) -> Optional[Tuple[np.ndarray, List[Dict]]]:
    """
    Generate a genuine sentence: all words spoken by the same genuine speaker.

    Args:
        speaker_id:      The genuine speaker
        sentence_digits: Ordered list of digit indices for the sentence
        processed_dir:   Root processed directory
        sessions:        Session IDs to draw from (test sessions)
        rng:             Random state

    Returns:
        (sentence_signal, segment_metadata_list) or None if failed
    """
    word_signals = []
    metadata     = []

    for pos, digit_idx in enumerate(sentence_digits):
        sig = load_digit_sample(speaker_id, digit_idx, processed_dir, sessions, rng)
        if sig is None:
            logger.debug(f"  Missing digit {digit_idx} for speaker {speaker_id}")
            return None
        word_signals.append(sig)
        metadata.append({
            "position":    pos,
            "digit_idx":   digit_idx,
            "digit_name":  config.DIGIT_NAMES[digit_idx],
            "speaker":     speaker_id,
            "is_impostor": False,
        })

    sentence = build_sentence_signal(word_signals)
    return sentence, metadata


# ─────────────────────────────────────────────────────────────────────────────
# FORGED SENTENCE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_forged_sentence(
    genuine_speaker:     str,
    impostor_speaker:    str,
    sentence_digits:     List[int],
    forged_positions:    List[int] = config.FORGED_DIGIT_POSITIONS,
    processed_dir:       Path = config.PROCESSED_DIR,
    genuine_sessions:    Optional[List[str]] = None,
    impostor_sessions:   Optional[List[str]] = None,
    rng:                 Optional[np.random.RandomState] = None,
) -> Optional[Tuple[np.ndarray, List[Dict]]]:
    """
    Generate a forged sentence by mixing digits from two speakers.

    The genuine speaker's sentence has some digit positions replaced
    by the impostor speaker's voice.

    Args:
        genuine_speaker:    The claimed/genuine speaker ID
        impostor_speaker:   The impostor speaker ID
        sentence_digits:    Digit indices for the sentence
        forged_positions:   0-indexed positions within the sentence where
                            the impostor's voice replaces the genuine speaker
        processed_dir:      Root processed directory
        genuine_sessions:   Sessions to use for genuine speaker
        impostor_sessions:  Sessions to use for impostor
        rng:                Random state

    Returns:
        (forged_signal, segment_metadata_list) or None if failed
    """
    word_signals = []
    metadata     = []

    for pos, digit_idx in enumerate(sentence_digits):
        is_impostor = pos in forged_positions

        if is_impostor:
            sig = load_digit_sample(
                impostor_speaker, digit_idx, processed_dir, impostor_sessions, rng
            )
            spk = impostor_speaker
        else:
            sig = load_digit_sample(
                genuine_speaker, digit_idx, processed_dir, genuine_sessions, rng
            )
            spk = genuine_speaker

        if sig is None:
            logger.debug(
                f"  Missing digit {digit_idx} for "
                f"{'impostor' if is_impostor else 'genuine'} speaker {spk}"
            )
            return None

        word_signals.append(sig)
        metadata.append({
            "position":    pos,
            "digit_idx":   digit_idx,
            "digit_name":  config.DIGIT_NAMES[digit_idx],
            "speaker":     spk,
            "is_impostor": is_impostor,
        })

    sentence = build_sentence_signal(word_signals)
    return sentence, metadata


# ─────────────────────────────────────────────────────────────────────────────
# BATCH GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    speaker_ids:       List[str],
    test_sessions_map: Dict[str, List[str]],  # {speaker_id: [test_session_ids]}
    out_dir:           Path = config.FORGED_DIR,
    sentences:         Dict[str, List[int]] = config.SENTENCES,
    forged_positions:  List[int] = config.FORGED_DIGIT_POSITIONS,
    processed_dir:     Path = config.PROCESSED_DIR,
    n_impostor_pairs:  int = -1,   # -1 = all combinations
    random_seed:       int = 42,
) -> Dict:
    """
    Generate the full forged + genuine audio dataset for evaluation.

    Structure saved to disk:
        out_dir/
            genuine/
                {sentence_name}/
                    {genuine_speaker}_{idx}.wav
            forged/
                {sentence_name}/
                    {genuine_speaker}_vs_{impostor_speaker}_{idx}.wav

    Args:
        speaker_ids:        List of all registered speaker IDs
        test_sessions_map:  Test session IDs per speaker
        out_dir:            Root output directory for generated audio
        sentences:          Dict of sentence_name → digit list
        forged_positions:   Which positions in a sentence use the impostor
        processed_dir:      Root of processed audio
        n_impostor_pairs:   Max impostor pairs per genuine speaker (-1 = all)
        random_seed:        For reproducibility

    Returns:
        dataset_info: Metadata dict describing all generated files
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(random_seed)
    dataset_info = {"genuine": {}, "forged": {}}

    for sent_name, sent_digits in sentences.items():
        logger.info(f"Generating sentence {sent_name}: digits {sent_digits}")

        gen_dir    = out_dir / "genuine" / sent_name
        forge_dir  = out_dir / "forged"  / sent_name
        gen_dir.mkdir(parents=True, exist_ok=True)
        forge_dir.mkdir(parents=True, exist_ok=True)

        dataset_info["genuine"][sent_name] = []
        dataset_info["forged"][sent_name]  = []

        # ── Genuine sentences ──
        for spk in speaker_ids:
            sessions = test_sessions_map.get(spk, None)
            result   = generate_genuine_sentence(
                spk, sent_digits, processed_dir, sessions, rng
            )
            if result is None:
                continue
            signal, meta = result
            fname = f"{spk}_genuine.wav"
            out_path = gen_dir / fname
            try:
                sf.write(str(out_path), signal, config.TARGET_SR, subtype="PCM_16")
                dataset_info["genuine"][sent_name].append({
                    "path":           str(out_path),
                    "claimed_speaker": spk,
                    "label":          "genuine",
                    "segments":       meta,
                })
            except Exception as e:
                logger.warning(f"  Could not write {out_path}: {e}")

        # ── Forged sentences ──
        for genuine_spk in speaker_ids:
            genuine_sessions = test_sessions_map.get(genuine_spk, None)

            # Pick impostors (all other speakers, optionally limited)
            impostors = [s for s in speaker_ids if s != genuine_spk]
            if n_impostor_pairs > 0:
                impostors = rng.choice(
                    impostors,
                    size=min(n_impostor_pairs, len(impostors)),
                    replace=False,
                ).tolist()

            for impostor_spk in impostors:
                impostor_sessions = test_sessions_map.get(impostor_spk, None)
                result = generate_forged_sentence(
                    genuine_spk, impostor_spk, sent_digits,
                    forged_positions, processed_dir,
                    genuine_sessions, impostor_sessions, rng,
                )
                if result is None:
                    continue
                signal, meta = result

                fname    = f"{genuine_spk}_vs_{impostor_spk}_forged.wav"
                out_path = forge_dir / fname
                try:
                    sf.write(str(out_path), signal, config.TARGET_SR, subtype="PCM_16")
                    dataset_info["forged"][sent_name].append({
                        "path":            str(out_path),
                        "claimed_speaker": genuine_spk,
                        "impostor":        impostor_spk,
                        "label":           "forged",
                        "segments":        meta,
                    })
                except Exception as e:
                    logger.warning(f"  Could not write {out_path}: {e}")

        n_gen   = len(dataset_info["genuine"][sent_name])
        n_forge = len(dataset_info["forged"][sent_name])
        logger.info(f"  {sent_name}: {n_gen} genuine, {n_forge} forged")

    # Save metadata
    meta_path = out_dir / "dataset_info.json"
    with open(meta_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    logger.info(f"Dataset info saved to {meta_path}")

    total_gen   = sum(len(v) for v in dataset_info["genuine"].values())
    total_forge = sum(len(v) for v in dataset_info["forged"].values())
    logger.info(f"Total: {total_gen} genuine, {total_forge} forged sentences.")

    return dataset_info


def load_dataset_info(forged_dir: Path = config.FORGED_DIR) -> Dict:
    """Load the saved dataset metadata JSON."""
    meta_path = forged_dir / "dataset_info.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Dataset info not found at {meta_path}. "
            f"Run scripts/03_generate_forgeries.py first."
        )
    with open(meta_path) as f:
        return json.load(f)


if __name__ == "__main__":
    print("Forgery Generator module ready.")
    print(f"Sentences: {config.SENTENCES}")
    print(f"Forged positions: {config.FORGED_DIGIT_POSITIONS}")
    print(f"Pattern: [G, I, I, G, G, I] for Sentence 1 (pos 1,2,5 = impostor)")
