"""
VoxDetect — src/authenticator.py
Audio authentication and forgery/impostor detection.

Implements Section 2.4 of the paper:
    Mubeen et al. (2021) — "Audio authentication and tampering detection"

Decision Process (per audio):
    For each word segment Xₜ in the test audio:
        1. Word Recognition:
           digit* = argmax_d  log p(Xₜ | Θ_speech_d)
        
        2. Speaker Recognition:
           speaker* = argmax_k  log p(Xₜ | Θ_speaker_{k, digit*})
        
        3. Authenticity Check:
           If speaker* ≠ claimed_speaker → TAMPERED SEGMENT
    
    Final decision:
        If ANY segment is tampered → audio is FORGED
        If ALL segments match claimed speaker → audio is GENUINE

The core metric is the log-likelihood (LLH):
    log p(X | Θ) = Σ_t log p(xₜ | Θ)   (via GMM.score_samples)
"""

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.mixture import GaussianMixture

from src import config
from src.vad import load_and_vad, extract_word_signals
from src.features import extract_mfcc_matrix
from src.gmm_models import log_likelihood

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentResult:
    """Authentication result for a single word segment."""
    position:          int
    recognized_digit:  int                  # Word recognized by speech model
    digit_name:        str
    claimed_speaker:   str
    identified_speaker: str                 # Speaker with highest LLH
    llh_claimed:       float                # LLH under claimed speaker's model
    llh_best:          float                # Best LLH across all speakers
    llh_all_speakers:  Dict[str, float]     # {speaker_id: llh}
    is_tampered:       bool                 # True if identified ≠ claimed


@dataclass
class AuthenticationResult:
    """Full authentication result for an audio file."""
    claimed_speaker:   str
    is_genuine:        bool                 # True = GENUINE, False = FORGED
    decision:          str                  # "GENUINE" or "FORGED"
    n_segments:        int
    n_tampered:        int
    tampered_positions: List[int]           # 0-indexed positions of tampered segments
    segment_results:   List[SegmentResult]  # Per-segment details
    audio_path:        Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# CORE AUTHENTICATION LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def recognize_word(
    features:      np.ndarray,
    speech_models: Dict[int, GaussianMixture],
) -> Tuple[int, Dict[int, float]]:
    """
    Identify which word (digit) is spoken using speech GMMs.

    Computes log p(X | Θ_speech_d) for each digit d and returns the best.

    Args:
        features:      MFCC feature matrix (n_frames, n_features)
        speech_models: {digit_idx: gmm}

    Returns:
        best_digit:  Digit with highest LLH
        llh_dict:    {digit_idx: llh} for all digits
    """
    llh_dict: Dict[int, float] = {}
    for d, gmm in speech_models.items():
        try:
            llh_dict[d] = log_likelihood(gmm, features)
        except Exception:
            llh_dict[d] = float("-inf")

    if not llh_dict:
        return -1, {}

    best_digit = max(llh_dict, key=llh_dict.__getitem__)
    return best_digit, llh_dict


def recognize_speaker(
    features:        np.ndarray,
    digit_idx:       int,
    speaker_models:  Dict[str, Dict[int, GaussianMixture]],
) -> Tuple[str, Dict[str, float]]:
    """
    Identify the speaker for a given word segment.

    Computes log p(X | Θ_{speaker,digit}) for all registered speakers
    and returns the one with the highest LLH.

    This directly implements the decision from the paper (Section 2.4):
    "Θ_{2K} where K = 1,2,...,115; X represents the test utterance;
     115 LLH values are computed by comparing extracted MFCC of X with each model."

    Args:
        features:       MFCC feature matrix (n_frames, n_features)
        digit_idx:      Which digit this segment is (from word recognition)
        speaker_models: {speaker_id: {digit_idx: gmm}}

    Returns:
        best_speaker:  Speaker ID with highest LLH
        llh_dict:      {speaker_id: llh} for all speakers
    """
    llh_dict: Dict[str, float] = {}
    for spk, digit_map in speaker_models.items():
        if digit_idx not in digit_map:
            continue
        gmm = digit_map[digit_idx]
        try:
            llh_dict[spk] = log_likelihood(gmm, features)
        except Exception:
            llh_dict[spk] = float("-inf")

    if not llh_dict:
        return "UNKNOWN", {}

    best_speaker = max(llh_dict, key=llh_dict.__getitem__)
    return best_speaker, llh_dict


def authenticate_segment(
    word_signal:     np.ndarray,
    position:        int,
    claimed_speaker: str,
    speech_models:   Dict[int, GaussianMixture],
    speaker_models:  Dict[str, Dict[int, GaussianMixture]],
) -> SegmentResult:
    """
    Authenticate a single word segment.

    Args:
        word_signal:     Audio signal for this word segment
        position:        0-indexed position within the sentence
        claimed_speaker: Speaker who claims to have spoken this audio
        speech_models:   {digit_idx: gmm} for word recognition
        speaker_models:  {speaker_id: {digit_idx: gmm}} for speaker recognition

    Returns:
        SegmentResult
    """
    features = extract_mfcc_matrix(word_signal)

    # Step 1: Word recognition
    rec_digit, digit_llh = recognize_word(features, speech_models)

    # Step 2: Speaker recognition (using recognized digit's models)
    best_spk, spk_llh = recognize_speaker(features, rec_digit, speaker_models)

    # Log-likelihoods for claimed speaker
    llh_claimed = spk_llh.get(claimed_speaker, float("-inf"))
    llh_best    = spk_llh.get(best_spk, float("-inf"))

    # Step 3: Authenticity check
    is_tampered = (best_spk != claimed_speaker)

    return SegmentResult(
        position=position,
        recognized_digit=rec_digit,
        digit_name=config.DIGIT_NAMES[rec_digit] if 0 <= rec_digit < 10 else "unknown",
        claimed_speaker=claimed_speaker,
        identified_speaker=best_spk,
        llh_claimed=llh_claimed,
        llh_best=llh_best,
        llh_all_speakers=spk_llh,
        is_tampered=is_tampered,
    )


def authenticate_audio(
    audio_path:      str,
    claimed_speaker: str,
    speech_models:   Dict[int, GaussianMixture],
    speaker_models:  Dict[str, Dict[int, GaussianMixture]],
) -> AuthenticationResult:
    """
    Authenticate an audio file: determine if it's genuine or forged.

    Full pipeline:
        Load audio → VAD → segment words → authenticate each segment
        → declare GENUINE if all match, FORGED if any tampered

    Args:
        audio_path:      Path to .wav file
        claimed_speaker: Speaker ID (e.g., "S01")
        speech_models:   Trained speech (word) GMMs
        speaker_models:  Trained speaker GMMs

    Returns:
        AuthenticationResult
    """
    # Load and run VAD
    try:
        signal, regions = load_and_vad(audio_path)
    except Exception as e:
        logger.error(f"Could not load audio {audio_path}: {e}")
        return AuthenticationResult(
            claimed_speaker=claimed_speaker,
            is_genuine=False,
            decision="ERROR",
            n_segments=0,
            n_tampered=0,
            tampered_positions=[],
            segment_results=[],
            audio_path=audio_path,
        )

    word_signals = extract_word_signals(signal, regions)

    if not word_signals:
        logger.warning(f"No words detected in {audio_path}")
        return AuthenticationResult(
            claimed_speaker=claimed_speaker,
            is_genuine=False,
            decision="FORGED",      # No words = can't authenticate
            n_segments=0,
            n_tampered=0,
            tampered_positions=[],
            segment_results=[],
            audio_path=audio_path,
        )

    # Authenticate each segment
    segment_results: List[SegmentResult] = []
    for pos, word_sig in enumerate(word_signals):
        try:
            seg_result = authenticate_segment(
                word_sig, pos, claimed_speaker,
                speech_models, speaker_models
            )
        except Exception as e:
            logger.warning(f"  Segment {pos} authentication failed: {e}")
            # Treat failed segment as tampered (conservative decision)
            seg_result = SegmentResult(
                position=pos,
                recognized_digit=-1,
                digit_name="unknown",
                claimed_speaker=claimed_speaker,
                identified_speaker="UNKNOWN",
                llh_claimed=float("-inf"),
                llh_best=float("-inf"),
                llh_all_speakers={},
                is_tampered=True,
            )
        segment_results.append(seg_result)

    tampered_positions = [r.position for r in segment_results if r.is_tampered]
    is_genuine         = (len(tampered_positions) == 0)

    return AuthenticationResult(
        claimed_speaker=claimed_speaker,
        is_genuine=is_genuine,
        decision="GENUINE" if is_genuine else "FORGED",
        n_segments=len(segment_results),
        n_tampered=len(tampered_positions),
        tampered_positions=tampered_positions,
        segment_results=segment_results,
        audio_path=audio_path,
    )


def authenticate_signal(
    signal:          np.ndarray,
    claimed_speaker: str,
    speech_models:   Dict[int, GaussianMixture],
    speaker_models:  Dict[str, Dict[int, GaussianMixture]],
) -> AuthenticationResult:
    """
    Authenticate a pre-loaded audio signal (useful for in-memory processing).

    Same as authenticate_audio() but takes a numpy signal directly.
    """
    from src.vad import extract_word_regions

    regions      = extract_word_regions(signal)
    word_signals = extract_word_signals(signal, regions)

    if not word_signals:
        return AuthenticationResult(
            claimed_speaker=claimed_speaker,
            is_genuine=False, decision="FORGED",
            n_segments=0, n_tampered=0,
            tampered_positions=[], segment_results=[],
        )

    segment_results = []
    for pos, word_sig in enumerate(word_signals):
        try:
            seg_result = authenticate_segment(
                word_sig, pos, claimed_speaker,
                speech_models, speaker_models
            )
        except Exception as e:
            logger.warning(f"  Segment {pos} failed: {e}")
            seg_result = SegmentResult(
                position=pos, recognized_digit=-1,
                digit_name="unknown", claimed_speaker=claimed_speaker,
                identified_speaker="UNKNOWN",
                llh_claimed=float("-inf"), llh_best=float("-inf"),
                llh_all_speakers={}, is_tampered=True,
            )
        segment_results.append(seg_result)

    tampered_positions = [r.position for r in segment_results if r.is_tampered]
    is_genuine = (len(tampered_positions) == 0)

    return AuthenticationResult(
        claimed_speaker=claimed_speaker,
        is_genuine=is_genuine,
        decision="GENUINE" if is_genuine else "FORGED",
        n_segments=len(segment_results),
        n_tampered=len(tampered_positions),
        tampered_positions=tampered_positions,
        segment_results=segment_results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRETTY PRINT RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def print_authentication_result(result: AuthenticationResult) -> None:
    """Print a detailed authentication result to stdout."""
    line = "─" * 60
    print(line)
    print(f"  Claimed Speaker : {result.claimed_speaker}")
    print(f"  Decision        : {'✅ GENUINE' if result.is_genuine else '🚨 FORGED'}")
    print(f"  Segments        : {result.n_segments} total, {result.n_tampered} tampered")
    if result.audio_path:
        print(f"  Audio           : {result.audio_path}")
    print(line)
    print(f"  {'Pos':>3}  {'Digit':>8}  {'Identified Speaker':>20}  {'LLH(claimed)':>14}  {'LLH(best)':>10}  Status")
    print(f"  {'─'*3}  {'─'*8}  {'─'*20}  {'─'*14}  {'─'*10}  {'─'*10}")
    for r in result.segment_results:
        status = "🚨 TAMPERED" if r.is_tampered else "✅ OK"
        print(f"  {r.position:>3}  {r.digit_name:>8}  {r.identified_speaker:>20}  "
              f"{r.llh_claimed:>14.2f}  {r.llh_best:>10.2f}  {status}")
    print(line)


if __name__ == "__main__":
    print("Authenticator module ready.")
    print("Run scripts/04_evaluate.py for full evaluation.")
