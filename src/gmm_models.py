"""
VoxDetect — src/gmm_models.py
Gaussian Mixture Model (GMM) training, saving, loading, and scoring.

Implements the GMM-based modeling described in the paper:
    Mubeen et al. (2021), Section 2.3

Two types of models (both use GMM; differ in # mixtures and training data):

    1. SPEECH MODELS (word recognition):
        - One GMM per digit word (9 models for digits 1–9)
        - Trained on MFCC features of that digit from ALL speakers
        - Optimal: 256 Gaussian components (Table 3 in paper)

    2. SPEAKER MODELS (speaker recognition):
        - One GMM per (speaker, digit) pair
        - Trained on MFCCs of that digit from ONE specific speaker
        - Optimal: 32 Gaussian components (Table 4 in paper)

GMM density: p(X|Θ) = Σᵢ wᵢ · g(X|μᵢ, Σᵢ)   (Eq. 3 in paper)
Decision:  argmax_k log p(X | Θ_k)             (log-likelihood scoring)
"""

import os
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from src import config
from src.features import features_from_files
from src.segmentation import get_digit_paths

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GMM CORE: Train / Score / Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def train_gmm(
    features:       np.ndarray,
    n_components:   int   = config.SPEECH_GMM_COMPONENTS,
    covariance_type: str  = config.GMM_COVARIANCE_TYPE,
    max_iter:       int   = config.GMM_MAX_ITER,
    n_init:         int   = config.GMM_N_INIT,
    tol:            float = config.GMM_TOL,
    reg_covar:      float = config.GMM_REG_COVAR,
    random_state:   int   = config.GMM_RANDOM_STATE,
) -> GaussianMixture:
    """
    Train a Gaussian Mixture Model on feature data.

    Internally uses:
      - k-means initialization (sklearn default when init_params='kmeans')
      - EM algorithm for parameter estimation (Eq. 3, 4, 5 in paper)

    Args:
        features:        2-D array (n_frames, n_features) — training MFCC data
        n_components:    Number of Gaussian mixture components
        covariance_type: 'diag' or 'full' (paper implies diagonal)
        max_iter:        Max EM iterations
        n_init:          Number of k-means restarts
        tol:             Convergence threshold
        random_state:    Random seed

    Returns:
        gmm: Trained GaussianMixture model
    """
    if features.shape[0] < n_components:
        # Not enough data: reduce components to avoid degenerate GMMs
        adjusted = max(2, features.shape[0] // 2)
        logger.warning(
            f"  Only {features.shape[0]} frames for {n_components} components. "
            f"Reducing to {adjusted}."
        )
        n_components = adjusted

    # sklearn GaussianMixture requires float64 for numerical stability
    features_64 = features.astype(np.float64)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        n_init=n_init,
        tol=tol,
        reg_covar=reg_covar,        # Regularization prevents ill-defined covariance
        random_state=random_state,
        init_params="kmeans",       # k-means initialization (paper method)
        verbose=0,
    )
    gmm.fit(features_64)

    if not gmm.converged_:
        logger.warning(f"  GMM did not converge in {max_iter} iterations.")

    return gmm


def log_likelihood(gmm: GaussianMixture, features: np.ndarray) -> float:
    """
    Compute the total log-likelihood of features under a GMM.

    log p(X | Θ) = Σ_frames log p(x_t | Θ)

    This is the core decision metric from the paper (Section 2.4):
    "log p(X | Θ_{speaker,word})" used to assign speakers.

    Args:
        gmm:      Trained GaussianMixture model
        features: 2-D array (n_frames, n_features)

    Returns:
        llh: Total log-likelihood (float, will be negative)
    """
    return float(gmm.score(features.astype(np.float64)) * features.shape[0])


def per_frame_log_likelihood(gmm: GaussianMixture, features: np.ndarray) -> np.ndarray:
    """
    Compute per-frame log-likelihood.

    Args:
        gmm:      Trained GaussianMixture model
        features: 2-D array (n_frames, n_features)

    Returns:
        llh_per_frame: 1-D array of shape (n_frames,)
    """
    return gmm.score_samples(features)


def save_model(gmm: GaussianMixture, path: Union[str, Path]) -> None:
    """Serialize and save a GMM model to disk using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(gmm, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Union[str, Path]) -> GaussianMixture:
    """Load a previously saved GMM model from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def model_path_speech(digit_idx: int,
                       n_components: int = config.SPEECH_GMM_COMPONENTS,
                       models_dir: Path = config.SPEECH_MODELS_DIR) -> Path:
    """Return the file path for a speech (word) model."""
    return models_dir / f"speech_digit{digit_idx}_gmm{n_components}.pkl"


def model_path_speaker(speaker_id: str,
                        digit_idx: int,
                        n_components: int = config.SPEAKER_GMM_COMPONENTS,
                        models_dir: Path = config.SPEAKER_MODELS_DIR) -> Path:
    """Return the file path for a speaker model."""
    return models_dir / speaker_id / f"speaker_{speaker_id}_digit{digit_idx}_gmm{n_components}.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# SPEECH MODELS (Word Recognition)
# ─────────────────────────────────────────────────────────────────────────────

def train_speech_model(
    digit_idx:     int,
    speaker_ids:   List[str],
    sessions_map:  Dict[str, List[str]],   # {speaker_id: [session_ids]}
    processed_dir: Path = config.PROCESSED_DIR,
    n_components:  int  = config.SPEECH_GMM_COMPONENTS,
    models_dir:    Path = config.SPEECH_MODELS_DIR,
    force_retrain: bool = False,
) -> Optional[GaussianMixture]:
    """
    Train a speech (word recognition) GMM for a specific digit.

    Pools MFCC features from ALL speakers' training sessions for this digit.
    One model per digit (digits 1–9 from paper; we do 0–9 for completeness).

    Args:
        digit_idx:     0-based digit index
        speaker_ids:   List of speaker IDs to include
        sessions_map:  {speaker_id: [session_ids_for_training]}
        processed_dir: Root of processed audio
        n_components:  GMM components (paper: 256)
        models_dir:    Where to save models
        force_retrain: If True, retrain even if model file exists

    Returns:
        gmm: Trained model (also saved to disk)
    """
    out_path = model_path_speech(digit_idx, n_components, models_dir)
    if out_path.exists() and not force_retrain:
        logger.info(f"  Speech model for digit {digit_idx} already exists. Skipping.")
        return load_model(out_path)

    # Collect WAV paths from all speakers
    all_paths = []
    for spk in speaker_ids:
        sessions = sessions_map.get(spk, None)
        paths    = get_digit_paths(spk, digit_idx, processed_dir, sessions)
        all_paths.extend(paths)

    if not all_paths:
        logger.warning(f"  No training files found for digit {digit_idx}. Skipping.")
        return None

    logger.info(f"  Training speech GMM digit {digit_idx}: "
                f"{len(all_paths)} files, {n_components} components")

    try:
        features = features_from_files(all_paths, verbose=False)
        gmm      = train_gmm(features, n_components=n_components)
        save_model(gmm, out_path)
        logger.info(f"  Saved → {out_path}")
        return gmm
    except Exception as e:
        logger.error(f"  Failed training speech GMM digit {digit_idx}: {e}")
        return None


def train_all_speech_models(
    speaker_ids:   List[str],
    sessions_map:  Dict[str, List[str]],
    processed_dir: Path = config.PROCESSED_DIR,
    n_components:  int  = config.SPEECH_GMM_COMPONENTS,
    models_dir:    Path = config.SPEECH_MODELS_DIR,
    digits:        List[int] = config.ACTIVE_DIGITS,
    force_retrain: bool = False,
) -> Dict[int, GaussianMixture]:
    """
    Train speech GMMs for all specified digits.

    Returns:
        {digit_idx: gmm}
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    speech_models: Dict[int, GaussianMixture] = {}

    logger.info(f"Training {len(digits)} speech models with {n_components} components...")
    for d in tqdm(digits, desc="Speech Models"):
        gmm = train_speech_model(
            d, speaker_ids, sessions_map, processed_dir,
            n_components, models_dir, force_retrain
        )
        if gmm is not None:
            speech_models[d] = gmm

    logger.info(f"Trained {len(speech_models)} speech models.")
    return speech_models


# ─────────────────────────────────────────────────────────────────────────────
# SPEAKER MODELS (Speaker Recognition)
# ─────────────────────────────────────────────────────────────────────────────

def train_speaker_model(
    speaker_id:    str,
    digit_idx:     int,
    sessions:      List[str],            # Training session IDs for this speaker
    processed_dir: Path = config.PROCESSED_DIR,
    n_components:  int  = config.SPEAKER_GMM_COMPONENTS,
    models_dir:    Path = config.SPEAKER_MODELS_DIR,
    force_retrain: bool = False,
) -> Optional[GaussianMixture]:
    """
    Train a speaker recognition GMM for one speaker × one digit.

    Only training (enrollment) sessions are used; test sessions are held out.

    Args:
        speaker_id:    e.g. "S01"
        digit_idx:     0-based digit index
        sessions:      List of session IDs to use for training
        processed_dir: Root of processed audio
        n_components:  GMM components (paper: 32)
        models_dir:    Where to save models
        force_retrain: Retrain even if model exists

    Returns:
        gmm: Trained model (saved to disk)
    """
    out_path = model_path_speaker(speaker_id, digit_idx, n_components, models_dir)
    if out_path.exists() and not force_retrain:
        return load_model(out_path)

    paths = get_digit_paths(speaker_id, digit_idx, processed_dir, sessions)
    if not paths:
        return None

    try:
        features = features_from_files(paths, verbose=False)
        gmm      = train_gmm(features, n_components=n_components)
        save_model(gmm, out_path)
        return gmm
    except Exception as e:
        logger.warning(f"  Could not train speaker {speaker_id} digit {digit_idx}: {e}")
        return None


def train_all_speaker_models(
    speaker_ids:   List[str],
    sessions_map:  Dict[str, List[str]],
    processed_dir: Path = config.PROCESSED_DIR,
    n_components:  int  = config.SPEAKER_GMM_COMPONENTS,
    models_dir:    Path = config.SPEAKER_MODELS_DIR,
    digits:        List[int] = config.ACTIVE_DIGITS,
    force_retrain: bool = False,
) -> Dict[str, Dict[int, GaussianMixture]]:
    """
    Train speaker GMMs for all speakers × all digits.

    Returns:
        {speaker_id: {digit_idx: gmm}}
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    speaker_models: Dict[str, Dict[int, GaussianMixture]] = {}

    logger.info(f"Training speaker models: {len(speaker_ids)} speakers × "
                f"{len(digits)} digits × {n_components} components")

    for spk in tqdm(speaker_ids, desc="Speaker Models"):
        speaker_models[spk] = {}
        sessions = sessions_map.get(spk, [])
        for d in digits:
            gmm = train_speaker_model(
                spk, d, sessions, processed_dir,
                n_components, models_dir, force_retrain
            )
            if gmm is not None:
                speaker_models[spk][d] = gmm

    total = sum(len(v) for v in speaker_models.values())
    logger.info(f"Trained {total} speaker models.")
    return speaker_models


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ALL SAVED MODELS
# ─────────────────────────────────────────────────────────────────────────────

def load_all_speech_models(
    digits:      List[int] = config.ACTIVE_DIGITS,
    n_components: int      = config.SPEECH_GMM_COMPONENTS,
    models_dir:  Path      = config.SPEECH_MODELS_DIR,
) -> Dict[int, GaussianMixture]:
    """Load all saved speech GMM models from disk."""
    models: Dict[int, GaussianMixture] = {}
    for d in digits:
        path = model_path_speech(d, n_components, models_dir)
        if path.exists():
            models[d] = load_model(path)
        else:
            logger.warning(f"  Speech model for digit {d} not found at {path}")
    return models


def load_all_speaker_models(
    speaker_ids:  List[str],
    digits:       List[int] = config.ACTIVE_DIGITS,
    n_components: int       = config.SPEAKER_GMM_COMPONENTS,
    models_dir:   Path      = config.SPEAKER_MODELS_DIR,
) -> Dict[str, Dict[int, GaussianMixture]]:
    """Load all saved speaker GMM models from disk."""
    speaker_models: Dict[str, Dict[int, GaussianMixture]] = {}
    for spk in speaker_ids:
        speaker_models[spk] = {}
        for d in digits:
            path = model_path_speaker(spk, d, n_components, models_dir)
            if path.exists():
                speaker_models[spk][d] = load_model(path)
    return speaker_models


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION: Evaluate different mixture counts (Tables 2, 3, 4 in paper)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mixture_counts(
    features_train: np.ndarray,
    features_test:  np.ndarray,
    mixture_counts: List[int],
    random_state:   int = config.GMM_RANDOM_STATE,
) -> Dict[int, float]:
    """
    Train and evaluate GMMs with different numbers of components.
    Used to reproduce Tables 2–4 in the paper.

    Args:
        features_train: Training feature matrix (n_frames, n_features)
        features_test:  Test feature matrix (n_frames, n_features)
        mixture_counts: List of component counts to try
        random_state:   For reproducibility

    Returns:
        {n_components: avg_log_likelihood_per_frame}
    """
    results: Dict[int, float] = {}
    for n in mixture_counts:
        try:
            gmm = train_gmm(features_train, n_components=n, random_state=random_state)
            llh = gmm.score(features_test)  # Average LLH per frame
            results[n] = float(llh)
        except Exception as e:
            logger.warning(f"  Failed for {n} components: {e}")
            results[n] = float("-inf")
    return results


def accuracy_from_llh_matrix(
    llh_matrix: np.ndarray,
    true_label: int,
) -> float:
    """
    Compute recognition accuracy given a matrix of LLH scores.

    Args:
        llh_matrix: (n_test_samples, n_classes) matrix of log-likelihoods
        true_label: Index of the correct class (column index)

    Returns:
        accuracy: Fraction of samples where the correct class has max LLH
    """
    predicted = np.argmax(llh_matrix, axis=1)
    correct   = (predicted == true_label).sum()
    return correct / len(predicted)


if __name__ == "__main__":
    print("GMM Models module ready.")
    print(f"Speech models dir: {config.SPEECH_MODELS_DIR}")
    print(f"Speaker models dir: {config.SPEAKER_MODELS_DIR}")

    # Smoke test: train a tiny GMM on random data
    X = np.random.randn(500, config.N_FEATURES).astype(np.float32)
    gmm = train_gmm(X, n_components=4)
    llh = log_likelihood(gmm, X)
    print(f"Smoke test GMM: LLH = {llh:.2f}  |  Converged: {gmm.converged_}")
    print("✓ GMM module OK")
