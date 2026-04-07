import numpy as np
import yaml
from scipy.linalg import svd
from typing import Dict, Optional, Tuple


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _fix_sign_flip(
    eigvecs: np.ndarray,
    prev_eigvecs: Optional[np.ndarray],
) -> np.ndarray:
    """
    Enforce sign continuity of eigenvectors across consecutive timesteps.

    SVD is unique only up to sign: flipping a column of U and the corresponding
    row of V^T yields an equally valid decomposition. Without correction, PC
    loadings can randomly flip sign each step, making time-series uninterpretable.

    Fix: for each column k, if dot(u_k_new, u_k_prev) < 0, negate u_k_new.
    """
    if prev_eigvecs is None:
        return eigvecs
    for k in range(eigvecs.shape[1]):
        if np.dot(eigvecs[:, k], prev_eigvecs[:, k]) < 0:
            eigvecs[:, k] *= -1.0
    return eigvecs


def _select_num_components(eigenvalues: np.ndarray, variance_threshold: float) -> int:
    """Return minimum K s.t. top-K eigenvalues explain >= variance_threshold of total."""
    total = eigenvalues.sum()
    if total <= 0.0:
        return 1
    cumvar = np.cumsum(eigenvalues) / total
    k = int(np.searchsorted(cumvar, variance_threshold)) + 1
    return min(k, len(eigenvalues))


def decompose_covariance_series(
    cov_series: np.ndarray,
    variance_threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Perform SVD/PCA on each covariance matrix in a rolling series.

    For a real symmetric PSD matrix Sigma = U diag(s) U^T, scipy.linalg.svd
    returns singular values s (= eigenvalues, descending) and left singular
    vectors U (= eigenvectors). Eigenvalues are clamped to >= 0 to suppress
    floating-point artefacts in near-singular matrices.

    Parameters
    ----------
    cov_series         : (T, N, N) rolling EWM covariance matrices
    variance_threshold : fraction of variance required for dynamic K selection

    Returns
    -------
    dict with:
        eigenvalues       : (T, N)    descending eigenvalues per timestep
        eigenvectors      : (T, N, N) eigenvector matrix (columns = PCs)
        num_components    : (T,)      dynamic K retained per timestep
        absorption_ratios : (T,)      fraction of variance in top-K PCs
    """
    T, N, _ = cov_series.shape

    all_eigenvalues = np.zeros((T, N))
    all_eigenvectors = np.zeros((T, N, N))
    all_num_components = np.zeros(T, dtype=int)
    all_absorption_ratios = np.zeros(T)

    prev_eigvecs: Optional[np.ndarray] = None

    for t in range(T):
        sigma = cov_series[t]
        U, s, _ = svd(sigma, full_matrices=True)
        s = np.maximum(s, 0.0)              # numerical stability clamp
        U = _fix_sign_flip(U, prev_eigvecs)
        prev_eigvecs = U.copy()

        k = _select_num_components(s, variance_threshold)
        ar = s[:k].sum() / s.sum() if s.sum() > 0.0 else 0.0

        all_eigenvalues[t] = s
        all_eigenvectors[t] = U
        all_num_components[t] = k
        all_absorption_ratios[t] = ar

    return {
        "eigenvalues": all_eigenvalues,
        "eigenvectors": all_eigenvectors,
        "num_components": all_num_components,
        "absorption_ratios": all_absorption_ratios,
    }


def decompose_all(
    cov_results: Dict[int, Tuple[np.ndarray, int]],
    variance_threshold: float,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Run spectral decomposition for all half-life covariance series.

    Returns
    -------
    dict mapping half_life -> {eigenvalues, eigenvectors, num_components,
                                absorption_ratios, warmup_steps}
    """
    results: Dict[int, Dict[str, np.ndarray]] = {}
    for hl, (cov_series, warmup_steps) in cov_results.items():
        decomp = decompose_covariance_series(cov_series, variance_threshold)
        decomp["warmup_steps"] = warmup_steps
        results[hl] = decomp
    return results
