"""Shared statistical utilities for IRIS modules."""
import numpy as np
import scipy.sparse

# Maximum dense array size in bytes before warning (2 GB)
_MAX_DENSE_BYTES = 2 * 1024 ** 3


def set_max_dense_bytes(n_bytes):
    """Set the module-level dense-array memory limit.

    :param n_bytes: maximum allowed dense array size in bytes.
    """
    global _MAX_DENSE_BYTES
    _MAX_DENSE_BYTES = int(n_bytes)


def bh_correct(pvalues):
    """Benjamini-Hochberg FDR correction.

    Handles NaN p-values by skipping them and preserving their position.

    :param pvalues: array-like of p-values (may contain NaN).
    :returns: array of adjusted p-values.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    if n == 0:
        return pvalues

    valid = ~np.isnan(pvalues)
    adjusted = np.full_like(pvalues, np.nan)

    if valid.sum() == 0:
        return adjusted

    valid_p = pvalues[valid]
    sort_idx = np.argsort(valid_p)
    sorted_p = valid_p[sort_idx]
    ranks = np.arange(1, len(sorted_p) + 1)
    adj_sorted = sorted_p * len(sorted_p) / ranks

    # Enforce monotonicity (from largest to smallest rank)
    for i in range(len(adj_sorted) - 2, -1, -1):
        adj_sorted[i] = min(adj_sorted[i], adj_sorted[i + 1])
    adj_sorted = np.minimum(adj_sorted, 1.0)

    unsort_idx = np.argsort(sort_idx)
    adjusted[valid] = adj_sorted[unsort_idx]
    return adjusted


def safe_toarray(sparse_mat, context=""):
    """Convert sparse matrix to dense with memory safety check.

    :param sparse_mat: scipy sparse matrix.
    :param context: description for error messages.
    :returns: dense numpy array.
    :raises MemoryError: if estimated size exceeds _MAX_DENSE_BYTES.
    """
    if not scipy.sparse.issparse(sparse_mat):
        return np.asarray(sparse_mat)

    n_elements = sparse_mat.shape[0] * sparse_mat.shape[1]
    est_bytes = n_elements * 8  # float64
    if est_bytes > _MAX_DENSE_BYTES:
        raise MemoryError(
            f"Densifying {sparse_mat.shape} matrix would require "
            f"~{est_bytes / 1e9:.1f} GB (limit: "
            f"{_MAX_DENSE_BYTES / 1e9:.0f} GB). {context}")
    return sparse_mat.toarray()
