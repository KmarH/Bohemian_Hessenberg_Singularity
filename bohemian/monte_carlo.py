"""
monte_carlo.py
==============
Monte Carlo estimators for the singularity of Bohemian upper-Hessenberg
matrices (normalized and free models) and the empirical distribution of the
condition number kappa = s_max / s_min.

Normalized model: subdiagonal = 1; upper triangle (i <= j) iid U{0..m-1}.
Free model:       subdiagonal also iid U{0..m-1} (can be 0).

Singularity = (d_n == 0), decided EXACTLY with the continuant recurrence
    d_k = sum_{j=1..k} (-1)^{k-j} h_{j,k} (prod s_i) d_{j-1},  d_0 = 1,
in batched int64 arithmetic. This avoids the floating-point determinant,
which is unreliable for n >~ 11 (Hadamard bound).
"""
import math

import numpy as np

RNG = np.random.default_rng(2025)


def _check_int64(n, m):
    """int64 is exact while the Hadamard bound of |d_n| stays below 2^62."""
    bound = ((m - 1) * math.sqrt(n) + 1) ** n
    if bound >= 2.0 ** 62:
        raise OverflowError(f"n={n}, m={m} may overflow int64 (bound {bound:.2e})")


# --------------------------------------------------------------------------
# Singularity estimators (integer recurrence, batched -> low memory).
# --------------------------------------------------------------------------
def psing_normalized(n, m, K, batch=50_000, rng=None):
    """Fraction of K sampled normalized matrices with d_n == 0."""
    rng = RNG if rng is None else rng
    _check_int64(n, m)
    singular, done = 0, 0
    iu = np.triu_indices(n)
    while done < K:
        b = min(batch, K - done)
        H = np.zeros((b, n, n), dtype=np.int64)
        H[:, iu[0], iu[1]] = rng.integers(0, m, size=(b, len(iu[0])))
        D = np.zeros((b, n + 1), dtype=np.int64)
        D[:, 0] = 1
        for k in range(1, n + 1):
            acc = np.zeros(b, dtype=np.int64)
            for j in range(1, k + 1):
                acc += ((-1) ** (k - j)) * H[:, j - 1, k - 1] * D[:, j - 1]
            D[:, k] = acc
        singular += int(np.sum(D[:, n] == 0))
        done += b
    return singular / K


def psing_free(n, m, K, batch=50_000, rng=None, return_samples=0):
    """Free model. Returns (P(d_n = 0), fraction of singular matrices with a
    zero subdiagonal entry). With return_samples > 0 it also returns up to
    that many singular matrices, for an exact corank check."""
    rng = RNG if rng is None else rng
    _check_int64(n, m)
    singular, with_zero, done = 0, 0, 0
    kept = []
    iu = np.triu_indices(n)
    while done < K:
        b = min(batch, K - done)
        H = np.zeros((b, n, n), dtype=np.int64)
        H[:, iu[0], iu[1]] = rng.integers(0, m, size=(b, len(iu[0])))
        sub = rng.integers(0, m, size=(b, n - 1))
        D = np.zeros((b, n + 1), dtype=np.int64)
        D[:, 0] = 1
        for k in range(1, n + 1):
            acc = np.zeros(b, dtype=np.int64)
            sprod = np.ones(b, dtype=np.int64)
            for j in range(k, 0, -1):
                acc += ((-1) ** (k - j)) * H[:, j - 1, k - 1] * sprod * D[:, j - 1]
                if j - 1 >= 1:
                    sprod = sprod * sub[:, j - 2]
            D[:, k] = acc
        sing = D[:, n] == 0
        singular += int(sing.sum())
        with_zero += int(np.sum(sing & np.any(sub == 0, axis=1)))
        if len(kept) < return_samples:
            for idx in np.flatnonzero(sing)[: return_samples - len(kept)]:
                M = H[idx].copy()
                M[np.arange(1, n), np.arange(n - 1)] = sub[idx]
                kept.append(M)
        done += b
    frac = with_zero / singular if singular else float("nan")
    if return_samples:
        return singular / K, frac, kept
    return singular / K, frac


# --------------------------------------------------------------------------
# Condition number
# --------------------------------------------------------------------------
def condition_numbers(n, m, K, normalized=True, batch=20_000, rng=None):
    """kappa = s_max / s_min of K sampled matrices. The exactly singular ones
    (flagged by the integer recurrence, not by a tiny s_min) are removed and
    counted separately, so the floating-point atom at kappa = inf does not
    contaminate the bulk. Returns (kappas, number_exactly_singular)."""
    rng = RNG if rng is None else rng
    _check_int64(n, m)
    ks, n_sing, done = [], 0, 0
    iu = np.triu_indices(n)
    while done < K:
        b = min(batch, K - done)
        H = np.zeros((b, n, n), dtype=np.int64)
        H[:, iu[0], iu[1]] = rng.integers(0, m, size=(b, len(iu[0])))
        sub = np.ones((b, n - 1), dtype=np.int64) if normalized \
            else rng.integers(0, m, size=(b, n - 1))
        H[:, np.arange(1, n), np.arange(n - 1)] = sub
        D = np.zeros((b, n + 1), dtype=np.int64)
        D[:, 0] = 1
        for k in range(1, n + 1):
            acc = np.zeros(b, dtype=np.int64)
            sprod = np.ones(b, dtype=np.int64)
            for j in range(k, 0, -1):
                acc += ((-1) ** (k - j)) * H[:, j - 1, k - 1] * sprod * D[:, j - 1]
                if j - 1 >= 1:
                    sprod = sprod * sub[:, j - 2]
            D[:, k] = acc
        regular = D[:, n] != 0
        n_sing += int((~regular).sum())
        s = np.linalg.svd(H[regular].astype(np.float64), compute_uv=False)
        ks.append(s[:, 0] / s[:, -1])
        done += b
    return np.concatenate(ks), n_sing
