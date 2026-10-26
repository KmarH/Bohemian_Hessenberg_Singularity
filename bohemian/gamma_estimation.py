"""
gamma_estimation.py
===================
Estimation of the geometric rate gamma(m) of P(d_n = 0) for the normalized
Bohemian upper-Hessenberg family, and of the Lyapunov exponent lambda(m).

  1. RAO-BLACKWELL estimator (variance reduced by ~ a factor m).
     d_n = (-1)^{n-1} h_{1n} + R, with R independent of h_{1n}.
     Instead of sampling h_{1n} and checking d_n == 0 (a noisy 0/1 event),
     h_{1n} is integrated analytically:
        P(d_n = 0 | rest) = (1/m) * 1{ (-1)^n R in {0, ..., m-1} }.
     Each contribution is 0 or 1/m instead of 0 or 1.

  2. THREE-PARAMETER FIT  log P(n) = a + n log(gamma) + alpha log(n),
     which separates the geometric base gamma from the polynomial prefactor.

  3. BOOTSTRAP confidence intervals. Since every contribution is 0 or 1/m,
     resampling K contributions with replacement is the same as drawing
     Binomial(K, p_hat) hits, which is what is done here (same distribution,
     without materializing K resamples).

  4. LYAPUNOV exponent lambda(m) = lim (1/n) log|d_n| (almost-sure growth
     rate), with the minor vector rescaled at every step to avoid overflow
     and one column drawn per step, so dimensions up to n = 1000 fit in memory.

int64 is exact within the ranges of n used for gamma.
"""
import math

import numpy as np

RNG = np.random.default_rng(20260803)


def psing_rb_contribs(n, m, K, batch=150_000, rng=None):
    """Rao-Blackwell contributions (each 0 or 1/m); their mean estimates P(d_n=0)."""
    rng = RNG if rng is None else rng
    iu = np.triu_indices(n)
    pos = {(r, c): t for t, (r, c) in enumerate(zip(iu[0], iu[1], strict=True))}
    chunks, done = [], 0
    while done < K:
        b = min(batch, K - done)
        U = rng.integers(0, m, size=(b, len(iu[0])))
        Dc = [np.ones(b, dtype=np.int64)]                 # d_0 = 1
        for k in range(1, n + 1):
            acc = np.zeros(b, dtype=np.int64)
            for j in range(1, k + 1):
                acc += ((-1) ** (k - j)) * U[:, pos[(j - 1, k - 1)]] * Dc[j - 1]
            Dc.append(acc)
        R = np.zeros(b, dtype=np.int64)                   # term without h_{1n}
        for j in range(2, n + 1):
            R += ((-1) ** (n - j)) * U[:, pos[(j - 1, n - 1)]] * Dc[j - 1]
        v = ((-1) ** n) * R
        chunks.append((((v >= 0) & (v <= m - 1)).astype(np.float64)) / m)
        done += b
    return np.concatenate(chunks)


def fit_gamma(ns, P):
    """Least squares on log P = a + n log(gamma) + alpha log(n)."""
    ns = np.asarray(ns, dtype=float)
    X = np.column_stack([np.ones(len(ns)), ns, np.log(ns)])
    c, *_ = np.linalg.lstsq(X, np.log(P), rcond=None)
    return math.exp(c[1]), c[2]                           # gamma, alpha


def estimate_gamma(m, ns, K, n_boot=2000, rng=None):
    """gamma(m) with a 95% bootstrap interval. Returns a dict with the
    point estimate, the interval, the prefactor exponent alpha (with its
    bootstrap interval) and the estimated P(n)."""
    rng = RNG if rng is None else rng
    hits = {n: round(psing_rb_contribs(n, m, K, rng=rng).sum() * m) for n in ns}
    P = np.array([hits[n] / (m * K) for n in ns])
    g, alpha = fit_gamma(ns, P)
    gs, als = [], []
    for _ in range(n_boot):
        Pb = np.array([rng.binomial(K, hits[n] / K) / (m * K) for n in ns])
        if np.all(Pb > 0):
            gb, ab = fit_gamma(ns, Pb)
            gs.append(gb)
            als.append(ab)
    return dict(gamma=g, ci=tuple(np.percentile(gs, [2.5, 97.5])), alpha=alpha,
                alpha_ci=tuple(np.percentile(als, [2.5, 97.5])), P=P, ns=list(ns))


def lyapunov(m, n=1000, K=2000, batch=500, checkpoints=(250, 500, 1000), rng=None):
    """lambda(m) from (1/n) log|d_n| with per-step rescaling.
    The vector (d_0, ..., d_k) is divided by its max-norm after every step and
    the log of the scale is accumulated, so log|d_k| = log|D_k| + sum(log scales).
    Returns {n_c: (mean, standard error)} for each checkpoint n_c <= n."""
    rng = RNG if rng is None else rng
    cps = sorted(c for c in checkpoints if c <= n)
    out = {c: [] for c in cps}
    done = 0
    while done < K:
        b = min(batch, K - done)
        D = np.zeros((b, n + 1))
        D[:, 0] = 1.0
        logscale = np.zeros(b)
        signs = (-1.0) ** np.arange(n + 1)
        for k in range(1, n + 1):
            col = rng.integers(0, m, size=(b, k)).astype(np.float64)   # h_{1..k, k}
            # sum_j (-1)^{k-j} h_{j,k} d_{j-1},  j = 1..k
            D[:, k] = (col * D[:, :k]) @ signs[k - 1::-1][:k] if False else \
                np.einsum("bj,bj,j->b", col, D[:, :k], signs[:k][::-1])
            scale = np.max(np.abs(D[:, : k + 1]), axis=1)
            D[:, : k + 1] /= scale[:, None]
            logscale += np.log(scale)
            if k in out:
                out[k].append((logscale + np.log(np.abs(D[:, k]))) / k)
        done += b
    res = {}
    for c in cps:
        v = np.concatenate(out[c])
        v = v[np.isfinite(v)]
        res[c] = (float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))))
    return res
