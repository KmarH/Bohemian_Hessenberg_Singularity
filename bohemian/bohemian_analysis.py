"""
bohemian_analysis.py
====================
Core tools for Bohemian upper-Hessenberg matrices with alphabet {0, ..., m-1}.

Models
------
* Normalized: subdiagonal fixed to 1, upper triangle (i <= j) iid U{0..m-1}.
* Free:       subdiagonal also iid U{0..m-1} (it can vanish).

Contents
--------
* Generators for both models.
* The continuant recurrence for the leading principal minors d_0, ..., d_n
  (exact Python integers, O(n^2)); H is singular iff d_n == 0.
* Bareiss integer elimination (independent exact determinant) and an exact
  rank over Q, used for cross-checks and for the corank of singular matrices.
* Classification of why a singular matrix is singular (structural vs genuine).
* Exhaustive enumeration with exact rational probabilities.
* Exact terms of the collision lower bound (Theorem 4).
"""
import itertools
from fractions import Fraction
from math import prod

import numpy as np


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def make_hessenberg(n, m, normalized=True, rng=None):
    """Random Bohemian upper-Hessenberg matrix.
    normalized=True: subdiagonal fixed at 1 (Corless-Thornton convention).
    normalized=False: subdiagonal also drawn from {0..m-1} (can be 0 ->
    reducible -> reintroduces structural singularity)."""
    if rng is None:
        rng = np.random.default_rng()
    H = np.zeros((n, n), dtype=np.int64)
    iu = np.triu_indices(n)
    H[iu] = rng.integers(0, m, size=len(iu[0]))
    for i in range(1, n):
        H[i, i - 1] = 1 if normalized else rng.integers(0, m)
    return H


# ---------------------------------------------------------------------------
# Exact determinants
# ---------------------------------------------------------------------------
def hessenberg_minors(H):
    """Sequence d_0=1, d_1, ..., d_n of leading principal minors via the
    Hessenberg continuant recurrence
        d_k = sum_j (-1)^{k-j} h_{j,k} (prod_{i=j}^{k-1} s_i) d_{j-1},
    with s_i = h_{i+1,i} the subdiagonal. H is singular iff d_n == 0."""
    n = len(H)
    d = [1]
    for k in range(1, n + 1):
        acc = 0
        sprod = 1
        for j in range(k, 0, -1):          # j = k, k-1, ..., 1
            acc += ((-1) ** (k - j)) * int(H[j - 1, k - 1]) * sprod * d[j - 1]
            if j - 1 >= 1:
                sprod *= int(H[j - 1, j - 2])   # subdiagonal s_{j-1}
        d.append(acc)
    return d


def bareiss_det(A):
    """Exact integer determinant by fraction-free (Bareiss) elimination."""
    A = [[int(x) for x in row] for row in A]
    n = len(A)
    prev, sign = 1, 1
    for i in range(n - 1):
        if A[i][i] == 0:
            piv = next((r for r in range(i + 1, n) if A[r][i] != 0), None)
            if piv is None:
                return 0
            A[i], A[piv] = A[piv], A[i]
            sign = -sign
        for r in range(i + 1, n):
            for c in range(i + 1, n):
                A[r][c] = (A[r][c] * A[i][i] - A[r][i] * A[i][c]) // prev
        prev = A[i][i]
    return sign * A[n - 1][n - 1]


def exact_rank(A):
    """Rank over Q by Gaussian elimination in rational arithmetic."""
    M = [[Fraction(int(x)) for x in row] for row in A]
    rows, cols = len(M), len(M[0])
    r = 0
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        for i in range(r + 1, rows):
            f = M[i][c] / M[r][c]
            if f:
                M[i] = [a - f * b for a, b in zip(M[i], M[r], strict=True)]
        r += 1
        if r == rows:
            break
    return r


# ---------------------------------------------------------------------------
# Why is a singular matrix singular?
# ---------------------------------------------------------------------------
def _proportional(u, v):
    """Exact test: u, v nonzero, distinct and parallel (u_i v_j == u_j v_i)."""
    if u == v:
        return False
    return all(u[i] * v[j] == u[j] * v[i]
               for i in range(len(u)) for j in range(i + 1, len(u)))


def why_singular(M):
    """Return the dominant reason a singular integer matrix is singular.
    Priority: zero row/col > equal rows > equal cols > proportional > genuine.
    'genuine' = dense, all rows/cols distinct & nonzero, non-proportional:
    the linear dependence is spread across the matrix."""
    M = np.asarray(M)
    n = len(M)
    rows = [tuple(int(x) for x in r) for r in M]
    cols = [tuple(int(x) for x in c) for c in M.T]
    if any(not any(r) for r in rows) or any(not any(c) for c in cols):
        return "zero_rowcol"
    if len(set(rows)) < n:
        return "equal_row"
    if len(set(cols)) < n:
        return "equal_col"
    if any(_proportional(a, b) for a, b in itertools.combinations(rows, 2)):
        return "prop_row"
    if any(_proportional(a, b) for a, b in itertools.combinations(cols, 2)):
        return "prop_col"
    return "genuine"


# ---------------------------------------------------------------------------
# Exhaustive enumeration (exact rational probabilities)
# ---------------------------------------------------------------------------
def exact_singular_probability(n, m, normalized=True):
    """P(det H = 0) as a Fraction, enumerating every matrix of the family.
    The normalized family has m^{n(n+1)/2} members; the free one m^{n(n+1)/2 + n-1}.
    The determinant is evaluated with the continuant recurrence in exact
    integers (column by column, so each column is enumerated only once per
    prefix)."""
    sub_choices = [(1,) * (n - 1)] if normalized else \
        list(itertools.product(range(m), repeat=n - 1))
    singular = 0
    for s in sub_choices:
        # states: Counter-like dict {(d_0..d_k): multiplicity}; d depends only on
        # columns 1..k, so we extend column by column.
        states = {(1,): 1}
        for k in range(1, n + 1):
            new = {}
            for d, mult in states.items():
                for col in itertools.product(range(m), repeat=k):
                    acc = 0
                    for j in range(1, k + 1):
                        sprod = prod(s[j - 1:k - 1])
                        acc += (-1) ** (k - j) * col[j - 1] * sprod * d[j - 1]
                    key = (*d, acc)
                    new[key] = new.get(key, 0) + mult
            states = new
        singular += sum(mult for d, mult in states.items() if d[-1] == 0)
    total = m ** (n * (n + 1) // 2) * len(sub_choices)
    return Fraction(singular, total)

# ---------------------------------------------------------------------------
# Collision lower bound (Theorem 4)
# ---------------------------------------------------------------------------
def collision_lower_bound(n, m):
    """Exact terms of the collision lower bound for the normalized model, n >= 2:
        p_n >= (1/m) Coll_n >= (1/m) E[1/((m-1) Y_n + 1)] >= 1/(m((m-1)((m+1)/2)^(n-2) + 1)),
    with Y_n = |d_0| + ... + |d_{n-2}| and Coll_n = P(d_n = 0 | h_nn = 1), the probability
    that two independent copies of <c, y> coincide (y = signed d_0..d_{n-2}, c uniform).
    Every matrix of the family is enumerated, so all four values are exact Fractions
    (the last one is a float). Returns (p_n, Coll_n / m, collision bound, explicit bound)."""
    p = exact_singular_probability(n, m)
    coll, bound, count = Fraction(0), Fraction(0), 0
    # y depends only on columns 1..n-2; enumerate those and get the law of <c, y> exactly
    for flat in itertools.product(range(m), repeat=(n - 2) * (n - 1) // 2):
        d, pos = [1], 0
        for k in range(1, n - 1):
            col = flat[pos:pos + k]
            pos += k
            d.append(sum((-1) ** (k - j) * col[j - 1] * d[j - 1] for j in range(1, k + 1)))
        law = {0: 1}
        for yj in d:                     # signs do not change the collision probability
            new = {}
            for s, w in law.items():
                for c in range(m):
                    new[s + c * yj] = new.get(s + c * yj, 0) + w
            law = new
        coll += Fraction(sum(w * w for w in law.values()), m ** (2 * (n - 1)))
        bound += Fraction(1, (m - 1) * sum(abs(v) for v in d) + 1)
        count += 1
    explicit = 1 / (m * ((m - 1) * ((m + 1) / 2) ** (n - 2) + 1))
    return p, coll / count / m, bound / count / m, explicit
