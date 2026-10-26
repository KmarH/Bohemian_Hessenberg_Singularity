"""
gcd_resonance.py
================
Does the gcd of consecutive minors explain the resonance factor rho(m)?
A test with a pre-registered decision rule for the normalized Bohemian upper-Hessenberg family.

The paper's hypothesis H_gcd: rho(m) is quantified by how the law of
gcd(d_{n-2}, d_{n-1}) deviates from that of random integers. It makes two predictions:

  1. LAW. Where rho is largest (m = 10), consecutive minors share common factors more
     often than random integers: D = P(gcd = 1 | both nonzero) - 6/pi^2 < 0.

  2. RATE. Since rho(m) is a rate, the gcd effect compounds with n:
        B_nz(n) = P(d_n = 0 | both nonzero) / P(d_n = 0 | gcd = 1)
     grows like rho(m)^n, so delta = [log B_nz(n_hi) - log B_nz(n_lo)] / (n_hi - n_lo)
     is close to log rho(m). Two fixed dimensions with independent samples are used,
     so no model of the n-dependence is needed.

Singularity is estimated with the Rao-Blackwell contribution of step n,
w = (1/m) 1{(-1)^n R in {0..m-1}}, R = sum_{j>=2} (-1)^{n-j} h_{jn} d_{j-1}, whose
conditional mean given any event that does not involve h_{1n} (such as the gcd class)
is the conditional singularity probability. Matrices with a vanishing d_{n-2} or
d_{n-1} are excluded from the gcd comparison.

Decision rule (verdict_v2). It was fixed after an inconclusive first round (a straight-line fit
in n, vanishing minors not separated) and before the data of this replication:
  rate[m] = reject if the 95% upper bound of delta < r_min(m)/2, support if the 95%
            interval is above 0 and reaches r_min(m)/2, inconclusive otherwise;
  law     = reject if D(m=10) >= -3 se, support otherwise;
  overall = rejected if rate[5], rate[10] and law all reject; supported if rate[5]
            and rate[10] both support; inconclusive otherwise.
r_min(m) is the smallest plausible log rho(m) across fit forms.

int64 is exact in the ranges used (checked against the Hadamard bound).
"""
import math

import numpy as np

SEED = 2026102602
PLAN = {2: (8, 20, 1_000_000), 3: (8, 16, 2_000_000), 5: (8, 14, 3_000_000),
        10: (6, 11, 6_000_000)}                 # m: (n_lo, n_hi, samples per dimension)
R_MIN = {2: math.log(1.01), 3: math.log(1.02), 5: math.log(1.10), 10: math.log(1.29)}


def _check_int64(n, m):
    """int64 is exact while the Hadamard bound of |d_k| stays below 2^62."""
    if ((m - 1) * math.sqrt(n) + 1) ** n >= 2.0 ** 62:
        raise OverflowError(f"n={n}, m={m} may overflow int64")


def sample_step(n, m, K, rng, batch=100_000):
    """K matrices -> (a, b, w, Y): a = d_{n-2}, b = d_{n-1}, the Rao-Blackwell
    contribution w of step n, and Y = |d_0| + ... + |d_{n-2}|."""
    if n < 3:
        raise ValueError("n must be at least 3")
    _check_int64(n, m)
    iu = np.triu_indices(n)
    pos = {(r, c): t for t, (r, c) in enumerate(zip(iu[0], iu[1], strict=True))}
    a_out, b_out, w_out, y_out = [], [], [], []
    done = 0
    while done < K:
        bsz = min(batch, K - done)
        U = rng.integers(0, m, size=(bsz, len(iu[0])), dtype=np.int64)
        D = [np.ones(bsz, dtype=np.int64)]
        for k in range(1, n):                                  # d_1 .. d_{n-1}
            acc = np.zeros(bsz, dtype=np.int64)
            for j in range(1, k + 1):
                acc += (-1) ** (k - j) * U[:, pos[(j - 1, k - 1)]] * D[j - 1]
            D.append(acc)
        R = np.zeros(bsz, dtype=np.int64)
        for j in range(2, n + 1):
            R += (-1) ** (n - j) * U[:, pos[(j - 1, n - 1)]] * D[j - 1]
        v = (-1) ** n * R
        a_out.append(D[n - 2])
        b_out.append(D[n - 1])
        w_out.append(((v >= 0) & (v <= m - 1)).astype(np.float64) / m)
        y_out.append(np.sum(np.abs(np.stack(D[: n - 1])), axis=0))
        done += bsz
    return (np.concatenate(a_out), np.concatenate(b_out), np.concatenate(w_out),
            np.concatenate(y_out))


def gcd_class(a, b, gmax=10):
    """0 if a == 0 or b == 0, otherwise min(gcd(|a|, |b|), gmax)."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    g = np.minimum(np.gcd(np.abs(a), np.abs(b)), gmax)
    return np.where((a == 0) | (b == 0), 0, g).astype(np.int64)


def nonzero_split(n, m, K, rng, gmax=10):
    """Among samples with d_{n-2}, d_{n-1} both nonzero: gcd law (f1, D), conditional
    singularity by gcd group (q1, q_ge2), B_nz = p_nz / q1 and E = q_ge2 / q1."""
    a, b, w, _ = sample_step(n, m, K, rng)
    cls = gcd_class(a, b, gmax)
    nz = cls > 0
    wn, cn = w[nz], cls[nz]
    k_nz = int(nz.sum())
    one, ge2 = cn == 1, cn >= 2
    f1 = float(one.mean())
    f1_se = math.sqrt(f1 * (1 - f1) / k_nz)
    s, s1, s2 = wn.sum(), wn[one].sum(), wn[ge2].sum()
    n1, n2 = one.sum(), ge2.sum()
    p_nz = float(wn.mean())
    q1 = float(s1 / n1)
    q_ge2 = float(s2 / n2) if n2 else float("nan")
    # delta method over the k_nz i.i.d. nonzero samples (as for v1's log B)
    zb = wn / s - wn * one / s1 + one / n1
    if s2 > 0:
        ze = wn * ge2 / s2 - ge2 / n2 - wn * one / s1 + one / n1
        loge_se = float(math.sqrt(k_nz * ze.var(ddof=1)))
    else:
        loge_se = float("nan")
    return {"K_nz": k_nz, "f1": f1, "f1_se": f1_se, "D": f1 - 6 / math.pi ** 2, "D_se": f1_se,
            "p_nz": p_nz, "q1": q1, "q_ge2": q_ge2, "B_nz": p_nz / q1,
            "logB_nz_se": float(math.sqrt(k_nz * zb.var(ddof=1))),
            "E": q_ge2 / q1, "logE_se": loge_se,
            "hits": {1: int((wn[one] > 0).sum()), "ge2": int((wn[ge2] > 0).sum())}}


def fixed_pair_rate(m, n_lo, n_hi, K, rng):
    """Per-step growth of log B_nz between two fixed dimensions, from independent samples."""
    if not 3 <= n_lo < n_hi:
        raise ValueError("need 3 <= n_lo < n_hi")
    lo = nonzero_split(n_lo, m, K, rng)
    hi = nonzero_split(n_hi, m, K, rng)
    dn = n_hi - n_lo
    return {"lo": lo, "hi": hi,
            "delta": (math.log(hi["B_nz"]) - math.log(lo["B_nz"])) / dn,
            "delta_se": math.sqrt(lo["logB_nz_se"] ** 2 + hi["logB_nz_se"] ** 2) / dn}


def verdict_v2(rates, laws, r_min):
    """Pre-registered rule of gcd-v2-spec.md."""
    rate = {}
    for m, (delta, se) in rates.items():
        hi, lo = delta + 1.96 * se, delta - 1.96 * se
        if hi < r_min[m] / 2:
            rate[m] = "reject"
        elif lo > 0 and hi >= r_min[m] / 2:
            rate[m] = "support"
        else:
            rate[m] = "inconclusive"
    d, d_se = laws[10]
    law = "reject" if d >= -3 * d_se else "support"
    if rate[5] == rate[10] == "reject" and law == "reject":
        overall = "rejected"
    elif rate[5] == rate[10] == "support":
        overall = "supported"
    else:
        overall = "inconclusive"
    return {"rate": rate, "law": law, "overall": overall}


def run_gcd_test(seed=SEED, plan=PLAN, r_min=R_MIN):
    """The full test: fixed_pair_rate for every m of the plan, then the verdict.
    Returns (per-m results, verdict)."""
    ss = np.random.SeedSequence(seed)
    out, rates, laws = {}, {}, {}
    for m, (n_lo, n_hi, K) in plan.items():
        r = fixed_pair_rate(m, n_lo, n_hi, K, np.random.default_rng(ss.spawn(1)[0]))
        out[m] = r
        rates[m] = (r["delta"], r["delta_se"])
        laws[m] = (r["hi"]["D"], r["hi"]["D_se"])
    return out, verdict_v2(rates, laws, r_min)
