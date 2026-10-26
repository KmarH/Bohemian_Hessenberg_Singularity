"""
bohemian_eigenvalues.py
=======================
Generate "Bohemian" eigenvalue density images: the eigenvalues of many n x n
matrices whose entries are drawn from a finite alphabet (e.g. {-1,0,1} or
{-20,-1,0,1,20}), plotted as a density in the complex plane.

Features
--------
* Random sampling (default; required for large cases) OR exhaustive enumeration
  (small cases only -- guarded).
* Real OR complex alphabets (e.g. [1, 1j, -1, -1j]).
* Matrix families via `structure`: 'general', 'hessenberg' (unit subdiagonal,
  the normalized Bohemian family), 'tridiagonal'.
* Memory-safe: eigenvalues are accumulated into a fixed 2D histogram, so 10^8+
  eigenvalues never need to be stored at once.
* Classic renderings (green-on-black, jet, turbo, inferno) with log or power
  normalization to reveal the fractal-like structure.

Quick start
-----------
    H, ext = bohemian_density(5, [-20,-1,0,1,20], num_samples=20_000_000)
    render(H, ext, "out.png", cmap="turbo")

Requires: numpy, matplotlib.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm

# a green-on-black colormap in the spirit of the classic {-1,0,1} images
bohemian_green = LinearSegmentedColormap.from_list(
    "bohemian_green",
    ["#000000", "#003b1f", "#0b7a3b", "#4cd07d", "#d8ffe6"])


# ----------------------------------------------------------------- generators
def sample_matrices(n, alphabet, batch, rng, structure="general"):
    """A batch of random n x n matrices with entries drawn from `alphabet`."""
    a = alphabet
    if structure == "general":
        return a[rng.integers(0, len(a), size=(batch, n, n))]
    M = np.zeros((batch, n, n), dtype=a.dtype)
    iu = np.triu_indices(n)                               # upper triangle
    M[:, iu[0], iu[1]] = a[rng.integers(0, len(a), size=(batch, len(iu[0])))]
    if structure == "hessenberg":                        # unit subdiagonal
        for i in range(1, n):
            M[:, i, i - 1] = 1
    elif structure == "tridiagonal":
        T = np.zeros_like(M)
        for i in range(n):
            T[:, i, i] = M[:, i, i]
        for i in range(n - 1):
            T[:, i, i + 1] = M[:, i, i + 1]
            T[:, i + 1, i] = 1
        M = T
    else:
        raise ValueError(f"unknown structure {structure!r}")
    return M


def _eigs(M):
    return np.linalg.eigvals(M).ravel()


def _estimate_extent(n, alphabet, structure, seed, q=99.8,
                     probe=200_000, batch=50_000):
    rng = np.random.default_rng(seed)
    mags, done = [], 0
    while done < probe:
        b = min(batch, probe - done)
        mags.append(np.abs(_eigs(sample_matrices(n, alphabet, b, rng, structure))))
        done += b
    return float(np.percentile(np.concatenate(mags), q)) * 1.02


# --------------------------------------------------------------- density (MC)
def bohemian_density(n, alphabet, num_samples=20_000_000, bins=1400,
                     structure="general", extent=None, batch=50_000, seed=0):
    """
    Sample `num_samples` matrices and accumulate their eigenvalues into a
    bins x bins histogram over [-extent, extent]^2 (extent auto-estimated).
    Returns (Hist, extent) with Hist[row, col] indexed as (imag, real).
    """
    alphabet = np.asarray(alphabet)
    if extent is None:
        extent = _estimate_extent(n, alphabet, structure, seed + 1)
    edges = np.linspace(-extent, extent, bins + 1)
    Hacc = np.zeros((bins, bins), dtype=np.float64)
    rng = np.random.default_rng(seed)
    done = 0
    while done < num_samples:
        b = min(batch, num_samples - done)
        ev = _eigs(sample_matrices(n, alphabet, b, rng, structure))
        h, _, _ = np.histogram2d(ev.real, ev.imag, bins=[edges, edges])
        Hacc += h
        done += b
    return Hacc.T, extent


# ------------------------------------------------------- density (exhaustive)
def bohemian_exhaustive(n, alphabet, bins=1400, structure="general",
                        extent=None, chunk=200_000, cap=3e8):
    """
    Enumerate ALL matrices of the given family (small cases only). Raises if the
    count exceeds `cap`. Use bohemian_density (sampling) for large cases.
    """
    alphabet = np.asarray(alphabet)
    k = len(alphabet)
    nfree = n * n if structure == "general" else n * (n + 1) // 2
    total = k ** nfree
    if total > cap:
        raise ValueError(f"exhaustive would need {total:.2e} matrices "
                         f"(> cap {cap:.0e}); use bohemian_density(...) instead.")
    if extent is None:
        extent = _estimate_extent(n, alphabet, structure, 0)
    edges = np.linspace(-extent, extent, bins + 1)
    Hacc = np.zeros((bins, bins))
    iu = np.triu_indices(n)
    it = itertools.product(range(k), repeat=nfree)
    buf, cnt = [], 0
    def flush(idxs):
        idxs = np.asarray(idxs)
        b = len(idxs)
        M = np.zeros((b, n, n), dtype=alphabet.dtype)
        if structure == "general":
            M.reshape(b, -1)[:] = alphabet[idxs]
        else:
            M[:, iu[0], iu[1]] = alphabet[idxs]
            if structure == "hessenberg":
                for i in range(1, n):
                    M[:, i, i - 1] = 1
        ev = _eigs(M)
        h, _, _ = np.histogram2d(ev.real, ev.imag, bins=[edges, edges])
        return h
    for combo in it:
        buf.append(combo)
        cnt += 1
        if cnt >= chunk:
            Hacc += flush(buf)
            buf, cnt = [], 0
    if buf:
        Hacc += flush(buf)
    return Hacc.T, extent


# ---------------------------------------------------------------- rendering
def render(Hist, extent, fname, cmap="turbo", gamma=0.33, bg="black",
           mode="power", vmax_pct=99.9, dpi=170, size=8):
    """
    Render the density. mode='power' (PowerNorm, good default) or 'log'.
    Try cmap='turbo'/'jet'/'nipy_spectral' (colorful) or bohemian_green.
    Smaller gamma (0.25-0.4) brings up faint structure.
    With fname=None the figure is returned instead of saved (for notebooks).
    """
    fig, ax = plt.subplots(figsize=(size, size), facecolor=bg)
    ax.set_facecolor(bg)
    data = Hist
    if mode == "power":
        vmax = np.percentile(data[data > 0], vmax_pct)
        norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
        shown = data
    else:
        shown = np.where(data > 0, data, np.nan)
        norm = LogNorm(vmin=1, vmax=np.nanpercentile(shown, vmax_pct))
    ax.imshow(shown, origin="lower", extent=[-extent, extent, -extent, extent],
              cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0)
    if fname is None:
        return fig
    plt.savefig(fname, dpi=dpi, facecolor=bg, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    return None


if __name__ == "__main__":
    plt.switch_backend("Agg")
    # (b)-style: 5x5, alphabet {-20,-1,0,1,20}, sampled, colorful
    H, e = bohemian_density(5, [-20, -1, 0, 1, 20], num_samples=8_000_000,
                            bins=1200, seed=1)
    render(H, e, "demo_colorful.png", cmap="turbo", gamma=0.30)
    print(f"demo_colorful.png done, extent={e:.2f}")

    # (a)-style: 5x5, alphabet {-1,0,1}, sampled (exhaustive is 3^25, infeasible)
    H, e = bohemian_density(5, [-1, 0, 1], num_samples=8_000_000,
                            bins=1200, seed=2)
    render(H, e, "demo_green.png", cmap=bohemian_green, gamma=0.35)
    print(f"demo_green.png done, extent={e:.2f}")
