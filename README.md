# Singularity in Bohemian Upper-Hessenberg Matrices: Anatomy and Geometric Rate

School of Physics, University of Costa Rica. Presented as a scientific poster (`Hessenberg_Poster.pdf`) in 2026 at the Wilhelm und Else Heraeus-Stiftung School of Quantum Science and Technology, organized with the University of Stuttgart and the University of Costa Rica.

The project studies the probability that a random Bohemian upper-Hessenberg matrix is singular, and the reason why. The upper triangle is drawn uniformly from {0, ..., m-1}, and two models are compared: the normalized model, with the subdiagonal fixed to 1, and the free model, where the subdiagonal is also random.

Main results:

- In the normalized model the corank is at most 1, and structural singularity (zero, equal or proportional rows or columns) is exponentially rare. The poster states it as O(m^-n); the notebook gives the bound (2(n-1)(m-1)+2) m^-n, since row 1 can also match any later row. A singular matrix is therefore singular for a genuine arithmetic reason, which is the opposite of the dense i.i.d. case.
- Writing the determinant as a continuant recurrence gives P(singular) ≤ 1/m, attained at n = m = 2 (checked by exhaustive enumeration).
- P(singular) decays geometrically, as γ(m)^n, with γ(m) far above the 1/m of dense matrices (e.g. γ(2) ≈ 0.895, γ(10) ≈ 0.41). It is estimated with exact integer arithmetic and a Rao-Blackwell estimator.
- γ(m) = e^(-λ(m)) ρ(m), where λ(m) is the Lyapunov exponent of the minors (computed up to n = 1000) and ρ(m) is an arithmetic-resonance factor that grows from ≈ 1 at m = 2 to ≈ 1.33 at m = 10. The existence of γ(m) is left as a conjecture.
- The free model goes back to structural singularity, driven by zero subdiagonal entries, so the dense upper triangle is what drives the phenomenon.
- The condition number grows with m. It is offered as a baseline for the Hessenberg matrices of Krylov methods and open quantum systems, not as a model of them.

## Structure

| Path | Content |
|---|---|
| `Bohemian_Hessenberg.ipynb` | Walks through every computation of the project, in the order of the poster, with the outputs saved. |
| `bohemian/bohemian_analysis.py` | Generators for both models, the continuant recurrence, Bareiss determinant, exact rank, classification of singular matrices and exhaustive enumeration. |
| `bohemian/paper_experiments.py` | Monte Carlo singularity estimators for both models and the condition number. |
| `bohemian/gamma_estimation.py` | Rao-Blackwell estimator, the fit of γ(m) with bootstrap intervals, and the Lyapunov exponent. |
| `bohemian/bohemian_eigenvalues.py` | Eigenvalue density of Bohemian families, accumulated into a 2D histogram (Figure 1 of the poster). |
| `Hessenberg_Poster.pdf` | The poster. |
| `Original_Spanish_version/` | The same notebook in Spanish. |

The notebook runs in about 10 minutes with `FULL = False`; `FULL = True` uses the poster's 10^8 samples for the eigenvalue density. Requirements are in `requirements.txt` (Python 3.12).

## Versión en español

La versión en español del cuaderno está en `Original_Spanish_version/`.
