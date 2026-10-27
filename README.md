# Singularity in Bohemian Upper-Hessenberg Matrices: Anatomy and Geometric Rate

School of Physics, University of Costa Rica. Presented in 2026 at the Wilhelm und Else Heraeus-Stiftung School of Quantum Science and Technology, organized with the University of Stuttgart and the University of Costa Rica.

The project studies the probability that a random Bohemian upper-Hessenberg matrix is singular, and the reason why. The upper triangle is drawn uniformly from {0, ..., m-1}, and two models are compared: the normalized model, with the subdiagonal fixed to 1, and the free model, where the subdiagonal is also random.

Main results:

- In the normalized model the corank is at most 1, and structural singularity (zero, equal or proportional rows or columns) has probability at most (2(n-1)(m-1)+2) m^-n.
- Writing the determinant as a continuant recurrence gives P(singular) ≤ 1/m, attained at n = m = 2 (checked by exhaustive enumeration).
- A collision argument on the last two columns gives the lower bound P(singular) ≥ m^-2 (2/(m+1))^(n-2), whose base exceeds 1/m. Together with the structural bound, this proves that a singular matrix of large dimension is singular for a genuine arithmetic reason, the opposite of the dense i.i.d. case.
- The minors' second moments obey a closed 2×2 recursion, even though the recurrence has unbounded memory. This bounds the growth of the minors and raises the proved lower rate to max(ρ₂^(-1/2), 2/(m+1)), 0.874 at m = 2 against a measured 0.894.
- P(singular) → 0 for every m, at rate O(n^(-1/2)). The bound is elementary (Littlewood-Offord); the measured decay is geometric.
- P(singular) decays geometrically, as γ(m)^n, with γ(m) far above the 1/m of dense matrices (e.g. γ(2) ≈ 0.894, γ(10) ≈ 0.43). It is estimated with exact integer arithmetic and a Rao-Blackwell estimator.
- γ(m) = e^(-λ(m)) ρ(m), where λ(m) is the Lyapunov exponent of the minors (computed up to n = 1000) and ρ(m) is an arithmetic-resonance factor that grows from ≈ 1 at m = 2 to ≈ 1.39 at m = 10 (1.29 to 1.39 depending on the form of the fit). The collision bound proves ρ(m) ≥ 1 whenever λ(m) and γ(m) exist; the existence of γ(m) is left as a conjecture.
- A replication with a decision rule fixed before its data rejects the gcd of consecutive minors as the source of ρ(m): where ρ is largest (m = 10), consecutive minors are coprime as often as random integers (to within 0.0004 at 95% confidence), and the gcd effect does not grow with n. The origin of ρ(m) remains open.
- The free model goes back to structural singularity, driven by zero subdiagonal entries, so the dense upper triangle is what drives the phenomenon.
- The condition number grows with m at fixed n. It is a baseline for the conditioning of the upper-Hessenberg shape class, not a model of the Hessenberg matrices produced by Krylov methods.

## Structure

| Path | Content |
|---|---|
| `Bohemian_Hessenberg.ipynb` | Walks through every computation of the project, step by step, with the outputs saved. |
| `bohemian/bohemian_analysis.py` | Generators for both models, the continuant recurrence, Bareiss determinant, exact rank, classification of singular matrices, exhaustive enumeration, the exact terms of the collision lower bound, and the exact second moments of the minors with the bounds that follow. |
| `bohemian/monte_carlo.py` | Monte Carlo singularity estimators for both models and the condition number. |
| `bohemian/gamma_estimation.py` | Rao-Blackwell estimator, the fit of γ(m) with bootstrap intervals, and the Lyapunov exponent. |
| `bohemian/gcd_resonance.py` | The gcd test, with its fixed decision rule: the gcd split of consecutive minors, the growth of the gcd effect between two fixed dimensions, and the decision rule. |
| `bohemian/bohemian_eigenvalues.py` | Eigenvalue density of Bohemian families, accumulated into a 2D histogram. |

The notebook runs in about 10 to 15 minutes with `FULL = False`; `FULL = True` uses 10^8 samples for the eigenvalue density. Requirements are in `requirements.txt` (Python 3.12).
