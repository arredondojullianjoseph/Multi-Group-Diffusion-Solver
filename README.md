# Multi-Group Diffusion Solver
Status: Work in Progress 🚧

Welcome! I'm building this multi-group neutron diffusion solver from scratch in Python, starting with the simplest case: a bare, homogeneous, one-group slab. That core solver and analytical verification are now in place.

**Next Up:** Generalizing to piecewise material properties

**Scope:** A self-study project to verify the neutron diffusion equation and experiment with numerical methods.

## Mathematical model

The script solves the one-group neutron diffusion equation for $k_{\text{eff}}$ and the flux $\phi(x)$:

$$ -D \frac{d^2 \phi(x)}{dx^2} + \Sigma_a \phi(x) = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi(x) $$

This is set for a bare slab of width $L$ (from $x = 0$ to $x = L$), with zero-flux boundary conditions at both edges: $\phi(0) = \phi(L) = 0$.

**Discretization:** I use finite differences on a uniform interior mesh with spacing $\Delta x = L / (N+1)$, placing nodes at $x_i = i\Delta x$ for $i = 1, \dots, N$. The two physical edges aren't unknowns, since $\phi$ is fixed at exactly zero there rather than solved for. Approximating the second derivative with a standard central difference and plugging it into the diffusion equation at node $i$ gives:

$$ \left( \frac{2D}{\Delta x^2} + \Sigma_a \right) \phi_i - \frac{D}{\Delta x^2} \phi_{i-1} - \frac{D}{\Delta x^2} \phi_{i+1} = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi_i $$

Writing this out for every interior node turns the continuous problem into a generalized matrix eigenvalue problem, $A\phi = \frac{1}{k_{\text{eff}}} F \phi$:

- **$A$ (loss operator):** a tridiagonal matrix capturing leakage and absorption, with diagonal entries $\frac{2D}{\Delta x^2} + \Sigma_a$ and off-diagonals $-\frac{D}{\Delta x^2}$.
- **$F$ (production operator):** conceptually a diagonal matrix holding $\nu\Sigma_f$ at each node, but since it's genuinely diagonal I keep it as a plain array and multiply elementwise instead of building an $N \times N$ matrix that's mostly zeros.

## Current implementation

- **Core solver:** sets up and solves the 1D reactor slab problem in Python, using `numpy` and `scipy`. Builds the tridiagonal matrix with `scipy.sparse.diags` and factors it once upfront with `splu`. Because the matrix is mostly zeros, factoring it once avoids repeated work that will matter once the mesh gets much finer. From there, the power iteration loop starts with a flat flux guess and $k^{(0)} = 1$. Each step estimates the fission source from the current flux, solves for the new flux through the pre-factored matrix, updates $k_{\text{eff}}$ using the ratio of the new and old source sums, and normalizes the flux by its maximum value.
- **Analytical verification:** tests the core solver against the exact solution for a bare slab. Refining the mesh from 25 to 800 nodes, $k_{\text{eff}}$ error drops by a factor of about 4 each time the mesh is halved, confirming the expected second-order accuracy of the central-difference stencil. The flux shape matches the exact sine solution to within $10^{-7}$ at every mesh size tested, independent of $N$: this discrete boundary condition makes the sine an exact eigenvector of the tridiagonal operator, so there's no discretization error left in the flux shape for this test case, only round-off.

## What's next?

- **Multi-region extension:** generalize to piecewise material properties ($D$, $\Sigma_a$, $\nu\Sigma_f$ varying across the slab), verified against a two-region analytical benchmark.
- **Multi-group extension:** generalize to multiple energy groups with inter-group scattering and a fission spectrum.
- **Testing:** add an automated test suite.
- **Documentation:** add flux and convergence plots, final polish pass.

## Usage

To run the core solver, make sure you have `numpy` and `scipy` installed, then run:

```bash
python Diffusion_1group.py
```

To run the analytical verification:

```bash
python analytical_verification.py
```

## Expected output

### Core solver

Running `python Diffusion_1group.py` as-is prints:

```
Converged in 79 iterations!
k_eff = 1.068368
```

That slab is slightly supercritical by construction ($\nu\Sigma_f$ a bit larger than $\Sigma_a$), so $k_{\text{eff}}$ a bit above 1 is the expected result, confirming the solver runs and converges cleanly.

### Analytical verification

Running `python analytical_verification.py` as-is prints:

#### Single run check (N = 200)

| Quantity | Value |
|---|---|
| Iterations to converge | 79 |
| Numerical k_eff | 1.068368 |
| Analytical k_eff | 1.068367 |
| Relative k_eff error | 5.834e-07 |
| Max flux error | 4.049e-08 |

#### Mesh convergence study

| Mesh points (N) | Grid spacing (Δx) | k_eff error (relative) | Observed order |
|---|---|---|---|
| 25 | 3.8462 | 3.497e-05 | N/A |
| 50 | 1.9608 | 9.090e-06 | 2.000 |
| 100 | 0.9901 | 2.317e-06 | 2.001 |
| 200 | 0.4975 | 5.834e-07 | 2.004 |
| 400 | 0.2494 | 1.451e-07 | 2.015 |
| 800 | 0.1248 | 3.485e-08 | 2.061 |

Observed order climbs slightly above 2 at the finest meshes, which is round-off in the error itself becoming comparable to the truncation error being measured, not a change in the discretization's accuracy.

The flux shape matches the exact sine solution to machine precision at every mesh size in this sweep (max error stays flat near $4\times10^{-8}$ regardless of $N$), for the reason given above: it isn't a useful signal for checking convergence order here, which is why this table tracks $k_{\text{eff}}$ error instead.

## References

- J. J. Duderstadt, L. J. Hamilton, *Nuclear Reactor Analysis*, John Wiley & Sons, 1976.
  - Chapter 5 — One-speed neutron diffusion equation, boundary conditions, and reactor criticality calculations for a bare slab.
  - Chapter 7 — Finite difference spatial discretization, matrix eigenvalue formulation $A\phi = \frac{1}{k_{\text{eff}}} F \phi$, and the power iteration (source iteration) algorithm implemented in the solver.
