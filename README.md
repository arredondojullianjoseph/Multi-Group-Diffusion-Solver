# Multi-Group Diffusion Solver
**Status:** On Hold ⏸️ 
 
Welcome! I'm building this multi-group neutron diffusion solver from scratch in Python, starting with the simplest case: a bare, homogeneous, one-group slab. That core solver and analytical verification are now in place as well as generalizations to handle multiple material regions, and verification of that against an analytical two-region benchmark.
 
**Next Up:** Adding a flux comparison plot for the two-region verification
 
**Scope:** A self-study project to verify the neutron diffusion equation and experiment with numerical methods.
 
## Mathematical model
 
The script solves the one-group neutron diffusion equation for $k_{\text{eff}}$ and the flux $\phi(x)$:
 
$$ -D \frac{d^2 \phi(x)}{dx^2} + \Sigma_a \phi(x) = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi(x) $$
 
This equation states that in a steady state, the rate of neutron loss (left side: leakage + absorption) equals the rate of neutron production from fission (right side), scaled by $1/k$.
 
The script's default is for a bare slab of width $L$ with zero-flux boundary conditions at both edges.
 
**Discretization:** the script uses finite differences on a uniform interior mesh with spacing $\Delta x = L / (N+1)$, and puts nodes at $x_i = i\Delta x$ for $i = 1, \dots, N$. The two physical edges aren't unknowns, since $\phi$ is fixed at exactly zero there rather than solved for. Approximating the second derivative and then plugging it into the diffusion equation at node $i$ gives:
 
$$ \left( \frac{2D}{\Delta x^2} + \Sigma_a \right) \phi_i - \frac{D}{\Delta x^2} \phi_{i-1} - \frac{D}{\Delta x^2} \phi_{i+1} = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi_i $$
 
Writing this out for every interior node turns the continuous problem into a generalized matrix eigenvalue problem, $A\phi = \frac{1}{k_{\text{eff}}} F \phi$:
 
- **$A$ (loss operator):** a tridiagonal matrix capturing leakage and absorption, with diagonal entries $\frac{2D}{\Delta x^2} + \Sigma_a$ and off-diagonals $-\frac{D}{\Delta x^2}$.
- **$F$ (production operator):** is like a diagonal matrix holding $\nu\Sigma_f$ at each node, but it's multiplied elementwise instead of building an $N \times N$ matrix that's mostly zeros.
At a material interface, $D$ is discontinuous, but the neutron current, $D \, d\phi/dx$, is not. Arithmetic averaging of $D$ across the interface gets this wrong, so the script uses the harmonic mean instead.
 
## Current implementation
 
- **Core solver:** sets up and solves the 1D reactor slab problem in Python, using `numpy` and `scipy`. Builds the tridiagonal matrix with `scipy.sparse.diags` and factors it once upfront with `splu`. Because the matrix is mostly zeros, factoring it once avoids repeated work that will matter once the mesh gets much finer. From there, the power iteration loop starts with a flat flux guess and $k^{(0)} = 1$. Each step estimates the fission source from the current flux, solves for the new flux through the pre-factored matrix, updates $k_{\text{eff}}$ using the ratio of the new and old source sums, and normalizes the flux by its maximum value. `Diffusion_1group.py` runs both a homogeneous sanity check and a quick two-region check when run directly.
- **Analytical verification:** tests the core solver against the exact solution for a bare slab. Refining the mesh from 25 to 800 nodes, $k_{\text{eff}}$ error drops by a factor of about 4 each time the mesh is halved, confirming the expected second-order accuracy of the central-difference stencil. The flux shape stays flat around 4e-08 regardless of mesh size, since the sine is an exact eigenvector of the discrete matrix — there's no spatial discretization error to shrink there, so that error floor comes from the power iteration's own stopping tolerance (1e-8), not round-off.
- **Multi-region extension:** `make_matrices` is generalized to take $D$, $\Sigma_a$, and $\nu\Sigma_f$ as either scalars or per-node arrays. Same function now handles the homogeneous slab and a slab built from multiple materials. `build_region_arrays` maps a list of `(start, end, D, Sigma_a, nu_Sigma_f)` regions onto the mesh.
- **Multi-region verification:** Runs the mesh convergence sweep for the two-region benchmark. Compares the
numerical solver against the exact two-region criticality solution as the
mesh refines. Error shrinks first-order, not second-order like the homogeneous case. The fuel-reflector interface rarely lands exactly on a mesh face, and that mismatch is what caps the convergence rate.
  
## What's next?
 
- **Visualization:** add a flux comparison plot (numerical vs. analytical) to the two-region verification.
- **Multi-group extension:** generalize to multiple energy groups with inter-group scattering and a fission spectrum.
- **Multi-group verification:** compare against a known two-group benchmark.
- **Testing:** add an automated test suite.
- **Documentation:** final polish pass.
  
## Usage
 
To run the core solver, make sure you have `numpy` and `scipy` installed, then run:
 
```bash
python Diffusion_1group.py
```
 
To run the analytical verification:
 
```bash
python analytical_verification.py
```
 
To run the two-region verification:
 
```bash
python two_region_verification.py
```
 
## Expected output
 
### Core solver
 
Running `python Diffusion_1group.py` as-is prints:
 
```
Homogeneous: converged in 79 iterations, k_eff = 1.068368
Two‑region : converged in 48 iterations, k_eff = 1.108446
```
 
The homogeneous case is slightly supercritical by construction ($\nu\Sigma_f$ a bit larger than $\Sigma_a$), so $k_{\text{eff}}$ a bit above 1 there is expected. The two-region line is just a quick sanity check that `build_region_arrays` and `make_matrices` run together without error — it's a different material setup than the actual verification benchmark below, not the same test.
 
## Analytical Verification
 
Running `python analytical_verification.py` runs two checks: a single run at N = 200 and a mesh sweep from N = 25 to N = 800.
 
### Single run check (N = 200)
| Metric | Value |
| :--- | :--- |
| **Iterations to Converge** | 79 |
| **$k_{\text{eff}}$ (Numerical)** | 1.068368 |
| **$k_{\text{eff}}$ (Exact)** | 1.068367 |
| **$k_{\text{eff}}$ Error** | $5.834 \times 10^{-7}$ |
| **Flux Max Error** | $4.049 \times 10^{-8}$ |

The solver finds the exact eigenvalue to within 6e-7 and the flux shape to within 4e-8.
 
### Mesh convergence test
 
The script sweeps the mesh from N = 25 to N = 800 and prints the flux error at each resolution:
 
| N | Delta_x | Flux Error | Order |
|---|---|---|---|
| 25 | 3.8462 | 3.8937e-08 | N/A |
| 50 | 1.9608 | 4.2155e-08 | -0.118 |
| 100 | 0.9901 | 4.0819e-08 | 0.047 |
| 200 | 0.4975 | 4.0486e-08 | 0.012 |
| 400 | 0.2494 | 4.0405e-08 | 0.003 |
| 800 | 0.1248 | 4.0385e-08 | 0.001 |
 
The flux error stays flat at around 4e-08 across all mesh sizes. This is expected: because the sine is an exact eigenvector of the discrete matrix, there is no spatial discretization error for the flux shape. The remaining error is dominated by the power iteration's stopping tolerance (1e-8), not by round-off.
 
The convergence test that actually tells us something is the **k_eff error**:
 
| N | Delta_x | k_eff Error | Order |
|---|---|---|---|
| 25 | 3.8462 | 3.4970e-05 | N/A |
| 50 | 1.9608 | 9.0903e-06 | 2.000 |
| 100 | 0.9901 | 2.3165e-06 | 2.001 |
| 200 | 0.4975 | 5.8341e-07 | 2.004 |
| 400 | 0.2494 | 1.4508e-07 | 2.015 |
| 800 | 0.1248 | 3.4852e-08 | 2.061 |
 
As the mesh refines, the observed order settles at 2. The slight overshoot at the finest mesh is round-off in the error measurement, not a change in how the discretization behaves. This confirms the central-difference stencil is second-order accurate for the eigenvalue.
 
## Two-Region Verification
 
The homogeneous slab has a known analytical solution, but a two-region slab doesn't, so I had to work out the correct answer myself before I could check the numerical solver against anything. Fuel occupies $[0, a]$ and satisfies $\phi(0) = 0$, so $\phi_1(x) = \sin(B_1 x)$. The reflector occupies $[a, L]$ and satisfies $\phi(L) = 0$, so $\phi_2(x) = \sinh(\kappa_2 (L - x))$. Matching flux and current at $x = a$ gives the criticality condition:
 
$$ D_1 B_1 \cot(B_1 a) = -D_2 \kappa_2 \coth(\kappa_2 (L - a)) $$
 
$k_{\text{eff}}$ is a root of this equation. It's periodic in $B_1 a$, so I bracket narrowly around the fundamental root ($B_1 a < \pi$) instead of scanning wide. A wide scan turns up poles that look like extra roots.
 
Running `python two_region_verification.py` on a 50 cm fuel / 20 cm reflector slab sweeps the mesh from N = 70 to N = 1120:
 
| N | Delta_x | k_eff error | Order |
|---:|---:|---:|---:|
| 70 | 0.98592 | 4.235e-04 | N/A |
| 140 | 0.49645 | 2.239e-04 | 0.929 |
| 280 | 0.24911 | 1.150e-04 | 0.966 |
| 560 | 0.12478 | 5.829e-05 | 0.983 |
| 1120 | 0.06244 | 2.934e-05 | 0.992 |
 
Order climbs toward 1, not 2 because the fuel-reflector interface typically doesn't land exactly on a mesh face.
## References
 
- J. J. Duderstadt, L. J. Hamilton, *Nuclear Reactor Analysis*, John Wiley & Sons, 1976.
  - Chapter 5 — One-speed neutron diffusion equation, boundary conditions, and reactor criticality calculations for a bare slab.
  - Chapter 7 — Finite difference spatial discretization, matrix eigenvalue formulation $A\phi = \frac{1}{k_{\text{eff}}} F \phi$, and the power iteration (source iteration) algorithm implemented in the solver.
