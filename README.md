# Multi-Group Diffusion Solver
Status: Work in Progress 🚧

Welcome! I'm building this multi-group neutron diffusion solver from scratch in Python, starting with the simplest case: a bare, homogeneous, one-group slab. That core solver and analytical verification are now in place as well as generalization of the core solver to handle multiple material regions.

**Next Up:** Verifying the multi-region solver against a two-region analytical benchmark

**Scope:** A self-study project to verify the neutron diffusion equation and experiment with numerical methods.

## Mathematical model

The script solves the one-group neutron diffusion equation for $k_{\text{eff}}$ and the flux $\phi(x)$:

$$ -D \frac{d^2 \phi(x)}{dx^2} + \Sigma_a \phi(x) = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi(x) $$

This equation states that in a steady state, the rate of neutron loss (left side: leakage + absorption) equals the rate of neutron production from fission (right side), scaled by $1/k$.

The script's default is for a bare slab of width $L$ with zero-flux boundary conditions at both edges.

**Discretization:** the script uses a finite differences on a uniform interior mesh with spacing $\Delta x = L / (N+1)$, and put's nodes at $x_i = i\Delta x$ for $i = 1, \dots, N$. The two physical edges aren't unknowns, since $\phi$ is fixed at exactly zero there rather than solved for. Approximating the second derivative and then plugging it into the diffusion equation at node $i$ gives:

$$ \left( \frac{2D}{\Delta x^2} + \Sigma_a \right) \phi_i - \frac{D}{\Delta x^2} \phi_{i-1} - \frac{D}{\Delta x^2} \phi_{i+1} = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi_i $$

Writing this out for every interior node turns the continuous problem into a generalized matrix eigenvalue problem, $A\phi = \frac{1}{k_{\text{eff}}} F \phi$:

- **$A$ (loss operator):** a tridiagonal matrix capturing leakage and absorption, with diagonal entries $\frac{2D}{\Delta x^2} + \Sigma_a$ and off-diagonals $-\frac{D}{\Delta x^2}$.
- **$F$ (production operator):** is like a diagonal matrix holding $\nu\Sigma_f$ at each node, but it's multiplied element wise instead of building an $N \times N$ matrix that's mostly zeros.

At a material interface, $D$ is discontinuous, but the neutron current, $D \, d\phi/dx$, is not. Arithmetic averaging of $D$ across the interface gets this wrong, so the script uses the harmonic mean instead.

## Current implementation

- **Core solver:** sets up and solves the 1D reactor slab problem in Python, using `numpy` and `scipy`. Builds the tridiagonal matrix with `scipy.sparse.diags` and factors it once upfront with `splu`. Because the matrix is mostly zeros, factoring it once avoids repeated work that will matter once the mesh gets much finer. From there, the power iteration loop starts with a flat flux guess and $k^{(0)} = 1$. Each step estimates the fission source from the current flux, solves for the new flux through the pre-factored matrix, updates $k_{\text{eff}}$ using the ratio of the new and old source sums, and normalizes the flux by its maximum value.
- **Analytical verification:** tests the core solver against the exact solution for a bare slab. Refining the mesh from 25 to 800 nodes, $k_{\text{eff}}$ error drops by a factor of about 4 each time the mesh is halved, confirming the expected second-order accuracy of the central-difference stencil. The flux shape matches the exact sine solution to within $10^{-7}$ at every mesh size tested, independent of $N$: this discrete boundary condition makes the sine an exact eigenvector of the tridiagonal operator, so there's no discretization error left in the flux shape for this test case, only round-off.
- **Multi-region extension:** `make_matrices` is generalized to take $D$, $\Sigma_a$, and $\nu\Sigma_f$ as either scalars or per-node arrays. Same function now handles the homogeneous slab and a slab built from multiple materials. `build_region_arrays` maps a list of `(start, end, D, Sigma_a, nu_Sigma_f)` regions onto the mesh. A two-region test case (50 cm fuel, 20 cm reflector, $N = 280$) converges to $k_{\text{eff}} = 0.951846$ in 70 iterations.

## What's next?

- **Multi-region verification:** compare against the analytical criticality condition for a two-region (fuel + reflector) bare slab.
- **Multi-group extension:** generalize to multiple energy groups with inter-group scattering and a fission spectrum.
- **Multi-group verification:** compare against a known two-group benchmark.
- **Testing:** add an automated test suite.
- **Documentation:** add flux and convergence plots, final polish pass.

## Usage

To run the core solver, make sure you have `numpy` and `scipy` installed, then run:

```bash
python Diffusion_1group.py
