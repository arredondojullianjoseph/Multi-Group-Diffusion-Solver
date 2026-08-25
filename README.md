# Multi-Group Diffusion Solver

Status: Work in Progress 🚧

Welcome! I'm building this multi-group neutron diffusion solver from scratch in Python, starting with the simplest case: a bare, homogeneous, one-group slab. That single-group solver is now in place. 

**Next Up:** Checking the script against the known analytical solution before moving on to multiple regions and multiple groups.

**Scope:** A self-study project to verify the neutron diffusion equation and experiment with numerical methods.

## Mathematical Model

The script solves the one-group neutron diffusion equation for $k_{\text{eff}}$ and the flux $\phi(x)$:

$$ -D \frac{d^2 \phi(x)}{dx^2} + \Sigma_a \phi(x) = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi(x) $$

This is set for a bare slab of width $L$ (from $x = 0$ to $x = L$), with zero-flux boundary conditions at both edges: $\phi(0) = \phi(L) = 0$.

**Discretization:** I'm using finite differences on a uniform interior mesh with spacing $\Delta x = L / (N+1)$, placing nodes at $x_i = i\Delta x$ for $i = 1, \dots, N$ the two physical edges themselves aren't unknowns, since $\phi$ is fixed at exactly zero there rather than solved for. Approximating the second derivative with a standard central difference and plugging it into the diffusion equation at node $i$ gives:

$$ \left( \frac{2D}{\Delta x^2} + \Sigma_a \right) \phi_i - \frac{D}{\Delta x^2} \phi_{i-1} - \frac{D}{\Delta x^2} \phi_{i+1} = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi_i $$

Writing this out for every interior node turns the continuous problem into a generalized matrix eigenvalue problem, $A\phi = \frac{1}{k_{\text{eff}}} F \phi$:

- **$A$ (Loss Operator):** A tridiagonal matrix capturing leakage and absorption, with diagonal entries $\frac{2D}{\Delta x^2} + \Sigma_a$ and off-diagonals $-\frac{D}{\Delta x^2}$.
- **$F$ (Production Operator):** Conceptually a diagonal matrix holding $\nu\Sigma_f$ at each node, but since it's genuinely diagonal I keep it as a plain array and multiply elementwise instead of building an $N \times N$ matrix that's mostly zeros.

## Current Implementation

- **Core Solver:** Sets up and solves our 1D reactor slab problem in in Python, using `numpy` and `scipy`. Builds the tridiagonal matrix with `scipy.sparse.diags` and factoring it just once upfront using `splu`. Because the matrix is mostly zeros, factoring it once avoids heavy lifting(this will matter once we start working with much finer meshes). From there, we kick off our power iteration loop starting with a flat guess for the flux and $k^{(0)} = 1$. In each iteration, we estimate the fission source using our current flux shape, solve for the new flux by running it through our pre-factored matrix, update $k_{\text{eff}}$ using the ratio of the new and old source sums, and finally normalize the flux by its maximum value to keep everything clean and stable for the next round.

## Usage

make sure you have `numpy` and `scipy` installed then run:

```bash
python Diffusion_1group.py
```

## Expected Output

Running `python diffusion_1group.py` as-is prints:

```
Converged in 79 iterations!
k_eff = 1.068368
```

That slab is slightly supercritical by construction ($\nu\Sigma_f$ a bit larger than $\Sigma_a$), so $k_{\text{eff}}$ a bit above 1 is the expected shape of the answer confirmed to run and converge cleanly.

## What's Next?

- **Analytical Verification:** Compare the computed flux and $k_{\text{eff}}$ against the known analytical solution.
- **Multi-Group Extension:** Generalize the script to handle multiple energy groups.
- **Documentation:** Expand usage instructions and expected outputs as new pieces (validation, multi-region, multi-group) get added.

## References

- J. J. Duderstadt, L. J. Hamilton, *Nuclear Reactor Analysis*, John Wiley & Sons, 1976.
  - Chapter 5 - One-speed neutron diffusion equation, boundary conditions, and reactor criticality calculations for a bare slab.
  - Chapter 7 - Finite difference spatial discretization, matrix eigenvalue formulation $A\phi = \frac{1}{k_{\text{eff}}} F \phi$, and the power iteration (source iteration) algorithm implemented in the solver.
