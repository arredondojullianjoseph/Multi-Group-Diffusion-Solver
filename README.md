# Multi-Group Diffusion Solver

Status: Work in Progress 🚧

Welcome! I'm building this multi-group neutron diffusion solver from scratch in Python, starting with the simplest case: a bare, homogeneous, one-group slab. The mathematical background for this project is in place. 

**Scope:** A self-study project to verify the neutron diffusion equation and experiment with numerical methods.

## Mathematical Model

The script solves the one-group neutron diffusion equation for $k_{\text{eff}}$ and the flux $\phi(x)$:

$$ -D \frac{d^2 \phi(x)}{dx^2} + \Sigma_a \phi(x) = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi(x) $$

This is set for a bare slab of width $L$ (from $x = 0$ to $x = L$), with zero-flux boundary conditions at both edges: $\phi(0) = \phi(L) = 0$.

**Discretization:** I'm using finite differences on a uniform interior mesh with spacing $\Delta x = L / (N+1)$, placing nodes at $x_i = i\Delta x$ for $i = 1, \dots, N$. Approximating the second derivative with a standard central difference and plugging it into the diffusion equation at node $i$ gives:

$$ \left( \frac{2D}{\Delta x^2} + \Sigma_a \right) \phi_i - \frac{D}{\Delta x^2} \phi_{i-1} - \frac{D}{\Delta x^2} \phi_{i+1} = \frac{1}{k_{\text{eff}}} \nu \Sigma_f \phi_i $$

Writing this out for every interior node turns the continuous problem into a generalized matrix eigenvalue problem, $A\phi = \frac{1}{k_{\text{eff}}} F \phi$:

* **$A$ (Loss Operator):** A tridiagonal matrix capturing leakage and absorption, with diagonal entries $\frac{2D}{\Delta x^2} + \Sigma_a$ and off-diagonals $-\frac{D}{\Delta x^2}$.
* **$F$ (Production Operator):** A diagonal matrix holding $\nu\Sigma_f$ at each node.

**Solver Strategy:** Rather than inverting $A$ outright every step, the script will use a single LU factorization of $A$ upfront. Starting from a flat flux guess and $k^{(0)} = 1$, each power iteration will:
1. Form the fission source.
2. Solve for the new flux.
3. Update $k_{\text{eff}}$ using the ratio of successive source sums.
4. Renormalize the flux by its maximum magnitude.

I'll check both $k_{\text{eff}}$ and the overall flux shape for convergence (with tolerance set to $10^{-8}$), since $k_{\text{eff}}$ can sometimes look settled a few iterations before the actual flux shape reaches its fundamental mode.

## What's Next?

* **Write the solver script:** Implement the core algorithm for the single-group case.
* **Analytical Verification:** Compare the computed flux and $k_{\text{eff}}$ against the known analytical solution.
* **Multi-Group Extension:** Generalize the script to handle multiple energy groups.
* **Documentation:** Add usage instructions, requirements, and expected outputs once everything is pushed.
