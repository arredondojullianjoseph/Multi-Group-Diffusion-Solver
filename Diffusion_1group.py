"""
Diffusion_1group.py

This script models neutron behavior across a 1D reactor slab. It solves the 
one-group diffusion equation using power iteration to find the effective multiplication factor (k_eff) and the resulting neutron 
flux shape.
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


def make_matrices(D, Sigma_a, nu_Sigma_f, L, N):
    """
    Builds the finite-difference operator for the diffusion equation:
    -D * d^2(phi)/dx^2 + Sigma_a * phi = (1 / k) * nu_Sigma_f * phi

    and returns the tridiagonal loss operator(A), the fission source scalar (nu_Sigma_f), the distance between adjacent nodes(Delta_x) and the physical location of each internal node (x).
    """
    Delta_x = L / (N + 1)
    x = np.linspace(Delta_x, L - Delta_x, N)

    # Central-difference stencil approximation for the diffusion and absorption terms
    A_ii = 2.0 * D / Delta_x**2 + Sigma_a
    A_ij = -D / Delta_x**2
    A = diags(
        [np.full(N - 1, A_ij), np.full(N, A_ii), np.full(N - 1, A_ij)],
        offsets=[-1, 0, 1],
        format="csc",
    )
    return A, nu_Sigma_f, Delta_x, x


def find_k_eff(A, nu_Sigma_f, tol=1e-8, max_iter=5000):
    """
    Runs power iteration to find the fundamental eigenvalue (k_eff) and the 
    corresponding neutron flux shape. 

    and returns the calculated effective multiplication factor(k_eff), the normalized neutron flux shape (phi_new), and the total number of iterations it took to converge(i).
    """
    N = A.shape[0]
    phi = np.ones(N)            # A flat initial guess works great for a homogeneous slab
    k = 1.0
    lu = splu(A)                # Factor 

    for i in range(1, max_iter + 1):
        S_old = nu_Sigma_f * phi               # Estimate fission source from the current flux
        phi_new = lu.solve(S_old / k)          # Solve A @ phi_new = S_old / k
        S_new = nu_Sigma_f * phi_new

        k_new = k * np.sum(S_new) / np.sum(S_old)
        phi_new /= np.max(np.abs(phi_new))     # Normalize to keep just the shape

        # Both the eigenvalue and the flux shape need to settle down before we call it converged
        if abs(k_new - k) < tol and np.max(np.abs(phi_new - phi)) < tol:
            return k_new, phi_new, i

        phi, k = phi_new, k_new

    raise RuntimeError(f"Power iteration failed to converge within the {max_iter} step limit.")


if __name__ == "__main__":
    # Tests a slab that's slightly supercritical to make sure the solver runs nicely.
    D = 1.5
    Sigma_a = 0.05
    nu_Sigma_f = 0.055
    L = 100.0
    N = 200

    A, nu_Sigma_f, Delta_x, x = make_matrices(D, Sigma_a, nu_Sigma_f, L, N)
    k_eff, phi, n_iter = find_k_eff(A, nu_Sigma_f)

    print(f"Converged in {n_iter} iterations!")
    print(f"k_eff = {k_eff:.6f}")
