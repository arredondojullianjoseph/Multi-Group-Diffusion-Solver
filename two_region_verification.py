"""
two_region_verification.py

Verifies our multi-region neutron diffusion solver against the exact 
analytical criticality condition for a two-region 
bare slab with vacuum boundary conditions 
"""

import numpy as np
from scipy.optimize import brentq
from Diffusion_1group import make_matrices, build_region_arrays, find_k_eff


def residual(k, D1, Sigma_a1, nu_Sigma_f1, D2, Sigma_a2, a, L):
    """
    Evaluates the two-region criticality equation. Roots of this function 
    correspond to valid k_eff values.
    """
    B1_sq = (nu_Sigma_f1 / k - Sigma_a1) / D1
    if B1_sq <= 0:
        return np.nan
    B1 = np.sqrt(B1_sq)
    kappa2 = np.sqrt(Sigma_a2 / D2)
    
    # Matches flux and current at the core-reflector interface 
    return D1 * B1 / np.tan(B1 * a) - (-D2 * kappa2 / np.tanh(kappa2 * (L - a)))


def exact_k_two_region(D1, Sigma_a1, nu_Sigma_f1, D2, Sigma_a2, a, L, bracket):
    """
    Root-finds the exact k_eff 
    """
    return brentq(lambda k: residual(k, D1, Sigma_a1, nu_Sigma_f1, D2, Sigma_a2, a, L), *bracket)


if __name__ == "__main__":
    """
    Finds the exact k_eff once, then compares it against the numerical solver at five
    mesh sizes to check the observed order of convergence.
    """
    D1, Sigma_a1, nu_Sigma_f1 = 1.5, 0.05, 0.060
    D2, Sigma_a2 = 1.0, 0.01
    a, L = 50.0, 70.0
    k_exact = exact_k_two_region(D1, Sigma_a1, nu_Sigma_f1, D2, Sigma_a2, a, L, bracket=(1.073, 1.1999))
    print(f"{'N':>5} | {'Delta_x':>9} | {'k_eff error':>12} | {'Order':>6}")
    print("-" * 45)
    prev_err, prev_dx = None, None
    for N in [70, 140, 280, 560, 1120]:
        D, Sigma_a, nu_Sigma_f = build_region_arrays(
            regions=[(0.0, a, D1, Sigma_a1, nu_Sigma_f1), (a, L, D2, Sigma_a2, 0.0)],
            L=L, N=N,
        )
        A, nu_Sigma_f_arr, Delta_x, x = make_matrices(D, Sigma_a, nu_Sigma_f, L, N)
        k_num, phi_num, n_iter = find_k_eff(A, nu_Sigma_f_arr)
        
        # Compute relative error against the analytical solution
        k_err = abs(k_num - k_exact) / k_exact
        if prev_err is not None:
            order = np.log(prev_err / k_err) / np.log(prev_dx / Delta_x)
            order_str = f"{order:.3f}"
        else:
            order_str = "N/A"
        print(f"{N:>5} | {Delta_x:>9.5f} | {k_err:>12.4e} | {order_str:>6}")
        prev_err, prev_dx = k_err, Delta_x
