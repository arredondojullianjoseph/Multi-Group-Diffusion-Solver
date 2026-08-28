"""
analytical_verification.py

Checks how close the numerical k_eff and flux shape are to the analytical solution at a fixed mesh size. It also verifies that the error shrinks quadratically (O(Delta_x^2)) as the mesh is refined.
"""

import numpy as np
from Diffusion_1group import make_matrices, find_k_eff


def exact_k(D, Sigma_a, nu_Sigma_f, L):
    """
    Finds the exact critical eigenvalue (k_eff)
    k_eff = nu_Sigma_f / (Sigma_a + D * (pi / L)^2)
    and returns the analytical multiplication factor (k_exact).
    """
    buckling_sq = (np.pi / L) ** 2 # How much the neutrons are leaking out
    return nu_Sigma_f / (Sigma_a + D * buckling_sq)


def exact_flux(x, L):
    """
    Finds the exact fundamental flux shape for a bare, homogeneous 1D slab.
    phi(x) = sin(pi * x / L)
    and returns the normalized flux.
    """
    phi_exact = np.sin(np.pi * x / L)
    return phi_exact / np.max(np.abs(phi_exact))


def check_mesh(D, Sigma_a, nu_Sigma_f, L, N):
    """
    Runs the numerical solver for a given mesh size and compares it to the analytical solution.
    and returns the numerical k_eff, exact k_eff, relative k_eff error, maximum flux error, number of iterations, and the mesh spacing (Delta_x).
    """
    A, nu_Sigma_f_arr, Delta_x, x = make_matrices(D, Sigma_a, nu_Sigma_f, L, N)
    k_num, phi_num, n_iter = find_k_eff(A, nu_Sigma_f_arr)
    k_exact = exact_k(D, Sigma_a, nu_Sigma_f, L)
    phi_exact = exact_flux(x, L)
    k_error = abs(k_num - k_exact) / k_exact
    flux_error = np.max(np.abs(phi_num - phi_exact))
    return k_num, k_exact, k_error, flux_error, n_iter, Delta_x


if __name__ == "__main__":
    # Tests the solver on a standard slab and then runs a convergence study.
    D = 1.5
    Sigma_a = 0.05
    nu_Sigma_f = 0.055
    L = 100.0
    print("--- Single run check (N=200) ---")
    k_num, k_exact, k_error, flux_error, n_iter, Delta_x = check_mesh(D, Sigma_a, nu_Sigma_f, L, 200)
    print(f"Converged in {n_iter} iterations!")
    print(f"k_eff (numerical) = {k_num:.6f}")
    print(f"k_eff (exact)     = {k_exact:.6f}")
    print(f"k_eff error       = {k_error:.3e}")
    print(f"Flux max error    = {flux_error:.3e}\n")
    
    print("--- Mesh convergence test (flux error) ---")
    print(f"{'N':>5} | {'Delta_x':>9} | {'Flux Error':>12} | {'Order':>6}")
    print("-" * 45)
    mesh_sizes = [25, 50, 100, 200, 400, 800]
    prev_err = None
    prev_dx = None
    flux_errors = []
    dx_values = []
    
    for N in mesh_sizes:
        _, _, _, flux_err, _, dx = check_mesh(D, Sigma_a, nu_Sigma_f, L, N)
        flux_errors.append(flux_err)
        dx_values.append(dx)
        
        if prev_err is not None:
            order = np.log(prev_err / flux_err) / np.log(prev_dx / dx)
            order_str = f"{order:.3f}"
        else:
            order_str = "N/A"
        print(f"{N:>5} | {dx:>9.4f} | {flux_err:>12.4e} | {order_str:>6}")
        prev_err, prev_dx = flux_err, dx
    
    print("\n--- Mesh convergence test (k_eff error) ---")
    print(f"{'N':>5} | {'Delta_x':>9} | {'k_eff Error':>12} | {'Order':>6}")
    print("-" * 45)
    
    prev_err = None
    prev_dx = None
    
    for i, N in enumerate(mesh_sizes):
        _, _, k_err, _, _, dx = check_mesh(D, Sigma_a, nu_Sigma_f, L, N)
        
        if prev_err is not None:
            order = np.log(prev_err / k_err) / np.log(prev_dx / dx)
            order_str = f"{order:.3f}"
        else:
            order_str = "N/A"
        print(f"{N:>5} | {dx:>9.4f} | {k_err:>12.4e} | {order_str:>6}")
        prev_err, prev_dx = k_err, dx
