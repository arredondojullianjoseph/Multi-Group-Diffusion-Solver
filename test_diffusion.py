"""
test_diffusion.py

Automated tests for the diffusion solver. Turns the checks from
analytical_verification.py and two_region_verification.py into pass/fail
assertions.
"""
import numpy as np

from Diffusion_1group import make_matrices, build_region_arrays, find_k_eff
from analytical_verification import check_mesh
from two_region_verification import exact_k_two_region, exact_flux_two_region

# Bare homogeneous slab (same case as analytical_verification.py)
D, SIGMA_A, NU_SIGMA_F, L = 1.5, 0.05, 0.055, 100.0

# Two-region fuel/reflector slab (same case as two_region_verification.py)
D1, SIGMA_A1, NU_SIGMA_F1 = 1.5, 0.05, 0.060
D2, SIGMA_A2 = 1.0, 0.01
A_INTERFACE, L_TWO = 50.0, 70.0
BRACKET = (1.073, 1.1999)


def observed_orders(errors, dxs):
    return [
        np.log(errors[i - 1] / errors[i]) / np.log(dxs[i - 1] / dxs[i])
        for i in range(1, len(errors))
    ]


def solve_two_region(N):
    D_arr, Sigma_a_arr, nu_Sigma_f_arr = build_region_arrays(
        regions=[(0.0, A_INTERFACE, D1, SIGMA_A1, NU_SIGMA_F1), (A_INTERFACE, L_TWO, D2, SIGMA_A2, 0.0)],
        L=L_TWO, N=N,
    )
    A, nu_Sigma_f_arr, Delta_x, x = make_matrices(D_arr, Sigma_a_arr, nu_Sigma_f_arr, L_TWO, N)
    k_num, phi_num, _ = find_k_eff(A, nu_Sigma_f_arr)
    return k_num, phi_num, Delta_x, x


def test_bare_slab_matches_exact():
    _, _, k_error, flux_error, _, _ = check_mesh(D, SIGMA_A, NU_SIGMA_F, L, 200)
    assert k_error < 1e-6
    assert flux_error < 1e-6


def test_bare_slab_is_second_order():
    errors, dxs = [], []
    for N in [25, 50, 100, 200, 400, 800]:
        _, _, k_error, _, _, dx = check_mesh(D, SIGMA_A, NU_SIGMA_F, L, N)
        errors.append(k_error)
        dxs.append(dx)
    for order in observed_orders(errors, dxs):
        assert 1.9 < order < 2.2


def test_two_region_converges_first_order():
    k_exact = exact_k_two_region(D1, SIGMA_A1, NU_SIGMA_F1, D2, SIGMA_A2, A_INTERFACE, L_TWO, BRACKET)
    errors, dxs = [], []
    for N in [70, 140, 280, 560, 1120]:
        k_num, _, dx, _ = solve_two_region(N)
        errors.append(abs(k_num - k_exact) / k_exact)
        dxs.append(dx)
    assert errors[-1] < 5e-5
    for order in observed_orders(errors, dxs):
        assert 0.9 < order < 1.1


def test_two_region_flux_matches_exact():
    k_exact = exact_k_two_region(D1, SIGMA_A1, NU_SIGMA_F1, D2, SIGMA_A2, A_INTERFACE, L_TWO, BRACKET)
    _, phi_num, _, x = solve_two_region(280)
    phi_exact = exact_flux_two_region(x, k_exact, D1, SIGMA_A1, NU_SIGMA_F1, D2, SIGMA_A2, A_INTERFACE, L_TWO)
    assert np.max(np.abs(phi_num - phi_exact)) < 1e-2
