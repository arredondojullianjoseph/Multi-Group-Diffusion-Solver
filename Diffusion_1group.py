import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

def make_matrices(D, Sigma_a, nu_Sigma_f, L, N):
    """
    Builds the finite-difference operator for the 1D neutron diffusion equation:
    -d/dx(D dphi/dx) + Sigma_a * phi = (1/k) * nu_Sigma_f * phi
    Returns the loss operator A, fission source array, grid spacing, and node positions.
    """
    Delta_x = L / (N + 1)
    x = np.linspace(Delta_x, L - Delta_x, N)

    D = np.full(N, D, dtype=float) if np.isscalar(D) else np.asarray(D, dtype=float)
    Sigma_a = np.full(N, Sigma_a, dtype=float) if np.isscalar(Sigma_a) else np.asarray(Sigma_a, dtype=float)
    nu_Sigma_f = np.full(N, nu_Sigma_f, dtype=float) if np.isscalar(nu_Sigma_f) else np.asarray(nu_Sigma_f, dtype=float)

    # D_face handles boundaries and internal interfaces.
    # Edges just use the neighboring node's D value since it's a vacuum boundary outside.
    D_face = np.empty(N + 1)
    D_face[0] = D[0]
    D_face[N] = D[-1]
    D_face[1:N] = 2.0 * D[:-1] * D[1:] / (D[:-1] + D[1:])

    A_ii = (D_face[:-1] + D_face[1:]) / Delta_x**2 + Sigma_a
    A_off = -D_face[1:N] / Delta_x**2    # Symmetric coupling between adjacent nodes

    A = diags(
        [A_off, A_ii, A_off],
        offsets=[-1, 0, 1],
        format="csc",
    )
    return A, nu_Sigma_f, Delta_x, x

def build_region_arrays(regions, L, N):
    """Maps out material regions across our N nodes.
    Each region is given as (start, end, D, Sigma_a, nu_Sigma_f).
    """
    Delta_x = L / (N + 1)
    x = np.linspace(Delta_x, L - Delta_x, N)
    D = np.empty(N)
    Sigma_a = np.empty(N)
    nu_Sigma_f = np.empty(N)
    for start, end, D_r, Sigma_a_r, nu_Sigma_f_r in regions:
        mask = (x >= start) & (x < end)
        D[mask] = D_r
        Sigma_a[mask] = Sigma_a_r
        nu_Sigma_f[mask] = nu_Sigma_f_r
    return D, Sigma_a, nu_Sigma_f

def find_k_eff(A, nu_Sigma_f, tol=1e-8, max_iter=5000):
    """Uses power iteration to find the fundamental eigenvalue (k_eff)
    and the resulting neutron flux shape. Returns k_eff, the normalized
    flux shape, and the total iteration count.
    """
    N = A.shape[0]
    phi = np.ones(N)             # A flat initial guess works great for a homogeneous slab
    k = 1.0
    lu = splu(A)                 # LU decomposition for efficient solves   
    for i in range(1, max_iter + 1):
        S_old = nu_Sigma_f * phi             # Estimate fission source from current flux
        phi_new = lu.solve(S_old / k)        # Solve the linear system
        S_new = nu_Sigma_f * phi_new
        k_new = k * np.sum(S_new) / np.sum(S_old)
        phi_new /= np.max(np.abs(phi_new))   # Normalize the flux shape
        
        # Check if both eigenvalue and flux have converged
        if abs(k_new - k) < tol and np.max(np.abs(phi_new - phi)) < tol:
            return k_new, phi_new, i
        phi, k = phi_new, k_new
    raise RuntimeError(f"Power iteration failed to converge within the {max_iter} step limit.")

if __name__ == "__main__":
    
    # Test a slightly supercritical slab to make sure everything runs smoothly
    D = 1.5
    Sigma_a = 0.05
    nu_Sigma_f = 0.055
    L = 100.0
    N = 200
    A, nu_Sigma_f, Delta_x, x = make_matrices(D, Sigma_a, nu_Sigma_f, L, N)
    k_eff, phi, n_iter = find_k_eff(A, nu_Sigma_f)
    print(f"Converged in {n_iter} iterations!")
    print(f"k_eff = {k_eff:.6f}")
