"""
=============================================================================
PHASE-DEPENDENT FLOQUET GAP IN GAPPED GRAPHENE
Using the same bicircular drive parameters as the chirality dynamics work.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qutip import sigmax, sigmay, sigmaz, qeye

# ── Physical constants (same as your chirality work) ──────────────────────
HBAR   = 0.6582119569   # eV·fs
t_hop  = 2.71           # eV
a_ang  = 2.46           # Å
v_F    = (np.sqrt(3)/2) * t_hop * a_ang    # eV·Å

SX = sigmax()
SY = sigmay()
SZ = sigmaz()
I2 = qeye(2)

# ── Parameters (same as your chirality work) ─────────────────────────────
mass_gap = 0.05          # eV
A1       = 0.15          # Å⁻¹
A2       = 0.08          # Å⁻¹
omega    = 0.10 / HBAR   # rad/fs (resonant with 2Δ)
helicity = +1
T_period = 2 * np.pi / omega   # One period of the fundamental frequency

# ── Helper: Build time-dependent Hamiltonian ──────────────────────────────
def H_dirac(kx, ky, t, phi, mass=mass_gap):
    """Full time-dependent Dirac Hamiltonian at time t."""
    Ax = A1 * np.cos(omega * t) + A2 * np.cos(2 * omega * t + phi)
    Ay = A1 * np.sin(omega * t) - A2 * np.sin(2 * omega * t + phi)
    return v_F * ((kx + Ax) * SX + (ky + Ay) * SY) + mass * SZ

# ── Floquet gap calculation for one valley ──────────────────────────────
def floquet_gap(kx, ky, phi, n_steps=400):
    """
    Compute the Floquet quasi-energy gap at (kx, ky) for a given phase phi.
    
    Method:
    1. Discretize the period [0, T] into n_steps.
    2. Build the time-evolution operator U(T, 0) as a product of
       small-time propagators: U = U_n ... U_2 U_1.
    3. Diagonalize U(T, 0) to obtain the quasi-energies ε_n via
       U |u_n> = exp(-i ε_n T / ħ) |u_n>.
    4. The Floquet gap is the minimum difference between quasi-energies
       in the Floquet zone [-πħ/T, πħ/T].
    """
    dt = T_period / n_steps
    U = np.eye(2, dtype=complex)  # 2x2 identity
    
    for i in range(n_steps):
        t = i * dt
        # Small-time propagator: exp(-i H dt / ħ)
        H_t = H_dirac(kx, ky, t, phi).full()
        U_step = expm(-1j * H_t * dt / HBAR)
        U = U_step @ U
    
    # Diagonalize U to get quasi-energies
    evals = np.linalg.eigvals(U)
    # Quasi-energies: ε = (i ħ / T) * log(evals)
    quasi_energies = (1j * HBAR / T_period) * np.log(evals)
    quasi_energies = np.real(quasi_energies)  # should be real for unitary U
    
    # Map to Floquet zone [-πħ/T, πħ/T]
    quasi_energies = np.mod(quasi_energies + np.pi * HBAR / T_period, 
                            2 * np.pi * HBAR / T_period) - np.pi * HBAR / T_period
    quasi_energies = np.sort(quasi_energies)
    
    # Gap = difference between the two quasi-energies
    gap = np.abs(quasi_energies[1] - quasi_energies[0])
    return gap

# ── Scan over phase φ for both valleys ──────────────────────────────────
phi_scan = np.linspace(0, 2 * np.pi, 25)
kx_K = 0.02
ky = 0.02
kx_Kp = -0.02

gap_K = []
gap_Kp = []

print("Computing Floquet gaps for each phase...")
for phi in phi_scan:
    print(f"  φ = {phi/np.pi:.3f} π")
    gap_K.append(floquet_gap(kx_K, ky, phi))
    gap_Kp.append(floquet_gap(kx_Kp, ky, phi))

print("Done.")

# ── Plot ──
plt.figure(figsize=(8, 5))
plt.plot(phi_scan/np.pi, gap_K, 'o-', color='#E8593C', lw=2, ms=6, label='K-valley')
plt.plot(phi_scan/np.pi, gap_Kp, 's-', color='#1D9E75', lw=2, ms=6, label="K'-valley")
plt.xlabel(r"Phase $\phi / \pi$", fontsize=12)
plt.ylabel(r"Floquet Gap (eV)", fontsize=12)
plt.title("Phase-Dependent Floquet Gap in Gapped Graphene", fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("floquet_gap_vs_phase.png", dpi=200)
plt.show()

print("✅ Figure saved as 'floquet_gap_vs_phase.png'")