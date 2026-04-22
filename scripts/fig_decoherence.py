"""
Figure for Chapter 4: Lindblad simulations of decoherence channels.

Simulates:
  (a) Free induction decay (FID): transverse coherence <sigma_x>(t)
  (b) T1 inversion recovery: longitudinal <sigma_z>(t)
  (c) Hahn spin-echo amplitude vs 2 tau, compared to FID envelope

Uses QuTiP's mesolve() with jump operators:
  L1 = sqrt(1/T1) sigma_-    -- energy relaxation
  L2 = sqrt(1/(2 T_phi)) sigma_z  -- pure dephasing

Reference parameters (matched loosely to Sheng 2018 and Manetsch 2025):
  T_1   = 30 s
  T_phi = 20 ms       -- dominated by differential light shift in baseline trap
  => T_2 = 1/(1/(2 T_1) + 1/T_phi) ~ 20 ms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import qutip as qt

rcParams["font.family"]      = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"]   = 12
rcParams["xtick.labelsize"]  = 10
rcParams["ytick.labelsize"]  = 10
rcParams["legend.fontsize"]  = 10
rcParams["axes.linewidth"]   = 0.8

# --- Parameters --------------------------------------------------------------
T1_s   = 30.0              # energy relaxation
Tphi_s = 20e-3             # pure dephasing (from diff light shift)
T2_s   = 1/(1/(2*T1_s) + 1/Tphi_s)   # transverse relaxation
print(f"T1 = {T1_s:.1f} s, T_phi = {Tphi_s*1e3:.0f} ms, T2 = {T2_s*1e3:.1f} ms")

# --- Pauli operators ---------------------------------------------------------
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
sm = qt.destroy(2)          # sigma_-

# Jump operators
L1 = np.sqrt(1/T1_s) * sm
L2 = np.sqrt(1/(2*Tphi_s)) * sz

# === Simulation (a): FID =====================================================
# Prepare |+> state = (|0> + |1>)/sqrt(2), which is the sigma_x +1 eigenstate
# Using convention: |0> = ground = sz eigenvalue -1 (QuTiP default)
psi0_plus = (qt.basis(2,0) + qt.basis(2,1)).unit()

tlist_FID = np.linspace(0, 80e-3, 400)   # 80 ms
result_FID = qt.mesolve(qt.Qobj(0*sx), psi0_plus, tlist_FID,
                        c_ops=[L1, L2], e_ops=[sx, sy, sz])

# Theoretical envelope exp(-t/T2)
env_FID = np.exp(-tlist_FID/T2_s)

# === Simulation (b): T1 inversion recovery ===================================
psi0_ex = qt.basis(2, 1)     # excited |1>
tlist_T1 = np.linspace(0, 100, 400)  # 100 s
result_T1 = qt.mesolve(qt.Qobj(0*sx), psi0_ex, tlist_T1,
                       c_ops=[L1, L2], e_ops=[sz])

# Theoretical: <sigma_z>(t) = -1 + 2 exp(-t/T1)
theo_T1 = -1 + 2*np.exp(-tlist_T1/T1_s)

# === Simulation (c): Hahn echo sequence ======================================
# For each delay tau, apply:
#   free evolution tau  ->  instantaneous pi-pulse (sigma_x)  ->  free evolution tau
# We simulate under pure dephasing + energy relaxation and add an extra
# QUASI-STATIC inhomogeneous dephasing that Hahn echo refocuses.
#
# To model T2* < T2, we add an ensemble over static frequency shifts drawn
# from a Gaussian with sigma_omega = 1/T2*.
T2_star_s = 5e-3   # 5 ms, limited by atomic temperature in non-magic trap
sigma_omega = np.sqrt(2)/T2_star_s  # so that FID decays as exp(-(t/T2*)^2/2)

tau_array = np.linspace(0, 50e-3, 40)

def hahn_signal(tau, n_ensemble=100):
    """Ensemble average of <sigma_x> after Hahn echo with delay tau."""
    signal = 0.0
    delta_samples = np.random.normal(0, sigma_omega, n_ensemble)
    for delta in delta_samples:
        H = 0.5 * delta * sz      # quasi-static detuning in rotating frame
        # Start in |+>, evolve tau under H + dissipators
        t_seg = np.linspace(0, tau, 40)
        # First free evolution
        r1 = qt.mesolve(H, psi0_plus, t_seg, c_ops=[L1, L2])
        psi_mid = r1.states[-1]
        # Pi-pulse about x-axis (instantaneous)
        U_pi = (-1j * np.pi/2 * sx).expm()
        psi_mid = U_pi * psi_mid * U_pi.dag() if psi_mid.isoper \
                  else U_pi * psi_mid
        # Second free evolution
        r2 = qt.mesolve(H, psi_mid, t_seg, c_ops=[L1, L2])
        psi_final = r2.states[-1]
        signal += qt.expect(sx, psi_final)
    return signal / n_ensemble

def FID_with_inhomogeneous(tau, n_ensemble=100):
    """Ensemble-averaged FID with quasi-static detuning + Markovian dephasing."""
    signal = 0.0
    delta_samples = np.random.normal(0, sigma_omega, n_ensemble)
    for delta in delta_samples:
        H = 0.5 * delta * sz
        t_seg = np.linspace(0, tau, 40)
        r = qt.mesolve(H, psi0_plus, t_seg, c_ops=[L1, L2])
        signal += qt.expect(sx, r.states[-1])
    return signal / n_ensemble

print("Simulating Hahn echo (this takes ~10 s)...")
np.random.seed(42)
hahn_vals = np.array([hahn_signal(t) for t in tau_array])
print("Simulating ensemble-averaged FID...")
np.random.seed(42)
fid_inhom = np.array([FID_with_inhomogeneous(t) for t in tau_array])

# === Figure 1: FID ============================================================
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(tlist_FID*1e3, result_FID.expect[0], color="#1f77b4", lw=1.6,
        label=r"$\langle \hat{\sigma}_x \rangle(t)$ (Lindblad)")
ax.plot(tlist_FID*1e3, env_FID, "--", color="#d62728", lw=1.2,
        label=rf"Envelope $e^{{-t/T_2}}$, $T_2 = {T2_s*1e3:.1f}$ ms")
ax.plot(tlist_FID*1e3, -env_FID, "--", color="#d62728", lw=1.2)
ax.set_xlabel(r"Time $t$ (ms)")
ax.set_ylabel(r"$\langle \hat{\sigma}_x \rangle$")
ax.set_xlim(0, 80)
ax.set_ylim(-1.1, 1.1)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig("fid_decay.pdf", bbox_inches="tight")
plt.savefig("fid_decay.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote fid_decay.pdf")

# === Figure 2: T1 inversion recovery ==========================================
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(tlist_T1, result_T1.expect[0], color="#1f77b4", lw=1.6,
        label=r"$\langle \hat{\sigma}_z \rangle(t)$ (Lindblad)")
ax.plot(tlist_T1, theo_T1, "--", color="#d62728", lw=1.2,
        label=rf"$-1 + 2e^{{-t/T_1}}$, $T_1 = {T1_s:.0f}$ s")
ax.axhline(0, color="gray", lw=0.3, alpha=0.5)
ax.axhline(-1, color="gray", lw=0.3, ls=":", alpha=0.5)
ax.axhline(+1, color="gray", lw=0.3, ls=":", alpha=0.5)
ax.set_xlabel(r"Time $t$ (s)")
ax.set_ylabel(r"$\langle \hat{\sigma}_z \rangle$")
ax.set_xlim(0, 100)
ax.set_ylim(-1.15, 1.15)
ax.grid(True, alpha=0.3)
ax.legend(loc="center right", frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig("t1_decay.pdf", bbox_inches="tight")
plt.savefig("t1_decay.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote t1_decay.pdf")

# === Figure 3: Hahn echo vs FID ===============================================
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.plot(tau_array*1e3, fid_inhom, "o-", color="#1f77b4", lw=1.4, markersize=4,
        label=rf"FID (decays as $T_2^* = {T2_star_s*1e3:.0f}$ ms)")
ax.plot(tau_array*1e3, hahn_vals, "s-", color="#d62728", lw=1.4, markersize=4,
        label=rf"Hahn echo (decays as $T_2 = {T2_s*1e3:.0f}$ ms)")

# theory curves
t_fine = np.linspace(0, tau_array[-1], 200)
ax.plot(t_fine*1e3, np.exp(-(t_fine/T2_star_s)**2/2)*np.exp(-t_fine/T2_s),
        "--", color="#1f77b4", lw=0.9, alpha=0.6)
ax.plot(t_fine*1e3, np.exp(-2*t_fine/T2_s),
        "--", color="#d62728", lw=0.9, alpha=0.6)

ax.set_xlabel(r"Total evolution time $2\tau$ (ms)")
ax.set_ylabel(r"Coherence $\langle \hat{\sigma}_x \rangle$")
ax.set_xlim(0, 50)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig("hahn_vs_fid.pdf", bbox_inches="tight")
plt.savefig("hahn_vs_fid.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote hahn_vs_fid.pdf")
