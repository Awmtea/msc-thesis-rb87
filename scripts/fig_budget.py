"""
Figure for Chapter 5: randomised benchmarking and the unified error budget.

Simulates:
  (a) Simulated RB decay: survival probability F(m) vs sequence length m
      for three operating scenarios (baseline, optimised, magic-intensity)
  (b) Error-budget comparison: bar chart of per-channel contributions

We DO NOT need to simulate the full Clifford RB in QuTiP here (expensive);
instead we use the well-known analytic result:
  F(m) = A * p^m + B,    p = 1 - 2*epsilon_per_Clifford
  epsilon ~ sum of individual error sources
This is what the experiments fit to, and the prediction is dominated by the
per-gate error we can compute analytically.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"]      = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"]   = 11
rcParams["xtick.labelsize"]  = 10
rcParams["ytick.labelsize"]  = 10
rcParams["legend.fontsize"]  = 9
rcParams["axes.linewidth"]   = 0.8

# --- Error models -----------------------------------------------------------
# For a pi-pulse at Rabi frequency Omega with:
#   - decoherence T2
#   - detuning error delta
#   - amplitude error epsilon
#   - leakage P_leak
#   - average Clifford = 1.875 pi-pulses

def error_decoherence(Omega, T2):
    """Infidelity from decoherence during a pi-pulse."""
    t_pi = np.pi / Omega
    return t_pi / (2 * T2)

def error_detuning(delta, Omega):
    return (np.pi * delta / (2 * Omega))**2

def error_amplitude(eps):
    return (np.pi * eps / 2)**2

def error_leakage(Omega, B_0):
    dnu_dB = 0.7e6  # Hz/G
    Delta_Z = 2*np.pi * dnu_dB * B_0
    return (Omega / Delta_Z)**2

# --- Three scenarios ---------------------------------------------------------

# Baseline: 850 nm linearly polarised trap, no pulse shaping
Omega_base = 2*np.pi * 20e3   # 20 kHz Rabi
T2_base    = 11e-3            # 11 ms Markovian T2
delta_base = 2*np.pi * 500    # 500 Hz detuning drift
eps_base   = 0.01             # 1% amplitude error
B0_base    = 3.23             # G

eps_dec_b  = error_decoherence(Omega_base, T2_base)
eps_det_b  = error_detuning(delta_base, Omega_base)
eps_amp_b  = error_amplitude(eps_base)
eps_leak_b = error_leakage(Omega_base, B0_base)
eps_total_baseline = eps_dec_b + eps_det_b + eps_amp_b + eps_leak_b

# Optimised: magic-intensity trap (suppresses diff light shift), better field
T2_opt     = 0.5    # 500 ms -- magic-intensity regime
delta_opt  = 2*np.pi * 200
eps_opt    = 0.005
B0_opt     = 5.0

eps_dec_o  = error_decoherence(Omega_base, T2_opt)
eps_det_o  = error_detuning(delta_opt, Omega_base)
eps_amp_o  = error_amplitude(eps_opt)
eps_leak_o = error_leakage(Omega_base, B0_opt)
eps_total_optimised = eps_dec_o + eps_det_o + eps_amp_o + eps_leak_o

# Sheng 2018: magic-intensity, near state-of-the-art (reported 3e-5)
eps_sheng = 3e-5

print(f"Baseline  epsilon per pi-pulse = {eps_total_baseline:.2e}")
print(f"Optimised epsilon per pi-pulse = {eps_total_optimised:.2e}")
print(f"  decoherence: {eps_dec_b:.2e} (base), {eps_dec_o:.2e} (opt)")
print(f"  detuning:    {eps_det_b:.2e} (base), {eps_det_o:.2e} (opt)")
print(f"  amplitude:   {eps_amp_b:.2e} (base), {eps_amp_o:.2e} (opt)")
print(f"  leakage:     {eps_leak_b:.2e} (base), {eps_leak_o:.2e} (opt)")

# --- Convert to per-Clifford error and RB decay ------------------------------
gates_per_clifford = 1.875
eps_clif_base = eps_total_baseline * gates_per_clifford
eps_clif_opt  = eps_total_optimised * gates_per_clifford
eps_clif_sheng = eps_sheng

def rb_curve(m, eps_clif):
    p = 1 - 2 * eps_clif
    A, B = 0.5, 0.5
    return A * p**m + B

m_array = np.arange(1, 1001)

# --- Figure 1: RB decay curves -----------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.semilogy(m_array, rb_curve(m_array, eps_clif_base) - 0.5, color="#1f77b4",
            lw=1.8, label=rf"Baseline: $\varepsilon_C = {eps_clif_base:.2e}$")
ax.semilogy(m_array, rb_curve(m_array, eps_clif_opt) - 0.5, color="#2ca02c",
            lw=1.8, ls="--", label=rf"Optimised: $\varepsilon_C = {eps_clif_opt:.2e}$")
ax.semilogy(m_array, rb_curve(m_array, eps_clif_sheng) - 0.5, color="#d62728",
            lw=1.8, ls=":", label=rf"Sheng 2018: $\varepsilon_C = 3{{\times}}10^{{-5}}$")

ax.set_xlabel(r"Sequence length $m$ (Clifford gates)")
ax.set_ylabel(r"RB signal $F(m) - 0.5$")
ax.set_xlim(1, 1000)
ax.set_ylim(1e-4, 1)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower left", frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig("rb_curves.pdf", bbox_inches="tight")
plt.savefig("rb_curves.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote rb_curves.pdf")

# --- Figure 2: Error budget bar chart ----------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.0))

categories = ["Decoherence\n($T_2$)", "Detuning", "Amplitude", "Leakage\n(m_F)"]
baseline_vals = [eps_dec_b,  eps_det_b,  eps_amp_b,  eps_leak_b]
optimised_vals = [eps_dec_o, eps_det_o, eps_amp_o, eps_leak_o]

x = np.arange(len(categories))
w = 0.35

ax.bar(x - w/2, baseline_vals,  w, color="#1f77b4",
       label=f"Baseline (total: {eps_total_baseline:.1e})", edgecolor="black", lw=0.5)
ax.bar(x + w/2, optimised_vals, w, color="#2ca02c",
       label=f"Optimised (total: {eps_total_optimised:.1e})", edgecolor="black", lw=0.5)

ax.axhline(1e-4, color="gray", ls="--", lw=0.8, alpha=0.7)
ax.text(3.3, 1.4e-4, "FT threshold $10^{-4}$", fontsize=9, color="gray", alpha=0.8)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel(r"Infidelity contribution")
ax.set_ylim(1e-6, 1e-1)
ax.grid(True, axis="y", which="both", alpha=0.3)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig("error_budget.pdf", bbox_inches="tight")
plt.savefig("error_budget.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote error_budget.pdf")

# --- Figure 3: Gate fidelity vs Rabi frequency (the optimisation plot) -------
Omega_array = 2*np.pi * np.logspace(3, 6, 200)  # 1 kHz to 1 MHz

fig, ax = plt.subplots(figsize=(6.5, 4.5))

for T2_val, B_val, color, label in [
    (T2_base, B0_base, "#1f77b4", "Baseline ($T_2 = 11$ ms, $B_0 = 3.23$ G)"),
    (T2_opt,  B0_opt,  "#2ca02c", "Optimised ($T_2 = 0.5$ s, $B_0 = 5$ G)"),
]:
    eps_dec  = np.array([error_decoherence(O, T2_val) for O in Omega_array])
    eps_leak = np.array([error_leakage(O, B_val) for O in Omega_array])
    eps_amp  = 0.01**2 * np.pi**2 / 4 + 0*Omega_array   # independent of Omega
    eps_det  = np.array([error_detuning(2*np.pi*500, O) for O in Omega_array])
    eps_tot  = eps_dec + eps_leak + eps_amp + eps_det
    ax.loglog(Omega_array/(2*np.pi)/1e3, eps_tot, color=color, lw=1.8, label=label)
    # mark minimum
    imin = np.argmin(eps_tot)
    ax.plot(Omega_array[imin]/(2*np.pi)/1e3, eps_tot[imin],
            "o", color=color, markersize=7)

ax.axhline(1e-4, color="gray", ls="--", lw=0.8, alpha=0.7)
ax.text(1.5, 1.3e-4, "FT threshold $10^{-4}$", fontsize=9, color="gray", alpha=0.8)

ax.set_xlabel(r"Rabi frequency $\Omega/2\pi$ (kHz)")
ax.set_ylabel(r"Total infidelity $\varepsilon_\pi$")
ax.set_xlim(1, 1000)
ax.set_ylim(1e-6, 1e-1)
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="lower right", frameon=True, framealpha=0.95)
plt.tight_layout()
plt.savefig("fidelity_vs_Omega.pdf", bbox_inches="tight")
plt.savefig("fidelity_vs_Omega.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote fidelity_vs_Omega.pdf")
