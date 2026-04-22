"""
Figure for Chapter 3: Rabi chevron and on-resonance Rabi oscillations.

We simulate the driven two-level Hamiltonian in the rotating frame:
  H_RWA = -(hbar*delta/2) * sigma_z + (hbar*Omega/2) * sigma_x

Starting from |0>, we compute the probability P|1>(t) = sin^2(Omega_eff * t/2) * (Omega/Omega_eff)^2
as a function of detuning delta and time t.

Outputs:
  rabi_chevron.pdf     -- 2D colormap of P|1>(delta, t)
  rabi_oscillations.pdf -- three on-resonance + off-resonance traces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- Style -------------------------------------------------------------------
rcParams["font.family"]      = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"]   = 12
rcParams["xtick.labelsize"]  = 10
rcParams["ytick.labelsize"]  = 10
rcParams["legend.fontsize"]  = 10
rcParams["axes.linewidth"]   = 0.8

# --- Parameters --------------------------------------------------------------
Omega_over_2pi = 20e3           # 20 kHz Rabi frequency
Omega          = 2*np.pi * Omega_over_2pi

# Time and detuning grids
t_array       = np.linspace(0, 250e-6, 501)        # 250 us
delta_array   = 2*np.pi * np.linspace(-100e3, 100e3, 401)  # +/- 100 kHz

# --- Rabi formula ------------------------------------------------------------
def P1(Omega, delta, t):
    """Excited-state population for a two-level Rabi drive."""
    Omega_eff = np.sqrt(Omega**2 + delta**2)
    return (Omega/Omega_eff)**2 * np.sin(Omega_eff * t / 2)**2

# === Figure 1: Rabi chevron ===================================================
T, D = np.meshgrid(t_array, delta_array)
P = P1(Omega, D, T)

fig, ax = plt.subplots(figsize=(6.8, 4.5))
pcm = ax.pcolormesh(T*1e6, D/(2*np.pi)/1e3, P,
                    cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
cbar = plt.colorbar(pcm, ax=ax, label=r"$P_{|1\rangle}$")

# Mark the pi-pulse and 2pi-pulse times on resonance
t_pi  = np.pi / Omega
t_2pi = 2*np.pi / Omega
ax.axvline(t_pi*1e6,  color="white", ls=":", lw=1.0)
ax.axvline(t_2pi*1e6, color="white", ls=":", lw=1.0)
ax.text(t_pi*1e6+2, 85,  r"$t_\pi$",  color="white", fontsize=10)
ax.text(t_2pi*1e6+2, 85, r"$t_{2\pi}$", color="white", fontsize=10)

ax.set_xlabel(r"Pulse time $t$ ($\mu$s)")
ax.set_ylabel(r"Detuning $\delta/2\pi$ (kHz)")
ax.set_title(rf"Rabi chevron, $\Omega/2\pi = {Omega_over_2pi/1e3:.0f}$ kHz")
plt.tight_layout()
plt.savefig("rabi_chevron.pdf", bbox_inches="tight")
plt.savefig("rabi_chevron.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote rabi_chevron.pdf")

# === Figure 2: Rabi oscillations at a few detunings ===========================
fig, ax = plt.subplots(figsize=(6.5, 4.0))

cases = [
    (0,            "Resonant ($\\delta = 0$)",                "black",    "-"),
    (2*np.pi*10e3, "$\\delta/2\\pi = 10$ kHz = $\\Omega/2$",   "#1f77b4", "--"),
    (2*np.pi*20e3, "$\\delta/2\\pi = 20$ kHz = $\\Omega$",     "#d62728", "-."),
    (2*np.pi*40e3, "$\\delta/2\\pi = 40$ kHz = $2\\Omega$",    "#2ca02c", ":"),
]

for delta_val, label, color, ls in cases:
    p = P1(Omega, delta_val, t_array)
    ax.plot(t_array*1e6, p, color=color, lw=1.6, label=label, linestyle=ls)

ax.axhline(1.0, color="gray", lw=0.5, alpha=0.5)
ax.axvline(t_pi*1e6, color="gray", lw=0.5, ls=":", alpha=0.7)
ax.text(t_pi*1e6+1, 0.05, r"$t_\pi$", color="gray", fontsize=9)

ax.set_xlabel(r"Time $t$ ($\mu$s)")
ax.set_ylabel(r"$P_{|1\rangle}(t)$")
ax.set_xlim(0, 250)
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rabi_oscillations.pdf", bbox_inches="tight")
plt.savefig("rabi_oscillations.png", bbox_inches="tight", dpi=200)
plt.close()
print("Wrote rabi_oscillations.pdf")
