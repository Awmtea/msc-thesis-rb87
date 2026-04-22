"""
Figure for Chapter 3: Leakage probability during a pi-pulse as a function
of bias field B_0, for several Rabi frequencies.

P_leak ~ (Omega / Delta_omega_Z)^2
where Delta_omega_Z = g_F2 * mu_B * B / hbar = 2*pi * (0.7 MHz/G) * B

Outputs: leakage_vs_B.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"]      = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"]   = 12
rcParams["xtick.labelsize"]  = 10
rcParams["ytick.labelsize"]  = 10
rcParams["legend.fontsize"]  = 10
rcParams["axes.linewidth"]   = 0.8

# Zeeman splitting between neighbouring m_F levels in F=2 manifold
# g_F=2 = +1/2, so Delta_omega_Z = mu_B B / (2 hbar) = 2*pi * 0.7 MHz/G * B
dnu_dB = 0.7e6  # Hz/G (first-order Zeeman shift per G, for m_F=1 in F=2)

B_array = np.logspace(-1, 1.5, 500)  # 0.1 to 30 G

fig, ax = plt.subplots(figsize=(6.5, 4.2))
colors = ["#1f77b4", "#d62728", "#2ca02c"]
linestyles = ["-", "--", ":"]
for (Omega_kHz, color, ls) in zip([5, 20, 100], colors, linestyles):
    Omega = 2*np.pi * Omega_kHz * 1e3
    # Delta_omega_Z in rad/s
    Delta_Z = 2*np.pi * dnu_dB * B_array  # rad/s
    P_leak = (Omega / Delta_Z)**2
    ax.loglog(B_array, P_leak, color=color, lw=1.7, linestyle=ls,
              label=rf"$\Omega/2\pi = {Omega_kHz}$ kHz")

ax.axhline(1e-3, color="gray", ls=":", lw=0.7, alpha=0.7)
ax.axhline(1e-4, color="gray", ls=":", lw=0.7, alpha=0.7)
ax.text(12, 1.2e-3, r"$10^{-3}$", fontsize=9, color="gray")
ax.text(12, 1.2e-4, r"$10^{-4}$", fontsize=9, color="gray")

# Sheng 2018 operating point
B_sheng = 3.23
ax.axvline(B_sheng, color="black", ls="-.", lw=0.7, alpha=0.5)
ax.text(B_sheng*1.1, 2e-7, "Sheng 2018\n$B_0=3.23$ G",
        fontsize=8, color="black", alpha=0.7)

ax.set_xlabel(r"Bias field $B_0$ (G)")
ax.set_ylabel(r"Leakage probability $P_{\mathrm{leak}}$")
ax.set_xlim(0.1, 30)
ax.set_ylim(1e-7, 1)
ax.legend(loc="upper right", frameon=True, framealpha=0.95)
ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("leakage_vs_B.pdf", bbox_inches="tight")
plt.savefig("leakage_vs_B.png", bbox_inches="tight", dpi=200)
print("Wrote leakage_vs_B.pdf")
