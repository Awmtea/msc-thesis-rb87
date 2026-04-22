"""
Figure 2.1: Breit-Rabi energy diagram for the 87Rb 5 2S_{1/2} ground state.

Plots all eight magnetic sublevels |F, m_F> as a function of the applied
magnetic field B in Gauss. The clock states |F=1, m_F=0> and |F=2, m_F=0>
are highlighted in bold to show their first-order field insensitivity.

Outputs: breit_rabi.pdf

Physics reference:
  Breit & Rabi, Phys. Rev. 38, 2082 (1931).
  Numerical constants from D.A. Steck, "Rubidium 87 D Line Data" (v2.3.4, 2024).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- Plot style ---------------------------------------------------------------
rcParams["font.family"]     = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"]   = 12
rcParams["xtick.labelsize"]  = 10
rcParams["ytick.labelsize"]  = 10
rcParams["legend.fontsize"]  = 9
rcParams["axes.linewidth"]   = 0.8

# --- Fundamental constants ----------------------------------------------------
h    = 6.62607015e-34        # Planck constant (J.s)
mu_B = 9.2740100783e-24      # Bohr magneton   (J/T)

# --- 87Rb 5 2S_{1/2} parameters (Steck, 2024 revision) ------------------------
nu_hf       = 6.834_682_610_904e9   # hyperfine splitting (Hz)
DeltaE_hf   = h * nu_hf             # hyperfine splitting in J
g_J         = 2.00233113            # electronic Lande g-factor
g_I         = -0.0009951414         # nuclear g-factor (in units of mu_B)
I_nuc       = 3/2                   # nuclear spin

# --- Breit-Rabi formula -------------------------------------------------------
def breit_rabi(B_gauss, F, m_F):
    """
    Energy of a |F, m_F> sublevel of the 5 2S_{1/2} ground state
    in the Breit-Rabi parameterisation.

    Returns energy / h in Hz.
    """
    B_tesla = B_gauss * 1e-4
    x       = (g_J - g_I) * mu_B * B_tesla / DeltaE_hf
    twoI1   = 2 * I_nuc + 1
    sign    = +1.0 if F == 2 else -1.0

    # Stretched states |F=2, m_F = +/- 2> have no F=1 partner; the
    # Breit-Rabi formula still reproduces them correctly because the
    # square-root collapses to |1 +/- x|.
    radical = np.sqrt(1.0 + 4.0*m_F*x/twoI1 + x**2)

    E = -DeltaE_hf/(2*twoI1) \
        + g_I * mu_B * m_F * B_tesla \
        + sign * (DeltaE_hf/2.0) * radical
    return E / h        # Hz

# --- Build the figure ---------------------------------------------------------
B_array = np.linspace(0.0, 500.0, 2001)   # field in Gauss

fig, ax = plt.subplots(figsize=(6.5, 5.0))

# F = 2 branch ----------------------------------------------------------------
for m_F in [-2, -1, 0, +1, +2]:
    E_MHz = np.array([breit_rabi(B, 2, m_F) for B in B_array]) / 1e6
    if m_F == 0:
        ax.plot(B_array, E_MHz, color="black", lw=2.0,
                label=r"$|F=2,\,m_F=0\rangle$ (clock)")
    else:
        ax.plot(B_array, E_MHz, color="#d62728", lw=1.0, alpha=0.85)

# F = 1 branch ----------------------------------------------------------------
for m_F in [-1, 0, +1]:
    E_MHz = np.array([breit_rabi(B, 1, m_F) for B in B_array]) / 1e6
    if m_F == 0:
        ax.plot(B_array, E_MHz, color="black", lw=2.0,
                label=r"$|F=1,\,m_F=0\rangle$ (clock)")
    else:
        ax.plot(B_array, E_MHz, color="#1f77b4", lw=1.0, alpha=0.85)

# --- Annotations --------------------------------------------------------------
def mF_label(m):
    """Pretty m_F label: '0', '+1', '-1', etc. (no '+0')."""
    return "$0$" if m == 0 else f"${m:+d}$"

# m_F labels at high field (right-hand end)
B_label = 500.0
for m_F in [-2, -1, 0, +1, +2]:
    E = breit_rabi(B_label, 2, m_F) / 1e6
    ax.text(B_label + 10, E, mF_label(m_F), fontsize=9,
            va="center", ha="left", color="#d62728")
for m_F in [-1, 0, +1]:
    E = breit_rabi(B_label, 1, m_F) / 1e6
    ax.text(B_label + 10, E, mF_label(m_F), fontsize=9,
            va="center", ha="left", color="#1f77b4")

# F=1, F=2 manifold labels (placed off to the side, away from curves)
ax.text(545, 2550, r"$F=2$", fontsize=12, color="#d62728",
        fontweight="bold", ha="left", va="center")
ax.text(545, -4270, r"$F=1$", fontsize=12, color="#1f77b4",
        fontweight="bold", ha="left", va="center")

# Hyperfine splitting arrow at B ~ 30 G
E_F2_0 = breit_rabi(30, 2, 0) / 1e6
E_F1_0 = breit_rabi(30, 1, 0) / 1e6
ax.annotate("", xy=(30, E_F2_0), xytext=(30, E_F1_0),
            arrowprops=dict(arrowstyle="<->", color="gray", lw=0.8))
ax.text(45, 0.5*(E_F2_0 + E_F1_0),
        r"$\nu_{\mathrm{hf}} \approx 6.835$ GHz",
        fontsize=9, color="gray", va="center")

# --- Cosmetics ----------------------------------------------------------------
ax.set_xlabel(r"Magnetic field $B$ (G)")
ax.set_ylabel(r"Energy / $h$ (MHz)")
ax.set_xlim(0, 600)
ax.set_ylim(-5500, 4500)
ax.axhline(0, color="gray", lw=0.3, ls=":")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left", frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig("breit_rabi.pdf", bbox_inches="tight")
plt.savefig("breit_rabi.png", bbox_inches="tight", dpi=200)  # convenience
print("Wrote breit_rabi.pdf and breit_rabi.png")
