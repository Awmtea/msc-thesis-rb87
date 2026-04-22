"""
Plot: Zeeman shift of the 87Rb clock transition.

Shows the shift of the |F=1, m_F=0> <-> |F=2, m_F=0> transition frequency
as a function of the magnetic field B, comparing:

  (i)  the exact Breit-Rabi result  nu_hf * sqrt(1 + x^2) - nu_hf
  (ii) the quadratic approximation  beta * B^2

with a zoomed inset at low field.

Generates: clock_zeeman_shift.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# --- Plot style ---------------------------------------------------------------
rcParams["font.family"]      = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.labelsize"]   = 12
rcParams["xtick.labelsize"]  = 10
rcParams["ytick.labelsize"]  = 10
rcParams["legend.fontsize"]  = 10
rcParams["axes.linewidth"]   = 0.8

# --- Fundamental constants ----------------------------------------------------
h      = 6.62607015e-34         # Planck constant (J.s)
mu_B   = 9.2740100783e-24       # Bohr magneton (J/T)

# --- 87Rb parameters ----------------------------------------------------------
nu_hf    = 6.834_682_610_904e9  # Hz
DeltaE   = h * nu_hf            # J
g_J      = 2.00233113
g_I      = -0.0009951414

# --- Derived: beta = (g_J - g_I)^2 * mu_B^2 / (2 * h * DeltaE)  ---------------
# In Hz/T^2, convert to Hz/G^2 (1 T = 1e4 G, so 1 T^2 = 1e8 G^2)
beta_Hz_per_T2 = (g_J - g_I)**2 * mu_B**2 / (2 * h * DeltaE)
beta_Hz_per_G2 = beta_Hz_per_T2 * 1e-8

print(f"beta = {beta_Hz_per_G2:.4f} Hz/G^2")

# --- Exact shift from Breit-Rabi ----------------------------------------------
def shift_exact(B_gauss):
    """Exact clock-transition shift (Hz) relative to zero-field value."""
    B_tesla = B_gauss * 1e-4
    x = (g_J - g_I) * mu_B * B_tesla / DeltaE
    return nu_hf * (np.sqrt(1 + x**2) - 1)

def shift_quadratic(B_gauss):
    """Quadratic approximation: beta * B^2."""
    return beta_Hz_per_G2 * B_gauss**2

# --- Generate data ------------------------------------------------------------
B_full = np.linspace(0, 500, 1001)          # wide range
B_zoom = np.linspace(0, 10,  501)           # low field (lab range)

shift_exact_full = shift_exact(B_full)
shift_quad_full  = shift_quadratic(B_full)

shift_exact_zoom = shift_exact(B_zoom)
shift_quad_zoom  = shift_quadratic(B_zoom)

# --- Figure -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 5.0))

# Main plot: 0 to 500 G, shift in MHz
ax.plot(B_full, shift_exact_full/1e6, color="black", lw=2.0,
        label=r"Exact: $\nu_{\mathrm{hf}}(\sqrt{1+x^2}-1)$")
ax.plot(B_full, shift_quad_full/1e6, color="#d62728", ls="--", lw=1.5,
        label=r"Quadratic approx.: $\beta B^2$")

ax.set_xlabel(r"Magnetic field $B$ (G)")
ax.set_ylabel(r"Clock transition shift $\nu_{\mathrm{clock}}(B) - \nu_{\mathrm{hf}}$ (MHz)")
ax.set_xlim(0, 500)
ax.set_ylim(0, shift_exact(500)/1e6 * 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", frameon=True, framealpha=0.95)

# Annotate where the approximation starts to deviate
B_mark = 230
ax.axvline(B_mark, color="gray", ls=":", lw=0.8, alpha=0.7)
ax.text(B_mark + 8, 10,
        r"$B \approx 230$ G:" "\n" r"$x \sim 0.1$" "\n" r"approx. $\sim 1\%$ off",
        fontsize=9, color="gray", va="bottom")

# Inset: zoom into lab field range 0-10 G, shift in kHz
axin = inset_axes(ax, width="45%", height="45%", loc="center right",
                  bbox_to_anchor=(-0.05, 0.05, 1, 1),
                  bbox_transform=ax.transAxes)
axin.plot(B_zoom, shift_exact_zoom/1e3, color="black", lw=2.0)
axin.plot(B_zoom, shift_quad_zoom/1e3,  color="#d62728", ls="--", lw=1.5)
axin.set_xlabel(r"$B$ (G)", fontsize=9)
axin.set_ylabel(r"shift (kHz)", fontsize=9)
axin.set_xlim(0, 10)
axin.set_ylim(0, 60)
axin.tick_params(labelsize=8)
axin.grid(True, alpha=0.3)
axin.set_title("Lab-field range", fontsize=9, pad=3)

# Annotate B=1 G point in the inset
shift_1G = shift_exact(1.0) / 1e3  # kHz
axin.plot([1], [shift_1G], "o", color="#1f77b4", markersize=6)
axin.annotate(f"$B=1$ G:\n575 Hz",
              xy=(1, shift_1G), xytext=(2.5, 15),
              fontsize=8, color="#1f77b4",
              arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8))

plt.tight_layout()
plt.savefig("clock_zeeman_shift.pdf", bbox_inches="tight")
plt.savefig("clock_zeeman_shift.png", bbox_inches="tight", dpi=200)
print("Wrote clock_zeeman_shift.pdf and clock_zeeman_shift.png")
