import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib as mpl

# --- Publication-ready settings (Standardized) ---
plt.style.use('default')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['figure.dpi'] = 300

# --- Creating the Figure with 1 Row, 2 Columns ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

# ==========================================
# LEFT PANEL (ax1): J^2 Total Tuning Objective
# ==========================================

mu = np.array([
    0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3,
    0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6
])

J2_true = np.array([
    0.01202, 0.01119, 0.01067, 0.01002, 0.00940, 0.00890, 0.00848, 0.00813,
    0.00799, 0.00759, 0.00751, 0.00686, 0.00711, 0.00685, 0.00685, 0.00675,
    0.00667, 0.00659, 0.00652, 0.00646, 0.00641, 0.00636, 0.00632
])

color_j2 = '#5FAF3A'  # Bright Green
lw1, ms1 = 1, 1.5

# Plot J^2
ax1.plot(mu, J2_true, marker='^', ls='-', color=color_j2, mfc='white', lw=lw1, ms=ms1, label=r'True $J^2(\omega)$')

# Formatting ax1
ax1.set_xlabel(r'Range-Separation Parameter $\mu$ (Bohr$^{-1}$)', fontweight='bold')
ax1.set_ylabel(r'Objective Function $J^2$ (Ha$^2$)', fontweight='bold')
ax1.set_xticks(np.arange(0, 0.65, 0.1))
ax1.grid(axis='both', alpha=0.4, lw=0.5, ls='--')

# Optimized compact legend for ax1
ax1.legend(loc='upper right', frameon=True, fancybox=False, fontsize=7,
           borderpad=0.3, labelspacing=0.25, handlelength=1.5, handletextpad=0.4)


# ==========================================
# RIGHT PANEL (ax2): Parameter Tuning & Inset
# ==========================================

alpha_beta = np.array([
    0.26, 0.27, 0.28, 0.29, 0.30, 
    0.31, 0.32, 0.33, 0.34, 0.40, 
    0.50, 0.60
])

gap_ev = np.array([
    2.44664, 2.47633, 2.50683, 2.53842, 2.56998, 
    2.60193, 2.63434, 2.66663, 2.70048, 2.91691, 
    3.33233, 3.82009
])

target_gap = 2.50
color_line = '#0057B8'   # Royal Blue
color_target = '#D55E00'  # Warm Orange

# Plot Main Data and Target Line on ax2
lw2, ms2 = 1, 1.5
ax2.plot(alpha_beta, gap_ev, marker='o', ls='-', color=color_line, mfc='white', lw=lw2, ms=ms2, label='Calculated Gap')
ax2.axhline(target_gap, color=color_target, ls='--', lw=1.2, label='Target Gap (2.50 eV)')

# Formatting ax2
ax2.set_xlabel(r'Range-Separation Parameter $\alpha + \beta$', fontweight='bold')
ax2.set_ylabel('HOMO-LUMO Gap (eV)', fontweight='bold')
ax2.set_xlim(0.24, 0.62)
ax2.set_yticks(np.arange(2.4, 4.3, 0.5))
ax2.grid(axis='both', alpha=0.4, lw=0.5, ls='--')

# Optimized compact legend for ax2
ax2.legend(loc='lower right', frameon=True, fancybox=False, fontsize=7.0,
           borderpad=0.3, labelspacing=0.2, handlelength=1.4, handletextpad=0.3)

# --- Creating the Inset Zoom inside ax2 ---
# Adjusted boundaries to make the inset box slightly smaller and prevent text crowding
axins = ax2.inset_axes([0.20, 0.56, 0.38, 0.34])
axins.plot(alpha_beta, gap_ev, marker='o', ls='-', color=color_line, mfc='white', lw=0.9, ms=3.5)
axins.axhline(target_gap, color=color_target, ls='--', lw=1.2)

# Set the focus limits of the zoomed inset box
axins.set_xlim(0.26, 0.30)
axins.set_ylim(2.42, 2.58)
axins.grid(axis='both', alpha=0.4, lw=0.5, ls='--')
axins.set_title('Fine-Tuning Region', fontsize=7, pad=3)

# Scale down text internal to the zoom box
axins.tick_params(axis='both', labelsize=7, pad=2)

# Add the zoom indicator lines connecting to ax2
ax2.indicate_inset_zoom(axins, edgecolor="black", alpha=0.25, lw=0.8)


# ==========================================
# AGGRESSIVE MANUAL MARGIN PACKING
# ==========================================
plt.subplots_adjust(
    left=0.13,    # Safe cushion for left axis titles and scientific notation
    right=0.96,   # Contain right plot boundaries
    bottom=0.18,  # Safe lift for the bold X axis descriptors
    top=0.93,     # Headroom allocation
    wspace=0.35   # Widened canal between charts to keep labels clean and separate
)

# plt.savefig('tetracene_merged_tuning_fixed.png', dpi=600, bbox_inches='tight')
plt.show()