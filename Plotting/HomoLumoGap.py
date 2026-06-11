import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Publication-ready settings (Consistent with your previous bar plot theme)
plt.style.use('default')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 3.5             # Scaled down globally
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['figure.dpi'] = 600

# Data from Table III
molecules = ['Benzene', 'Naphthalene', 'Anthracene', 'Tetracene', 'Pentacene']
x = np.arange(len(molecules))

# Computational Functional Datasets
lda_rs = [12.793, 10.138, 8.398, 7.176, 6.286]
lrc_wpbe = [10.806, 8.339, 6.880, 5.839, 5.065]
cam_default = [9.023, 6.606, 5.146, 4.090, 3.290]
b3lyp = [6.422, 4.366, 3.121, 2.283, 1.694]
cam_tuned = [6.812, 4.674, 3.385, 2.507, 1.885]

# Experimental Reference Data (Note: Benzene is omitted/None)
experimental = [6.93, 4.400, 3.400, 2.500, 2.200]

# Figure dimensions matching your previous plot perfectly
fig, ax = plt.subplots(figsize=(3.5, 2.6))

# --- PLOT DATA WITH DISTINCT MARKERS AND LINE STYLES ---
ax.plot(x, lda_rs, label='LDA_RS', color='#1f77b4', marker='o', 
        markersize=1, linewidth=0.8, linestyle='-')

ax.plot(x, lrc_wpbe, label='LRC-wPBE', color='#9467bd', marker='v', 
        markersize=1, linewidth=0.8, linestyle='--')

ax.plot(x, cam_default, label='CAM-B3LYP (Def.)', color='#ff7f0e', marker='^', 
        markersize=1, linewidth=0.8, linestyle='-.')

ax.plot(x, b3lyp, label='B3LYP', color='#2ca02c', marker='s', 
        markersize=1, linewidth=0.8, linestyle=':')
ax.plot(x, cam_tuned, label='CAM-B3LYP (Tuned)', color='#d62728', marker='D', 
        markersize=1, linewidth=0.8, linestyle='-')

# Experimental data plotted as discrete points (no line) per standard convention
ax.plot(x, experimental, label='Experimental', color='black', marker='o', 
        markersize=2, linestyle='None', markerfacecolor='white', markeredgewidth=0.8)

# --- AXES GRID AND LIMITS ---
ax.grid(axis='both', alpha=0.25, lw=0.4, linestyle='--')
ax.set_ylim(0.0, 14.0)
ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])

# --- LABELS & TYPOGRAPHY ---
ax.tick_params(axis='both', labelsize=2.5, pad=2)

ax.set_xticks(x)
ax.set_xticklabels(molecules, fontweight='bold', fontsize=2.5)

ax.set_xlabel('Linear Polyacenes', fontsize=3.5, fontweight='bold', labelpad=2)
ax.set_ylabel('HOMO–LUMO Gap (eV)', fontsize=3.5, fontweight='bold', labelpad=4)

# --- FIXED LEGEND POSITION ---
# Neatly stacked in the upper right white space area
ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=2.5,
          borderpad=0.4, labelspacing=0.5, handlelength=1.5, handletextpad=0.4)

# Margin adjustments matching the visual weight of your prior plot
plt.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.15)

plt.show()