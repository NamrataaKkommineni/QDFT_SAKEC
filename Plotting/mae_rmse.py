import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib as mpl

# Publication-ready settings (Scaled down for compact figure sizes)
plt.style.use('default')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 3.5             # Scaled down globally
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['figure.dpi'] = 600

# Data from Table II
functionals = ['LDA_RS', 'LRC-wPBE', 'CAM-B3LYP\n(Def.)', 'B3LYP', 'CAM-B3LYP\n(Tuned)']
mae_data = [0.1135, 0.2018, 0.4399, 1.3627, 1.1321]
rmse_data = [0.1251, 0.2245, 0.4956, 1.4773, 1.2376]

# Compact figure dimensions (Now a single subplot)
fig, ax = plt.subplots(figsize=(3.5, 2.6))

x = np.arange(len(functionals))
width = 0.30

# Fine hatch rendering
mpl.rcParams['hatch.linewidth'] = 0.45

# Premium palette matching your original plot
edgecolor = ['#D55E00', '#0057B8']
hatches = ['///////////////', '\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\']

# Plot bars directly on the single axis
ax.bar(x - 0.5*width, mae_data, width, label='MAE', facecolor='white', 
       hatch=hatches[0], edgecolor=edgecolor[0], linewidth=0.5)
ax.bar(x + 0.5*width, rmse_data, width, label='RMSE', facecolor='white', 
       hatch=hatches[1], edgecolor=edgecolor[1], linewidth=0.5)
    
ax.grid(axis='y', alpha=0.25, lw=0.4, linestyle='--')
ax.tick_params(axis='both', labelsize=3.5, pad=3)

# --- Y-LIMITS ---
# Set a clean ceiling just above your highest value (1.4773)
ax.set_ylim(0.0, 1.7)  

# --- X-AXIS LABELS ---
ax.set_xticks(x)
ax.set_xticklabels(functionals, fontweight='bold', fontsize=3.5, rotation=0, ha='center')

# --- Y-AXIS LABEL ---
ax.set_ylabel('MAE / RMSE (mHa)', fontsize=4.5, fontweight='bold')

# --- LEGEND ---
ax.legend(loc='upper left', 
          frameon=True, fancybox=False, fontsize=3.5,
          borderpad=0.3, labelspacing=0.3, handlelength=1.5, handletextpad=0.3)

# 1. Apply tight_layout for general internal spacing
plt.tight_layout()

# 2. OVERRIDE: Manually push the bottom margin up to ensure multi-line X-labels fit perfectly
plt.subplots_adjust(bottom=0.15) 
# 2. OVERRIDE: Manually push the left margin up to ensure multi-line Y-labels fit perfectly
plt.subplots_adjust(left=0.12) 
# Uncomment to save (using bbox_inches='tight' guarantees nothing is clipped in the saved file)
# plt.savefig('mae_rmse_single_axis.png', dpi=600, bbox_inches='tight')

plt.show()