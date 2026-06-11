import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib as mpl

# Publication-ready settings
plt.style.use('default')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['figure.dpi'] = 300
rcParams['legend.fontsize'] = 8

# Data
molecules = ['Benzene', 'Naphthalene', 'Anthracene', 'Tetracene', 'Pentacene']

# Absolute values
diffs_2e6o = np.abs([0.0000, 0.0000, 0.0000, -0.0020, 0.0000])
diffs_4e6o = np.abs([-0.0090, -0.1740, -0.1850, -0.1740, -0.0400])
diffs_6e6o = np.abs([-0.0100, -0.1580, -0.1370, -0.1340, -0.1290])
diffs_8e6o = np.abs([-0.0090, -0.6990, -0.1920, -0.1830, -0.0390])

# Compact figure - USING TWO SUBPLOTS
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3.4, 2.8), 
                               gridspec_kw={'height_ratios': [1, 3]})

x = np.arange(len(molecules))
width = 0.2

# Fine hatch rendering
mpl.rcParams['hatch.linewidth'] = 0.55
rcParams['figure.dpi'] = 600

# Color palette 
edgecolor = [
    '#D55E00',   # warm orange
    '#0057B8',   # royal blue
    '#5FAF3A',   # balanced bright green
    '#4D4D4D'    # soft charcoal black
]

# Fine integrated textures
hatches = [
    '///////////////',
    '\\\\\\\\\\\\\\\\\\\\\\',
    '||||||||||||',
    '-------------'
]

# Plot bars on BOTH axes
for ax in [ax1, ax2]:
    ax.bar(x - 1.5*width, diffs_2e6o, width, label='2e,6o', facecolor='white', 
           hatch=hatches[0], edgecolor=edgecolor[0], linewidth=0.5)
    ax.bar(x - 0.5*width, diffs_4e6o, width, label='4e,6o', facecolor='white', 
           hatch=hatches[1], edgecolor=edgecolor[1], linewidth=0.5)
    ax.bar(x + 0.5*width, diffs_6e6o, width, label='6e,6o', facecolor='white', 
           hatch=hatches[2], edgecolor=edgecolor[2], linewidth=0.5)
    ax.bar(x + 1.5*width, diffs_8e6o, width, label='8e,6o', facecolor='white', 
           hatch=hatches[3], edgecolor=edgecolor[3], linewidth=0.5)
    
    ax.grid(axis='y', alpha=0.4, lw=0.5, linestyle='--')
    ax.tick_params(axis='both', labelsize=7.5)

# --- SET Y-LIMITS FOR THE BREAK ---
ax1.set_ylim(0.65, 0.72)  
ax2.set_ylim(0.0, 0.22)   

# Hide the spines between ax1 and ax2 so they merge
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.tick_params(axis='x', which='both', bottom=False)  

# --- ADD DIAGONAL CUT MARKS ---
d = .015  
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth=1.0)
ax1.plot((-d, +d), (-d, +d), **kwargs)               
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)         

kwargs.update(transform=ax2.transAxes)               
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)         
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)   

# --- FORMATTING ---
ax2.set_xticks(x)

# 1. BOLD THE X-AXIS LABELS
ax2.set_xticklabels(molecules, fontweight='bold')

# 2. BOLD THE Y-AXIS LABEL
fig.text(0.02, 0.5, '|ΔE$_{QDFT-FCI}$| (mHa)', va='center', rotation='vertical', 
         fontsize=8.5, fontweight='bold')

# --- LEGEND FIX ---
# 3. Use fig.legend instead of ax1.legend so it is immune to subplot clipping
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), 
           loc='upper right',
           bbox_to_anchor=(0.94, 0.94), # Coordinates relative to the whole figure
           frameon=True, fancybox=False, fontsize=7,
           borderpad=0.3, handlelength=1.2, handletextpad=0.4)

# Keep hspace at 0.0 to ensure the continuous bar effect
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15, hspace=0.0)

plt.show()