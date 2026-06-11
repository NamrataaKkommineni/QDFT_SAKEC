import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib as mpl

# --- Publication-ready settings ---
plt.style.use('default')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 9
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['figure.dpi'] = 300

# --- Creating the Figure with 1 Row, 2 Columns ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8))

# ==========================================
# PLOT 1: camb3lyp_alpha_delta_rho (Left)
# ==========================================

# --- Data Extraction ---
iter_a20 = np.arange(2, 26)
dp_a20 = np.array([3.79E-02, 2.84E-02, 2.24E-02, 1.83E-02, 1.54E-02, 1.31E-02, 1.13E-02, 9.93E-03, 8.77E-03, 7.81E-03, 7.01E-03, 6.32E-03, 5.73E-03, 5.22E-03, 4.92E-03, 4.63E-03, 4.36E-03, 4.11E-03, 3.88E-03, 3.65E-03, 3.44E-03, 3.24E-03, 3.06E-03, 2.88E-03])

iter_a40 = np.arange(2, 26)
dp_a40 = np.array([6.07E-02, 3.85E-02, 2.64E-02, 1.91E-02, 1.44E-02, 1.11E-02, 8.76E-03, 7.04E-03, 5.74E-03, 4.73E-03, 3.94E-03, 3.31E-03, 2.81E-03, 2.40E-03, 2.06E-03, 1.78E-03, 1.54E-03, 1.35E-03, 1.18E-03, 1.04E-03, 9.14E-04, 8.10E-04, 7.19E-04, 6.40E-04])

iter_a60 = np.arange(2, 26)
dp_a60 = np.array([6.87E-02, 3.56E-02, 2.08E-02, 1.31E-02, 8.74E-03, 6.05E-03, 4.31E-03, 3.15E-03, 2.35E-03, 1.79E-03, 1.38E-03, 1.07E-03, 8.45E-04, 6.72E-04, 5.41E-04, 4.38E-04, 3.56E-04, 2.93E-04, 2.42E-04, 2.01E-04, 1.68E-04, 1.40E-04, 1.19E-04, 9.98E-05])

iter_a75 = np.arange(2, 19)
dp_a75 = np.array([6.49E-02, 2.81E-02, 1.42E-02, 7.99E-03, 4.80E-03, 3.03E-03, 1.99E-03, 1.35E-03, 9.36E-04, 6.65E-04, 4.76E-04, 3.53E-04, 2.59E-04, 1.99E-04, 1.51E-04, 1.14E-04, 9.06E-05])

color_a20 = '#0057B8'    # Royal Blue
color_a40 = '#D55E00'    # Warm Orange
color_a60 = '#5FAF3A'    # Bright Green
color_a75 = '#C1272D'    # Strong Red

lw1, ms1 = 0.7, 2.0
ax1.plot(iter_a20, dp_a20, marker='o', ls='-', color=color_a20, mfc='white', lw=lw1, ms=ms1, label=r'$\alpha=0.2$')
ax1.plot(iter_a40, dp_a40, marker='s', ls='-', color=color_a40, mfc='white', lw=lw1, ms=ms1, label=r'$\alpha=0.4$')
ax1.plot(iter_a60, dp_a60, marker='D', ls='-', color=color_a60, mfc='white', lw=lw1, ms=ms1, label=r'$\alpha=0.6$')
ax1.plot(iter_a75, dp_a75, marker='^', ls='-', color=color_a75, mfc='white', lw=lw1, ms=ms1, label=r'$\alpha=0.75$')

ax1.set_yscale('log')
ax1.set_xlabel('Iteration', fontweight='bold')
ax1.set_ylabel(r'$\Delta\rho$', fontweight='bold')
ax1.set_xticks(np.arange(2, 27, 4))
ax1.grid(axis='both', alpha=0.4, lw=0.5, ls='--')

# Tightened Left Legend
ax1.legend(loc='upper right', frameon=True, fancybox=False, fontsize=6.5,
           borderpad=0.25, labelspacing=0.2, handlelength=1.2, handletextpad=0.3)


# ==========================================
# PLOT 2: camb3lyp_st_sp_delta_rho (Right)
# ==========================================

# --- Data Extraction: sp = 4 ---
iter_sp4_st9 = np.arange(4, 12)
dp_sp4_st9 = np.array([9.74E-03, 5.49E-03, 3.33E-03, 2.12E-03, 2.10E-03, 2.06E-03, 4.88E-04, 2.60E-05])

iter_sp4_st10 = np.arange(4, 17)
dp_sp4_st10 = np.array([9.74E-03, 5.49E-03, 3.33E-03, 2.13E-03, 1.42E-03, 1.46E-03, 8.88E-04, 2.25E-04, 1.46E-04, 2.58E-04, 2.33E-04, 1.05E-04, 4.06E-05])

iter_sp4_st8 = np.arange(4, 15)
dp_sp4_st8 = np.array([9.74E-03, 5.49E-03, 3.33E-03, 3.45E-03, 4.11E-03, 7.90E-04, 2.26E-04, 3.10E-04, 5.96E-04, 3.19E-04, 3.50E-05])

# --- Data Extraction: sp = 3 ---
iter_sp3_st9 = np.arange(4, 11)
dp_sp3_st9 = np.array([9.74E-03, 5.49E-03, 3.33E-03, 2.13E-03, 2.06E-03, 1.17E-03, 5.60E-05])

iter_sp3_st10 = np.arange(4, 12)
dp_sp3_st10 = np.array([9.74E-03, 5.49E-03, 3.33E-03, 2.13E-03, 1.42E-03, 2.22E-03, 1.43E-03, 8.43E-05])

iter_sp3_st8 = np.arange(4, 14)
dp_sp3_st8 = np.array([9.74E-03, 5.49E-03, 3.33E-03, 2.20E-03, 1.54E-03, 3.71E-04, 3.20E-04, 4.88E-04, 3.23E-04, 9.07E-05])

color_st9 = '#0057B8'    
color_st10 = '#D55E00'   
color_st8 = '#5FAF3A'   

lw2, ms2 = 0.7, 2.0
ax2.plot(iter_sp4_st9, dp_sp4_st9, marker='o', ls='-', color=color_st9, mfc='white', lw=lw2, ms=ms2, label='st=9, sp=4')
ax2.plot(iter_sp4_st10, dp_sp4_st10, marker='s', ls='-', color=color_st10, mfc='white', lw=lw2, ms=ms2, label='st=10, sp=4')
ax2.plot(iter_sp4_st8, dp_sp4_st8, marker='D', ls='-', color=color_st8, mfc='white', lw=lw2, ms=ms2, label='st=8, sp=4')

ax2.plot(iter_sp3_st9, dp_sp3_st9, marker='^', ls='--', color=color_st9, mfc='white', lw=lw2, ms=ms2, label='st=9, sp=3')
ax2.plot(iter_sp3_st10, dp_sp3_st10, marker='v', ls='--', color=color_st10, mfc='white', lw=lw2, ms=ms2, label='st=10, sp=3')
ax2.plot(iter_sp3_st8, dp_sp3_st8, marker='p', ls='--', color=color_st8, mfc='white', lw=lw2, ms=ms2, label='st=8, sp=3')

ax2.set_yscale('log')
ax2.set_xlabel('Iteration', fontweight='bold')
ax2.set_ylabel(r'$\Delta\rho$', fontweight='bold')
ax2.set_xticks(np.arange(4, 17, 2))
ax2.grid(axis='both', alpha=0.4, lw=0.5, ls='--')

# Tightened Right Legend (2 columns)
ax2.legend(loc='upper right', ncol=2, frameon=True, fancybox=False, fontsize=6.0, 
           borderpad=0.25, labelspacing=0.2, handlelength=1.2, handletextpad=0.3, columnspacing=0.6)


# ==========================================
# AGGRESSIVE MANUAL MARGIN PACKING
# ==========================================
plt.subplots_adjust(
    left=0.12,    # Increased significantly to protect outer left exponents from clipping
    right=0.96,   # Keeps right plot edges safely contained
    bottom=0.15,  # Substantial lift to fully display 'Iteration' labels
    top=0.93,     # Safe ceiling spacing
    wspace=0.35   # Expanded inner canal gap so plots don't crash into each other
)

# Best practice for saving without risk of clipping:
# plt.savefig('cam_final_merged_fixed.png', dpi=600, bbox_inches='tight')

plt.show()