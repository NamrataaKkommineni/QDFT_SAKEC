import matplotlib.pyplot as plt
import numpy as np

# Data
molecules = ['Benzene', 'Naphthalene', 'Anthracene', 'Tetracene', 'Pentacene']
active_spaces = ['2e,6o', '4e,6o', '6e,6o', '8e,6o']

# Original data values
data = {
    '2e,6o': [0.0018, 0.0005, -0.0010, -0.0015, 0.0000],
    '4e,6o': [-0.0121, -0.1724, -0.1885, -0.1743, -0.0399],
    '6e,6o': [-0.0089, -0.1574, -0.1391, -0.1335, -0.1287],
    '8e,6o': [-0.0077, -0.6961, -0.1891, -0.1834, -0.0385]
}

# Take absolute values for plotting on log scale
data_abs = {k: np.abs(v) for k, v in data.items()}

# Plot configuration
x = np.arange(len(molecules))
width = 0.18
colors = ['#4A79A7', '#E46C7C', '#2D8B3D', '#D4C85D']

fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each active space
for i, (space, values) in enumerate(data_abs.items()):
    offset = (i - 1.5) * width
    # Replace 0 with a very small value to prevent log errors (won't be visible on plot)
    plot_values = [v if v > 0 else 1e-10 for v in values]
    bars = ax.bar(x + offset, plot_values, width, label=space, color=colors[i], edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., val * 1.1,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7)

# Styling
ax.set_yscale('log')
ax.set_ylabel(r'$|\Delta E_{QDFT - FCI}|$ (mHa)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(molecules, fontsize=12)
ax.yaxis.grid(True, which='both', linestyle='-', alpha=0.3)
ax.set_axisbelow(True)

# Set limits for clear visibility
ax.set_ylim(1e-4, 2)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.legend(title='Active Space', frameon=True, edgecolor='lightgray', fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('mha_difference_log_barplot.png', dpi=300)