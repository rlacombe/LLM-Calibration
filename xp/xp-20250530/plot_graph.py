import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Data
budgets = np.array([0, 64, 128, 192, 256, 320, 384, 448, 512,
                    1024, 2048, 4096, 8192, 16384, 24576])
accuracy = np.array([41.7, 43.3, 44.7, 45.0, 41.0, 39.7, 39.0, 38.7,
                     40.0, 42.3, 41.3, 38.7, 40.7, 41.3, 41.7])

fig, ax = plt.subplots(figsize=(12, 7))

# Navy blue line
ax.plot(budgets, accuracy, marker='o', linewidth=3, color='navy')

# Symmetric log: linear up to 512 with linscale chosen so linear occupies ~ half axis
linscale_val = 5  # approx log2(24576/512) ≈ 5.585
ax.set_xscale('symlog', base=2, linthresh=512, linscale=linscale_val)

# Tick positions and labels
linear_ticks = [0, 64, 128, 192, 256, 320, 384, 448, 512]
log_ticks = [1024, 2048, 4096, 8192, 16384, 24576]
tick_positions = linear_ticks + log_ticks
tick_labels = ['None', '64', '128', '192', '256', '320', '384', '448', '512',
               '1024', '2048', '4096', '8192', '16384', '24576']

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45)
ax.get_xaxis().set_major_formatter(ScalarFormatter())
ax.get_xaxis().set_minor_formatter(plt.NullFormatter())

# Labels and title
ax.set_xlabel('Thinking Budget')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Gemini 2.5 Flash – Accuracy vs Thinking Budget', pad=15)

# Grid
ax.grid(True, which='both', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()