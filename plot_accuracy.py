import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from confidence_intervals import bootstrap_ci

# Data
budgets = np.array([0, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024, 2048, 4096, 8192, 16384, 24576])
ipcc_accuracy = np.array([41.67, 43.33, 44.67, 45.0, 41.0, 39.67, 39.0, 38.67, 40.0, 35.67, 42.33, 41.33, 38.67, 40.67, 41.33, 41.67])
iarc_accuracy = np.array([66.86, 62.39, 59.73, 61.44, 63.34, 60.4, 61.06, 60.68, 60.4, 59.92, 61.16, 60.87, 60.87, 62.39, 63.63, 62.2])

# Convert accuracy from percentage to proportion
ipcc_accuracy_prop = ipcc_accuracy / 100
iarc_accuracy_prop = iarc_accuracy / 100

# Compute bootstrap confidence intervals for IPCC (n=300)
n_samples_ipcc = 300
n_bootstrap = 10000
confidence_level = 0.95

ipcc_lower_bounds = []
ipcc_upper_bounds = []

for acc in ipcc_accuracy_prop:
    lower, mean, upper = bootstrap_ci(acc, n_samples_ipcc, n_bootstrap, confidence_level)
    ipcc_lower_bounds.append(lower * 100)  # Convert back to percentage
    ipcc_upper_bounds.append(upper * 100)

ipcc_lower_bounds = np.array(ipcc_lower_bounds)
ipcc_upper_bounds = np.array(ipcc_upper_bounds)

# Compute bootstrap confidence intervals for IARC (n=1053)
n_samples_iarc = 1053

iarc_lower_bounds = []
iarc_upper_bounds = []

for acc in iarc_accuracy_prop:
    lower, mean, upper = bootstrap_ci(acc, n_samples_iarc, n_bootstrap, confidence_level)
    iarc_lower_bounds.append(lower * 100)  # Convert back to percentage
    iarc_upper_bounds.append(upper * 100)

iarc_lower_bounds = np.array(iarc_lower_bounds)
iarc_upper_bounds = np.array(iarc_upper_bounds)

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Symmetric log scale parameters
linscale_val = 4  # approx log2(24576/512) ≈ 5.585

# Tick positions and labels
linear_ticks = [0, 64, 128, 192, 256, 320, 384, 448, 512]
log_ticks = [1024, 2048, 4096, 8192, 16384, 24576]
tick_positions = linear_ticks + log_ticks
tick_labels = ['None', '64', '128', '192', '256', '320', '384', '448', '512',
               '1024', '2048', '4096', '8192', '16384', '24576']

# Plot IPCC data (left subplot)
ax1.fill_between(budgets, ipcc_lower_bounds, ipcc_upper_bounds, alpha=0.3, color='skyblue', label=f'{int(confidence_level*100)}% Confidence Interval')
ax1.plot(budgets, ipcc_accuracy, marker='o', linewidth=3, color='navy', label='Mean Accuracy')
ax1.set_xscale('symlog', base=2, linthresh=512, linscale=linscale_val)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45)
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(plt.NullFormatter())
ax1.set_xlabel('Thinking Budget')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('IPCC Dataset', pad=15)
ax1.grid(True, which='both', linestyle='--', alpha=0.4)
ax1.legend()

# Plot IARC data (right subplot)
ax2.fill_between(budgets, iarc_lower_bounds, iarc_upper_bounds, alpha=0.3, color='lightcoral', label=f'{int(confidence_level*100)}% Confidence Interval')
ax2.plot(budgets, iarc_accuracy, marker='s', linewidth=3, color='red', label='Mean Accuracy')
ax2.set_xscale('symlog', base=2, linthresh=512, linscale=linscale_val)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45)
ax2.get_xaxis().set_major_formatter(ScalarFormatter())
ax2.get_xaxis().set_minor_formatter(plt.NullFormatter())
ax2.set_xlabel('Thinking Budget')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('IARC Dataset', pad=15)
ax2.grid(True, which='both', linestyle='--', alpha=0.4)
ax2.legend()

# Main title
fig.suptitle('Gemini 2.5 Flash – Accuracy vs Thinking Budget', fontsize=16, y=0.98)

plt.tight_layout()
plt.savefig('figure-accuracy.png', dpi=600, bbox_inches="tight")
plt.show()