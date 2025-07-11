import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from confidence_intervals import bootstrap_ci

# Data
budgets = np.array([0, 64, 128, 192, 256, 320, 384, 448, 512, 768,
                    1024, 2048, 4096, 8192, 16384, 24576])
ipcc_confidence = np.array([1.59, 1.61, 1.64, 1.77, 1.79, 1.73, 1.82, 1.81, 1.8, 1.76, 1.68, 1.74, 1.73, 1.71, 1.66, 1.76])
iarc_confidence = np.array([1.04, 1.08, 1.22, 1.18, 1.17, 1.22, 1.15, 1.13, 1.13, 1.11, 1.12, 1.09, 1.14, 1.12, 1.09, 1.05])

# Convert confidence to proportion for bootstrap (assuming max confidence is 3 for IPCC and 3 for IARC)
ipcc_confidence_prop = ipcc_confidence / 3
iarc_confidence_prop = iarc_confidence / 3

# Compute bootstrap confidence intervals for IPCC (n=300)
n_samples_ipcc = 300
n_bootstrap = 10000
confidence_level = 0.95

ipcc_lower_bounds = []
ipcc_upper_bounds = []

for conf in ipcc_confidence_prop:
    lower, mean, upper = bootstrap_ci(conf, n_samples_ipcc, n_bootstrap, confidence_level)
    ipcc_lower_bounds.append(lower * 3)  # Convert back to confidence scale
    ipcc_upper_bounds.append(upper * 3)

ipcc_lower_bounds = np.array(ipcc_lower_bounds)
ipcc_upper_bounds = np.array(ipcc_upper_bounds)

# Compute bootstrap confidence intervals for IARC (n=1053)
n_samples_iarc = 1053

iarc_lower_bounds = []
iarc_upper_bounds = []

for conf in iarc_confidence_prop:
    lower, mean, upper = bootstrap_ci(conf, n_samples_iarc, n_bootstrap, confidence_level)
    iarc_lower_bounds.append(lower * 3)  # Convert back to confidence scale
    iarc_upper_bounds.append(upper * 3)

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
ax1.plot(budgets, ipcc_confidence, marker='o', linewidth=3, color='navy', label='Mean Confidence')
ax1.axhline(y=1.5, color='gray', linestyle='--', alpha=0.7, label='Ground Truth Confidence')
ax1.set_xscale('symlog', base=2, linthresh=512, linscale=linscale_val)
ax1.set_xlim(0, 24576)
ax1.set_xticks(tick_positions)
ax1.set_xticklabels(tick_labels, rotation=45)
ax1.get_xaxis().set_major_formatter(ScalarFormatter())
ax1.get_xaxis().set_minor_formatter(plt.NullFormatter())
ax1.set_xlabel('Thinking Budget')
ax1.set_ylabel('Average Confidence')
ax1.set_title('IPCC Dataset', pad=15)
ax1.grid(True, which='both', linestyle='--', alpha=0.4)
ax1.legend()

# Plot IARC data (right subplot)
ax2.fill_between(budgets, iarc_lower_bounds, iarc_upper_bounds, alpha=0.3, color='lightcoral', label=f'{int(confidence_level*100)}% Confidence Interval')
ax2.plot(budgets, iarc_confidence, marker='s', linewidth=3, color='red', label='Mean Confidence')
ax2.axhline(y=0.87, color='gray', linestyle='--', alpha=0.7, label='Ground Truth Confidence')
ax2.set_xscale('symlog', base=2, linthresh=512, linscale=linscale_val)
ax2.set_xlim(0, 24576)
ax2.set_xticks(tick_positions)
ax2.set_xticklabels(tick_labels, rotation=45)
ax2.get_xaxis().set_major_formatter(ScalarFormatter())
ax2.get_xaxis().set_minor_formatter(plt.NullFormatter())
ax2.set_xlabel('Thinking Budget')
ax2.set_ylabel('Average Confidence')
ax2.set_title('IARC Dataset', pad=15)
ax2.grid(True, which='both', linestyle='--', alpha=0.4)
ax2.legend()

# Main title
fig.suptitle('Gemini 2.5 Flash – Average Confidence vs Thinking Budget', fontsize=16, y=0.98)

plt.tight_layout()
plt.savefig('figure-confidence.png', dpi=600, bbox_inches="tight")
plt.show()
