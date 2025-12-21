import os
import torch
import numpy as np
import matplotlib.pyplot as plt

folder = "results/dimension_test"  # change if needed

# CORRECT way to load .pt files saved with torch.save()
stats       = torch.load(os.path.join(folder, 'log_z.pt'))
w2_stats    = torch.load(os.path.join(folder, 'w2.pt'))
method_names = np.load(os.path.join(folder, 'method_names.npy'))

# If they are tensors, convert to numpy once and for all
if isinstance(stats, torch.Tensor):
    stats = stats.cpu().numpy()
if isinstance(w2_stats, torch.Tensor):
    w2_stats = w2_stats.cpu().numpy()

dimensions = np.arange(1, 8)  # D = 1..7

plt.rcParams.update({'font.size': 14, 'text.usetex': False})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

markers = ['o', 's', '^', 'D', 'v', '*', 'p']
colors  = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

for i, method in enumerate(method_names):
    method = str(method)
    if method[-2:] != 'MC' and method != 'SLIPS':
        continue
    
    label = method
    if method == 'ZOD-MC': label = 'ZOD-MC'
    if method == 'RDMC':   label = 'RDMC'
    if method == 'RSDMC':  label = 'RSDMC'
    if method == 'SLIPS':  label = 'SLIPS'

    ax1.plot(dimensions, np.abs(stats[i] - stats[0]),
             label=label, marker=markers[i%len(markers)], markersize=9,
             linewidth=2.5, color=colors[i])
    ax2.plot(dimensions, w2_stats[i],
             label=label, marker=markers[i%len(markers)], markersize=9,
             linewidth=2.5, color=colors[i])

ax1.set_xlabel('Dimension')
ax1.set_ylabel('Error in E[f(x)]')
ax1.set_ylim(0, 800)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_xlabel('Dimension')
ax2.set_ylabel('Wasserstein-2 Distance')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
output_path = os.path.join(folder, 'dimension_mmd_results.pdf')
fig.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Success! PDF saved to: {output_path}")
plt.close()