import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Plot gradient weight comparison curves.')
    parser.add_argument(
        '--line-width',
        type=float,
        default=3.0,
        help='Line width for plotted curves (default: 3.0).',
    )
    parser.add_argument(
        '--x-axis',
        choices=['ratio', 'log', 'both'],
        default='both',
        help='Choose x-axis type to plot: ratio, log, or both (default: both).',
    )
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['gipo', 'ppo', 'sapo', 'all'],
        default=['all'],
        help='Methods to plot. Example: --methods ppo or --methods gipo ppo. Use all for all methods.',
    )
    return parser.parse_args()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Set parameters
sigma_gipo = 0.5
eps_ppo = 0.2
tau_sapo_pos = 1.0
tau_sapo_neg = 2.0
args = parse_args()
if 'all' in args.methods:
    args.methods = ['gipo', 'ppo', 'sapo']
else:
    # Remove duplicates while preserving order.
    args.methods = list(dict.fromkeys(args.methods))

# Generate r values
# For log plot, we want a wide range in log space
r_log_range = np.logspace(-2, 2, 10000) # 0.01 to 100
r = r_log_range
x = np.log(r)

# --- 1. GIPO ---
y_gipo = np.exp(-0.5 * (x / sigma_gipo) ** 2)

# --- 2. PPO ---
# Case A > 0:
y_ppo_pos = np.where(r <= 1 + eps_ppo, 1.0, 0.0)
# Case A < 0:
y_ppo_neg = np.where(r >= 1 - eps_ppo, 1.0, 0.0)

# --- 3. SAPO ---
u_pos = tau_sapo_pos * (r - 1.0)
sig_pos = sigmoid(u_pos)
y_sapo_pos = 4.0 * sig_pos * (1.0 - sig_pos)

u_neg = tau_sapo_neg * (r - 1.0)
sig_neg = sigmoid(u_neg)
y_sapo_neg = 4.0 * sig_neg * (1.0 - sig_neg)

# Ensure directory exists
os.makedirs('rollouts/tmp', exist_ok=True)

# Filter r for better visualization (e.g. 0 to 3)
mask = (r <= 3.0)
r_lin = r[mask]
y_gipo_lin = y_gipo[mask]
y_ppo_pos_lin = y_ppo_pos[mask]
y_ppo_neg_lin = y_ppo_neg[mask]
y_sapo_pos_lin = y_sapo_pos[mask]
y_sapo_neg_lin = y_sapo_neg[mask]

style_map = {
    'gipo': {'color': '#ff7f0e', 'label': f'GIPO (σ={sigma_gipo})'},
    'ppo': {'color': '#1f77b4', 'label': f'PPO (ε={eps_ppo})'},
    'sapo': {'color': '#2ca02c', 'label': f'SAPO ($τ_{{pos}}$={tau_sapo_pos}, $τ_{{neg}}$={tau_sapo_neg})'},
}


def plot_selected_methods(ax, x_values, y_pos_map, y_neg_map, is_positive):
    for method in args.methods:
        y_values = y_pos_map[method] if is_positive else y_neg_map[method]
        ax.plot(
            x_values,
            y_values,
            '-',
            color=style_map[method]['color'],
            linewidth=args.line_width,
            label=style_map[method]['label'],
        )


if args.x_axis == 'both':
    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 0.1, 1], hspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])  # ratio, A > 0
    ax2 = fig.add_subplot(gs[0, 1])  # ratio, A < 0
    ax3 = fig.add_subplot(gs[2, 0])  # log, A > 0
    ax4 = fig.add_subplot(gs[2, 1])  # log, A < 0
    axes = [ax1, ax2, ax3, ax4]
else:
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.2)
    ax1 = fig.add_subplot(gs[0, 0])  # selected x-axis, A > 0
    ax2 = fig.add_subplot(gs[0, 1])  # selected x-axis, A < 0
    axes = [ax1, ax2]

y_pos_lin_map = {'gipo': y_gipo_lin, 'ppo': y_ppo_pos_lin, 'sapo': y_sapo_pos_lin}
y_neg_lin_map = {'gipo': y_gipo_lin, 'ppo': y_ppo_neg_lin, 'sapo': y_sapo_neg_lin}
y_pos_log_map = {'gipo': y_gipo, 'ppo': y_ppo_pos, 'sapo': y_sapo_pos}
y_neg_log_map = {'gipo': y_gipo, 'ppo': y_ppo_neg, 'sapo': y_sapo_neg}

if args.x_axis in ['ratio', 'both']:
    plot_selected_methods(ax1, r_lin, y_pos_lin_map, y_neg_lin_map, is_positive=True)
    plot_selected_methods(ax2, r_lin, y_pos_lin_map, y_neg_lin_map, is_positive=False)
    ax1.set_xlabel('ρ', fontsize=35)
    ax2.set_xlabel('ρ', fontsize=35)
    ax1.axvline(x=1.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)
    ax2.axvline(x=1.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)

if args.x_axis == 'both':
    plot_selected_methods(ax3, x, y_pos_log_map, y_neg_log_map, is_positive=True)
    plot_selected_methods(ax4, x, y_pos_log_map, y_neg_log_map, is_positive=False)
    ax3.set_xlabel('log(ρ)', fontsize=35)
    ax4.set_xlabel('log(ρ)', fontsize=35)
    ax3.axvline(x=0.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)
    ax4.axvline(x=0.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)
    ax3.set_xlim([-1.5, 1.5])
    ax4.set_xlim([-1.5, 1.5])
elif args.x_axis == 'log':
    plot_selected_methods(ax1, x, y_pos_log_map, y_neg_log_map, is_positive=True)
    plot_selected_methods(ax2, x, y_pos_log_map, y_neg_log_map, is_positive=False)
    ax1.set_xlabel('log(ρ)', fontsize=35)
    ax2.set_xlabel('log(ρ)', fontsize=35)
    ax1.axvline(x=0.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)
    ax2.axvline(x=0.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)
    ax1.set_xlim([-1.5, 1.5])
    ax2.set_xlim([-1.5, 1.5])

for idx, ax in enumerate(axes):
    if idx in [0, 2]:
        ax.set_ylabel('Gradient Weight', fontsize=35)
    ax.set_title('A > 0' if idx in [0, 2] else 'A < 0', fontsize=37)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, alpha=0.3)

# Add shared legend on top of the figure
handles, labels = axes[0].get_legend_handles_labels()
if args.x_axis == 'both':
    legend_y = 1.02
    layout_top = 0.94
else:
    # Single-row layout needs more top space to avoid title/legend overlap.
    legend_y = 1.10
    layout_top = 0.86
fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=min(3, len(args.methods)),
    fontsize=28,
    bbox_to_anchor=(0.5, legend_y),
    frameon=True,
    fancybox=True,
    shadow=False,
)

# Adjust layout to make room for the legend
fig.subplots_adjust(top=layout_top)

# Save the combined plot
if args.x_axis == 'both' and args.methods == ['gipo', 'ppo', 'sapo']:
    save_path_png = 'rollouts/tmp/clip_comparison_combined.png'
    save_path_pdf = 'rollouts/tmp/clip_comparison_combined.pdf'
else:
    method_suffix = '_'.join(args.methods)
    save_path_png = f'rollouts/tmp/clip_comparison_{args.x_axis}_{method_suffix}.png'
    save_path_pdf = f'rollouts/tmp/clip_comparison_{args.x_axis}_{method_suffix}.pdf'
fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
fig.savefig(save_path_pdf, bbox_inches='tight')
print(f'Combined plot saved to: {save_path_png}')
print(f'Combined plot PDF saved to: {save_path_pdf}')
plt.close(fig)
