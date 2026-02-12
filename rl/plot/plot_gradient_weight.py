import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Set parameters
sigma_gipo = 0.5
eps_ppo = 0.2
tau_sapo_pos = 1.0
tau_sapo_neg = 2.0

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
x_lin = x[mask]
y_gipo_lin = y_gipo[mask]
y_ppo_pos_lin = y_ppo_pos[mask]
y_ppo_neg_lin = y_ppo_neg[mask]
y_sapo_pos_lin = y_sapo_pos[mask]
y_sapo_neg_lin = y_sapo_neg[mask]

# --- Combined Plot: 3x2 layout (Linear Scale on row 1, blank row 2, Log Scale on row 3) ---
fig = plt.figure(figsize=(20, 18))
gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1, 0.1, 1], hspace=0.2)

# Row 1: Linear Scale (Ratio)
# Top Left: A > 0, Linear Scale
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(r_lin, y_gipo_lin, '-', color='#ff7f0e', linewidth=3, label=f'GIPO (σ={sigma_gipo})')
ax1.plot(r_lin, y_ppo_pos_lin, '-', color='#1f77b4', linewidth=3, label=f'PPO (ε={eps_ppo})')
ax1.plot(r_lin, y_sapo_pos_lin, '-', color='#2ca02c', linewidth=3, label=f'SAPO ($τ_{{pos}}$={tau_sapo_pos}, $τ_{{neg}}$={tau_sapo_neg})')
ax1.set_xlabel('ρ ', fontsize=35)
ax1.set_ylabel('Gradient Weight', fontsize=35)
ax1.set_title('A > 0', fontsize=37, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=28)
ax1.grid(True, alpha=0.3)
ax1.axvline(x=1.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)

# Top Right: A < 0, Linear Scale
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(r_lin, y_gipo_lin, '-', color='#ff7f0e', linewidth=3, label=f'GIPO (σ={sigma_gipo})')
ax2.plot(r_lin, y_ppo_neg_lin, '-', color='#1f77b4', linewidth=3, label=f'PPO (ε={eps_ppo})')
ax2.plot(r_lin, y_sapo_neg_lin, '-', color='#2ca02c', linewidth=3, label=f'SAPO ($τ_{{pos}}$={tau_sapo_pos}, $τ_{{neg}}$={tau_sapo_neg})')
ax2.set_xlabel('ρ ', fontsize=35)
ax2.set_title('A < 0', fontsize=37, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=28)
ax2.grid(True, alpha=0.3)
ax2.axvline(x=1.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)

# Row 2: Blank (spacing row - no subplot created)

# Row 3: Log Scale
# Bottom Left: A > 0, Log Scale
ax3 = fig.add_subplot(gs[2, 0])
ax3.plot(x, y_gipo, '-', color='#ff7f0e', linewidth=3, label=f'GIPO (σ={sigma_gipo})')
ax3.plot(x, y_ppo_pos, '-', color='#1f77b4', linewidth=3, label=f'PPO (ε={eps_ppo})')
ax3.plot(x, y_sapo_pos, '-', color='#2ca02c', linewidth=3, label=f'SAPO ($τ_{{pos}}$={tau_sapo_pos}, $τ_{{neg}}$={tau_sapo_neg})')
ax3.set_xlabel('log(ρ)', fontsize=35)
ax3.set_ylabel('Gradient Weight', fontsize=35)
ax3.set_title('A > 0', fontsize=37, fontweight='bold')
ax3.tick_params(axis='both', which='major', labelsize=28)
ax3.grid(True, alpha=0.3)
ax3.axvline(x=0.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)
ax3.set_xlim([-1.5, 1.5])

# Bottom Right: A < 0, Log Scale
ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(x, y_gipo, '-', color='#ff7f0e', linewidth=3, label=f'GIPO (σ={sigma_gipo})')
ax4.plot(x, y_ppo_neg, '-', color='#1f77b4', linewidth=3, label=f'PPO (ε={eps_ppo})')
ax4.plot(x, y_sapo_neg, '-', color='#2ca02c', linewidth=3, label=f'SAPO ($τ_{{pos}}$={tau_sapo_pos}, $τ_{{neg}}$={tau_sapo_neg})')
ax4.set_xlabel('log(ρ)', fontsize=35)
ax4.set_title('A < 0', fontsize=37, fontweight='bold')
ax4.tick_params(axis='both', which='major', labelsize=28)
ax4.grid(True, alpha=0.3)
ax4.axvline(x=0.0, color='k', linestyle='dashed', alpha=1, linewidth=2.5)
ax4.set_xlim([-1.5, 1.5])

# Add shared legend on top of the figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=28, 
           bbox_to_anchor=(0.5, 1.02), frameon=True, fancybox=True, shadow=False)

# Adjust layout to make room for the legend
fig.tight_layout(rect=[0, 0, 1, 0.94])

# Save the combined plot
save_path_png = 'rollouts/tmp/clip_comparison_combined.png'
save_path_pdf = 'rollouts/tmp/clip_comparison_combined.pdf'
fig.savefig(save_path_png, dpi=300, bbox_inches='tight')
fig.savefig(save_path_pdf, bbox_inches='tight')
print(f'Combined plot saved to: {save_path_png}')
print(f'Combined plot PDF saved to: {save_path_pdf}')
plt.close(fig)
