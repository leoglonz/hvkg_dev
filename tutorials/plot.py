import matplotlib.pyplot as plt
import numpy as np

# --- 1. Your Data (Example for multiple runs) ---
# You would collect this data by running your experiment script multiple times for each setup.

# Example data for "Standard HV-KG" (3 separate runs)
costs_standard = [
    np.array([229.95148498323778, 234.95148498323778, 239.9514849832378]), # seed=42
    np.array([229.95148498323778, 234.95148498323778, 239.9514849832378]), # seed=0
    np.array([241.3008625420798]), # seed=123456
    np.array([229.95148498323778, 234.95148498323778, 239.9514849832378]),

]

regret_standard = [
    np.array([-0.68129849, -1.42985002, -1.31769778]),
    np.array([-0.64413843, -0.98723451,-1.27792712]),
    np.array([-0.60180077]),
    np.array([-0.59129849, -1.12985002, -1.01769778]),
]

# Example data for "HV-KG with Sensitivity Constraint" (3 separate runs)
costs_sc = [
    np.array([229.95148498323778, 234.95148498323778, 239.9514849832378]), # seed=42
    np.array([195.59475438779637]), # seed=0
    np.array([241.3008625420798]), # seed=123456
    np.array([229.95148498323778, 234.95148498323778, 239.9514849832378]),
]
regret_sc = [
    np.array([-0.55, -1.05, -1.07]),
    np.array([-1.15008121,]),
    np.array([-1.333876]),
    np.array([-0.8, -1.2, -1.34]),

]

# --- 2. Interpolation and Aggregation ---

def process_runs(all_costs, all_regrets, common_cost_axis):
    """
    Interpolates multiple runs onto a common cost axis.
    """
    interpolated_regrets = []
    for costs, regrets in zip(all_costs, all_regrets):
        # np.interp performs linear interpolation.
        # It needs to be told where the new x-points are (common_cost_axis),
        # and what the original x and y points were (costs, regrets).
        interp_regret = np.interp(common_cost_axis, costs, regrets)
        interpolated_regrets.append(interp_regret)
    
    # Now that all runs are on the same axis, we can safely take the mean and std
    mean_regret = np.mean(interpolated_regrets, axis=0)
    std_regret = np.std(interpolated_regrets, axis=0)
    
    return mean_regret, std_regret

# Define the common axis for comparison.
# It should start from the lowest cost observed and end at the highest.
min_cost = min(min(c) for c in costs_standard + costs_sc)
max_cost = max(max(c) for c in costs_standard + costs_sc)
# Create 200 points for a smooth curve.
common_costs = np.linspace(min_cost, max_cost, 200)

# Process both sets of experiments
mean_regret_standard, std_regret_standard = process_runs(costs_standard, regret_standard, common_costs)
mean_regret_sc, std_regret_sc = process_runs(costs_sc, regret_sc, common_costs)


# --- 3. Plotting Setup and Execution (Same as before) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot for Standard HV-KG
ax.plot(common_costs, mean_regret_standard, color='crimson', label='Standard HV-KG')
ax.fill_between(
    common_costs,
    mean_regret_standard - std_regret_standard,
    mean_regret_standard + std_regret_standard,
    color='crimson', alpha=0.2
)

# Plot for HV-KG with Sensitivity Constraint
ax.plot(common_costs, mean_regret_sc, color='darkgreen', label='HV-KG with Sensitivity')
ax.fill_between(
    common_costs,
    mean_regret_sc - std_regret_sc,
    mean_regret_sc + std_regret_sc,
    color='darkgreen', alpha=0.2
)

ax.plot(costs_standard[-1], regret_standard[-1], color='red', linestyle='--')
ax.plot(costs_sc[-1], regret_sc[-1], color='green', linestyle='--')
ax.scatter(costs_standard[-1], regret_standard[-1], color='red')
ax.scatter(costs_sc[-1], regret_sc[-1], color='green')

# Customize and Show the Plot
ax.set_xlabel('Computational Cost', fontsize=14)
ax.set_ylabel('Log Hypervolume Regret', fontsize=14)
ax.set_title('HV-KG Performance Comparison', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower left', frameon=True)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()

fig.savefig('hvkg_comparison.png', dpi=300)
# fig.savefig('hvkg_comparison.pdf')