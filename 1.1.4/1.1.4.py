import numpy as np
import matplotlib.pyplot as plt

# Define the Lorenz system equations
def lorenz(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

# Euler integration method
def euler_integration(func, state0, dt, steps):
    states = np.zeros((steps, len(state0)))
    states[0] = state0
    for i in range(steps - 1):
        states[i+1] = states[i] + dt * func(states[i])
    return states

# RK4 integration method
def rk4_integration(func, state0, dt, steps):
    states = np.zeros((steps, len(state0)))
    states[0] = state0
    for i in range(steps - 1):
        k1 = func(states[i])
        k2 = func(states[i] + dt/2 * k1)
        k3 = func(states[i] + dt/2 * k2)
        k4 = func(states[i] + dt * k3)
        states[i+1] = states[i] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return states

# Simulation parameters
state0 = np.array([1.0, 1.0, 1.0])
tmax = 40.0
dt_values = [0.001, 0.01, 0.1]  # different time step sizes
# dt_values = np.round(np.logspace(-3, -1, num=1000), 3)
# dt_values = np.arange(0.001, 0.1, 0.05)  # dt from 0.001 to 0.1 in steps of 0.01
methods = {"Euler": euler_integration, "RK4": rk4_integration}

# Run simulations for each method and each dt
results = {}
for method_name, method_func in methods.items():
    results[method_name] = {}
    for dt in dt_values:
        steps = int(tmax / dt)
        t = np.linspace(0, tmax, steps)
        states = method_func(lorenz, state0, dt, steps)
        results[method_name][dt] = (t, states)

# -------------------------
# Figure 1: Phase Portraits
# -------------------------
fig1, axs1 = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
fig1.suptitle("Lorenz Attractor Phase Portraits: Comparison of Numerical Methods", fontsize=20, weight='bold')

for i, (method_name, method_results) in enumerate(results.items()):
    for j, dt in enumerate(dt_values):
        t, states = method_results[dt]
        ax = axs1[i, j]
        ax.plot(states[:, 0], states[:, 2], lw=0.5, color='black')
        ax.set_title(f"{method_name} (dt = {dt})", fontsize=12)
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("z", fontsize=10)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
fig1.savefig("1.1.4\images\phase_portraits.png", dpi=300)
plt.show()

# -----------------------------------------------------------
# Figure 2: Quantitative Comparison of Simulation Accuracy
# -----------------------------------------------------------
# Here we compare each simulation (for dt > 0.001) to the reference simulation (dt = 0.001)
# by computing the average Euclidean norm difference between the state trajectories.
errors = {"Euler": {}, "RK4": {}}
ref_dt = 0.001  # reference time step
for method_name, method_results in results.items():
    t_ref, states_ref = method_results[ref_dt]
    for dt in dt_values:
        if dt == ref_dt:
            errors[method_name][dt] = 0.0
        else:
            t_sim, states_sim = method_results[dt]
            factor = int(dt / ref_dt)  # since dt values are multiples of 0.001
            # Sample the reference simulation to align with the coarser time grid
            ref_states_sampled = states_ref[::factor][:len(states_sim)]
            # Compute the Euclidean error at each time step
            error_array = np.linalg.norm(states_sim - ref_states_sampled, axis=1)
            avg_error = np.mean(error_array)
            errors[method_name][dt] = avg_error

# Plot the average error vs. time step for each method
fig3, ax3 = plt.subplots(figsize=(10, 6))
for method_name, error_dict in errors.items():
    dt_list = sorted(error_dict.keys())
    error_list = [error_dict[dt] for dt in dt_list]
    ax3.plot(dt_list, error_list, marker='o', label=method_name)
ax3.set_title("Average Euclidean Error vs. Time Step (Reference: dt = 0.001)", fontsize=14)
ax3.set_xlabel("Time Step (dt)", fontsize=12)
ax3.set_ylabel("Average Euclidean Error", fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(fontsize=10)
fig3.savefig("1.1.4\images\error_comparison.png", dpi=300)
plt.show()
