import numpy as np
import optuna
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

# Lorenz system parameters
sigma = 10.0
rho = 28.0
beta = 8/3

def lorenz_deriv(state):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def rk4_step(state, dt):
    k1 = lorenz_deriv(state)
    k2 = lorenz_deriv(state + 0.5 * dt * k1)
    k3 = lorenz_deriv(state + 0.5 * dt * k2)
    k4 = lorenz_deriv(state + dt * k3)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def simulate_lorenz(initial_state, dt, T):
    n_steps = int(T/dt)
    solution = np.empty((n_steps + 1, 3))
    solution[0] = initial_state
    state = initial_state.copy()
    for i in range(n_steps):
        state = rk4_step(state, dt)
        solution[i + 1] = state
    return solution

def compute_error(simulated, reference):
    # Mean Euclidean error over all time steps
    error = np.mean(np.linalg.norm(simulated - reference, axis=1))
    return error

# Simulation parameters
T = 40.0  # total simulation time
initial_state = np.array([1.0, 1.0, 1.0])
dt_ref = 0.001  # reference time step

# Generate a high-accuracy reference solution
print("Generating reference solution...")
reference_solution = simulate_lorenz(initial_state, dt_ref, T)
ref_times = np.linspace(0, T, reference_solution.shape[0])

def objective_composite(trial):
    # Suggest a candidate time step in the range [0.001, 0.1]
    dt = trial.suggest_float("dt", 0.001, 0.1, log=True)
    
    start_time = time.time()
    simulated_solution = simulate_lorenz(initial_state, dt, T)
    # runtime = time.time() - start_time  # measured runtime
    
    sim_times = np.linspace(0, T, simulated_solution.shape[0])
    # Interpolate the reference solution to match simulated times
    reference_interp = np.empty(simulated_solution.shape)
    for i in range(3):
        reference_interp[:, i] = np.interp(sim_times, ref_times, reference_solution[:, i])
    
    error = compute_error(simulated_solution, reference_interp)
    
    # Composite objective: error + alpha * (number of steps)
    # Using number of steps as a proxy for computational cost: n_steps = T / dt
    alpha = 0.001  # weight for the cost term
    cost = T / dt
    composite_objective = error + alpha * cost
    return composite_objective

# Optimize using the composite objective approach
study_composite = optuna.create_study(direction="minimize")
study_composite.optimize(objective_composite, n_trials=100)

best_dt_composite = study_composite.best_params["dt"]
print("Composite Approach - Best time step (dt):", best_dt_composite)
print("Composite Approach - Best composite objective:", study_composite.best_value)

# Simulate best solution using best_dt_composite
best_solution = simulate_lorenz(initial_state, best_dt_composite, T)
sim_times = np.linspace(0, T, best_solution.shape[0])

# Interpolate reference solution to match best solution times
reference_interp = np.empty(best_solution.shape)
for i in range(3):
    reference_interp[:, i] = np.interp(sim_times, ref_times, reference_solution[:, i])

# Compute mean Euclidean error between best solution and reference
mean_error = compute_error(best_solution, reference_interp)
print("Mean Euclidean error between best solution and reference:", mean_error)

# Plot the reference trajectory and best solution trajectory
fig = plt.figure(figsize=(12, 6))

# Plot the reference trajectory in the left subplot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(reference_solution[:, 0], reference_solution[:, 1], reference_solution[:, 2],
         lw=0.5, label="Reference Trajectory")
ax1.set_title("Reference Trajectory")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Plot the best solution trajectory in the right subplot
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(best_solution[:, 0], best_solution[:, 1], best_solution[:, 2],
         lw=0.5, color='r', label="Best Solution Trajectory")
ax2.set_title("Best Solution Trajectory")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()

plt.suptitle(f"Mean Euclidean Error: {mean_error:.4f}")
plt.tight_layout()
plt.show()
