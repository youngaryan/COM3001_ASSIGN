import numpy as np
import matplotlib.pyplot as plt

# Define the Lorenz system equations
def lorenz(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

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

# Parameters
initial_conditions = [
    np.array([0.1, 2.0, 1.0]),
    np.array([0.01, 2.0, 1.0])  # slightly perturbed
]
labels = ["Initial Condition 1 (0.1, 2.0, 1.0)", "Initial Condition 2 (0.01, 2.0, 1.0)"]
tmax = 40.0
dt = 0.0073
steps = int(tmax / dt)
t = np.linspace(0, tmax, steps)

# Run simulation
trajectories = [rk4_integration(lorenz, ic, dt, steps) for ic in initial_conditions]

# Plot 2D phase space (x vs z) - separate subplots
# fig2d, axs2d = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
# fig2d.suptitle("Lorenz Attractor: x vs z for Two Initial Conditions", fontsize=16, weight='bold')
# for i, ax in enumerate(axs2d):
#     ax.plot(trajectories[i][:, 0], trajectories[i][:, 2], color='black', lw=0.8)
#     ax.set_title(labels[i], fontsize=12)
#     ax.set_xlabel("x")
#     ax.set_ylabel("z")
#     ax.grid(True, linestyle='--', alpha=0.5)
# fig2d.savefig("1.1.4/images/two_initial_conditions_xz_separate.png", dpi=300)
plt.show()

# Plot 3D phase space - separate subplots
fig3d = plt.figure(figsize=(14, 6), constrained_layout=True)
fig3d.suptitle("Lorenz Attractor (3D) for Two Initial Conditions", fontsize=16, weight='bold')

for i in range(2):
    ax = fig3d.add_subplot(1, 2, i+1, projection='3d')
    states = trajectories[i]
    ax.plot(states[:, 0], states[:, 1], states[:, 2], color='black', lw=0.8)
    ax.set_title(labels[i], fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
fig3d.savefig("1.1.4/images/two_initial_conditions_3d_separate.png", dpi=300)
plt.show()
