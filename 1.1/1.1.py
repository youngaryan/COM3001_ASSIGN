import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the Lorentz system
def lorentz_system(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters for the Lorentz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial state
initial_state = [0.0,0.5,0.55]

# Time points
t = np.linspace(0, 50, 10000)

# Solve the system of ODEs
solution = odeint(lorentz_system, initial_state, t, args=(sigma, rho, beta))

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 3D plot of the Lorentz attractor
ax = axes[0]
ax = fig.add_subplot(121, projection='3d')
ax.plot(solution[:, 0], solution[:, 1], solution[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorentz Attractor')

# Phase space plot (x vs y)
axes[1].plot(solution[:, 0], solution[:, 1])
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].set_title('Phase Space (X vs Y)')

plt.tight_layout()
plt.show()
