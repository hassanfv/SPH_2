import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
M = 5.972e24  # Mass of Earth, kg
mu = G * M

# Initial conditions
r0 = 1  # Initial radius, m
vr0 = 0  # Initial radial velocity, m/s
vt0 = np.sqrt((1 - 0.5) * mu / r0)  # Initial tangential velocity, m/s

# Time settings
period = 2 * np.pi * np.sqrt(r0**3 / mu)  # Orbital period, s
t_eval = np.linspace(0, 16 * period, 16 * 24)  # Evaluation times
dt = t_eval[1] - t_eval[0]  # Time step

# Equations of motion
def equations(t, y):
    r, vr, vt = y
    return np.array([vr, vt**2 / r - mu / r**2, -vr * vt / r])

# Initial state vector
y0 = np.array([r0, vr0, vt0])

# Leapfrog integrator
def leapfrog(y, dt, steps):
    r, vr, vt = y
    ys = [y]
    for _ in range(steps):
        vr_mid = vr + equations(0, [r, vr, vt])[1] * dt / 2  # Half step for velocity
        r += vr_mid * dt  # Full step for position
        vr = vr_mid + equations(0, [r, vr, vt])[1] * dt / 2  # Another half step for velocity
        vt += -vr * dt  # Full step for tangential velocity
        ys.append(np.array([r, vr, vt]))
    return np.array(ys).T

# Runge-Kutta integrator
def runge_kutta(t, y):
    return solve_ivp(equations, [t[0], t[-1]], y, t_eval=t, method='RK45').y

# Integrate
y_leapfrog = leapfrog(y0, dt, len(t_eval))
y_runge_kutta = runge_kutta(t_eval, y0)

# Plot
plt.figure()
#plt.plot(y_leapfrog[0], y_leapfrog[1], label='Leapfrog')
plt.plot(y_runge_kutta[0], y_runge_kutta[1], label='Runge-Kutta')
plt.xlabel('Radius, m')
plt.ylabel('Radial velocity, m/s')
plt.legend()
plt.show()

