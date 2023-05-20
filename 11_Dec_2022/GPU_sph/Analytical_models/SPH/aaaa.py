
import numpy as np
import matplotlib.pyplot as plt


# Constants
Msun = 1.989e33  # Solar mass in grams
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
kB = 1.380649e-16  # Boltzmann constant in erg K^-1

# Parameters
M_halo = 1e12 * Msun
f_gas = 0.17
c = 10
T_gas = 1e4  # Gas temperature in Kelvin
N_dm = 1000000  # Number of dark matter particles
N_gas = int(N_dm * f_gas)  # Number of gas particles

# Derived parameters
R_vir = 162.62 * 3.086e21  # Virial radius in cm
a = 28.06 * 3.086e21  # Scale length in cm
M_gas = M_halo * f_gas
M_dm = M_halo * (1 - f_gas)
rho0 = M_halo / (2 * np.pi * a**3)

# Functions
def hernquist_density(r, a, M_halo):
    return M_halo * a / (2 * np.pi * r * (r + a)**3)

def hernquist_potential(r, a, M_halo):
    return -G * M_halo / (r + a)

def hernquist_mass(r, a, M_halo):
    return M_halo * r**2 / (r + a)**2

def generate_positions(N, a, M_halo):
    u = np.random.uniform(size=N)
    r = a * np.sqrt(u) / (1 - np.sqrt(u))
    theta = np.arccos(2 * np.random.uniform(size=N) - 1)
    phi = 2 * np.pi * np.random.uniform(size=N)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack((x, y, z))

def generate_velocities(N, r, a, M_halo):
    v_circ = np.sqrt(G * hernquist_mass(r, a, M_halo) / r)
    v_r = np.random.normal(scale=v_circ / np.sqrt(2), size=(N, 3))
    return v_r



# Generate positions
pos_dm = generate_positions(N_dm, a, M_dm)
pos_gas = generate_positions(N_gas, a, M_gas)

# Compute radial distances
r_dm = np.linalg.norm(pos_dm, axis=1)
r_gas = np.linalg.norm(pos_gas, axis=1)

# Generate velocities
v_dm = generate_velocities(N_dm, r_dm, a, M_dm)
v_gas = generate_velocities(N_gas, r_gas, a, M_gas)


# Gas properties
k_B = 1.380649e-16  # Boltzmann constant in erg/K
m_p = 1.6726219e-24  # Proton mass in grams

T_gas = 1e4  # Gas temperature in K
X = 0.76
Y = 0.24
mu = 1 / (2 * X + 3/4 * Y)

internal_energy_gas = (3/2) * k_B * T_gas / mu / m_p
internal_energy_gas_array = np.full(N_gas, internal_energy_gas)

print(internal_energy_gas_array.shape)


mu = 0.6  # Mean molecular weight for a fully ionized gas with primordial composition
m_p = 1.6726219e-24  # Proton mass in grams
rho_gas = hernquist_density(r_gas, a, M_gas)
m_gas = rho_gas * (4 / 3) * np.pi * r_gas**3 / N_gas
P_gas = rho_gas * kB * T_gas / (mu * m_p)



# Save initial conditions to file
# (You may want to use a specific file format depending on







