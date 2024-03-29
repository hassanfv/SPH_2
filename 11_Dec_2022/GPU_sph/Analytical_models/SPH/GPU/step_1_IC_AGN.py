
# Here the Gas and DM have similar number of particles!

import numpy as np
import matplotlib.pyplot as plt
import time
from libsx import *
from shear import *
import pandas as pd
import pickle


# Constants
Msun = 1.989e33  # Solar mass in grams
G = 6.67430e-8  # Gravitational constant in cm^3 g^-1 s^-2
kB = 1.380649e-16  # Boltzmann constant in erg K^-1
clight = 29979245800. # cm/s

# Parameters
M_halo = 1e12 * Msun
f_gas = 0.17
c = 10
k_B = 1.380649e-16  # Boltzmann constant in erg/K
m_p = 1.6726219e-24  # Proton mass in grams

T_gas = 1e4  # Gas temperature in K
X = 0.76
Y = 0.24
mu = 1 / (2 * X + 3/4 * Y)

N = 100000

N_dm = N_gas = N

# Derived parameters
R_vir = 162.62 * 3.086e21  # Virial radius in cm
a = 28.06 * 3.086e21  # Scale length in cm

m_dm = M_halo / N / (1.0 + f_gas) # The mass of a single DM particle.
m_gas= f_gas * m_dm # The mass of a single gas particle.

M_gas = N_gas * m_gas
M_dm = N_dm * m_dm

rho0 = M_halo / (2 * np.pi * a**3)

# Functions
def hernquist_density(r, a, M_halo):
    return M_halo * a / (2 * np.pi * r * (r + a)**3)

def hernquist_potential(r, a, M_halo):
    return -G * M_halo / (r + a)

def hernquist_mass(r, a, M_halo):
    return M_halo * r**2 / (r + a)**2

def generate_positions(N, a):
    max_radius = 100 * a
    u = np.random.uniform(size=N)
    r = a * np.sqrt(u) / (1 - np.sqrt(u))
    
    # Resample radii that are larger than the maximum radius
    while np.any(r > max_radius):
        mask = r > max_radius
        u_resample = np.random.uniform(size=np.sum(mask))
        r_resample = a * np.sqrt(u_resample) / (1 - np.sqrt(u_resample))
        r[mask] = r_resample
        
    theta = np.arccos(2 * np.random.uniform(size=N) - 1)
    phi = 2 * np.pi * np.random.uniform(size=N)
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.column_stack((x, y, z))



def generate_velocities(N, r, a, M_halo):
    vc = np.sqrt(G * M_halo * r / (r + a)**2)
    
    theta = np.arccos(2 * np.random.uniform(size=N) - 1)
    phi = 2 * np.pi * np.random.uniform(size=N)
    
    vx = vc * np.sin(theta) * np.cos(phi)
    vy = vc * np.sin(theta) * np.sin(phi)
    vz = vc * np.cos(theta)
    
    return np.column_stack((vx, vy, vz))



# Generate positions
pos_dm = generate_positions(N_dm, a)
pos_gas = generate_positions(N_gas, a)

# Compute radial distances - needed for the velocity
r_dm = np.linalg.norm(pos_dm, axis=1)
r_gas = np.linalg.norm(pos_gas, axis=1)

# Generate velocities
v_dm = generate_velocities(N_dm, r_dm, a, M_dm)
v_gas = generate_velocities(N_gas, r_gas, a, M_gas)


# Gas properties
u = (3/2) * k_B * T_gas / mu / m_p
u = np.full(N_gas, u)


epsilon = 0.5 * 3.086e21 # 0.5 kpc to cm
epsilon = np.full(N_dm, epsilon) # for the gas particles we will equal it to the smoothing length h!


# Setting the units of the simulation
grav_const_in_cgs = 6.67430e-8
unitMass_in_g = M_halo
unitLength_in_cm = R_vir
unitTime_in_s = (unitLength_in_cm**3/grav_const_in_cgs/unitMass_in_g)**0.5
unitVelocity_in_cm_per_s = unitLength_in_cm / unitTime_in_s
Unit_u_in_cgs = grav_const_in_cgs * unitMass_in_g / unitLength_in_cm
UnitDensity_in_cgs = unitMass_in_g / unitLength_in_cm**3

print(f'Unit_time_in_s = {unitTime_in_s:.3E} seconds')
print(f'Unit_time in kyrs = {round(unitTime_in_s/3600./24./365.25/1000., 2)} kyrs')
print(f'Unit_time in Myrs = {round(unitTime_in_s/3600./24./365.25/1e6, 4)} Myrs')
print(f'unitVelocity_in_cm_per_s = {round(unitVelocity_in_cm_per_s, 2)} cm/s')
print(f'Unit_u_in_cgs = {Unit_u_in_cgs:.4E}')
print(f'UnitDensity_in_cgs = {UnitDensity_in_cgs:.3E}')

# Converting to code unit
pos_dm = pos_dm / unitLength_in_cm
pos_gas = pos_gas / unitLength_in_cm

v_dm = v_dm / unitVelocity_in_cm_per_s
v_gas = v_gas / unitVelocity_in_cm_per_s

u = u / Unit_u_in_cgs

epsilon = epsilon / unitLength_in_cm

print('epsilon = ', epsilon)

print(epsilon.shape)


mDM = np.full(N, M_dm / N) / unitMass_in_g
mGas = np.full(N, M_gas / N) / unitMass_in_g

m = np.hstack((mGas, mDM))

M_BH = 3e8 * Msun
mBH = M_BH / unitMass_in_g

# Eddington luminosity calculation
kappa = 0.34  # cm^2/g
L_Edd = 4 * np.pi * G * M_BH * clight / kappa # erg/s
L_Edd = L_Edd / (Unit_u_in_cgs * unitMass_in_g / unitTime_in_s) # L_Edd in code unit. Note that we multiply u by mass because u is in per unit mass!!
								 # So L_Edd is now in energy per unit mass per unit time.

print(f'LEdd = {L_Edd:.2E}')
print(u)
print()
print('NGas = ', pos_gas.shape[0])
print('NDM = ', pos_dm.shape[0])

#dictx = {'r_gas': pos_gas, 'r_dm': pos_dm, 'v_gas': v_gas, 'v_dm': v_dm, 'u': u, 'm': m, 'mBH': mBH, 'eps_dm': epsilon, 'h': h, 'L_Edd': L_Edd}

xG = pos_gas[:, 0]
yG = pos_gas[:, 1]
zG = pos_gas[:, 2]

vxG = v_gas[:, 0]
vyG = v_gas[:, 1]
vzG = v_gas[:, 2]

xD = pos_dm[:, 0]
yD = pos_dm[:, 1]
zD = pos_dm[:, 2]

vxD = v_dm[:, 0]
vyD = v_dm[:, 1]
vzD = v_dm[:, 2]


x = np.hstack((xG, xD))
y = np.hstack((yG, yD))
z = np.hstack((zG, zD))

vx = np.hstack((vxG, vxD))
vy = np.hstack((vyG, vyD))
vz = np.hstack((vzG, vzD))


x = np.round(x, 5)
y = np.round(y, 5)
z = np.round(z, 5)

vx = np.round(vx, 5)
vy = np.round(vy, 5)
vz = np.round(vz, 5)

x = x.astype(np.float32)
y = y.astype(np.float32)
z = z.astype(np.float32)

vx = vx.astype(np.float32)
vy = vy.astype(np.float32)
vz = vz.astype(np.float32)

m = m.astype(np.float32)
epsilon = epsilon.astype(np.float32)
u = u.astype(np.float32)


N_tot = N_dm + N_gas


dictx = {'x': x, 'y': y, 'z': z, 'vx': vx, 'vy': vy, 'vz': vz, 'm': m, 'epsilon': epsilon, 'u': u}

with open('tmp_IC.pkl', 'wb') as f:
	pickle.dump(dictx, f)

G = 1.0

# Saving the parameters and constants!
with open('params.txt', 'w') as f:
    # Write each variable on its own line
    f.write(f'{N_tot}\n')
    f.write(f'{N_dm}\n')
    f.write(f'{N_gas}\n')
    f.write(f'{G}\n')


print(x[-10:])

xy = 0.25

print('x shape = ', x.shape)

#plt.hist(x, bins = np.arange(-1, 1, 0.1))
#plt.show()

plt.scatter(x[:N_gas], y[:N_gas], s = 0.01, color = 'k')
plt.scatter(x[N_gas:], y[N_gas:], s = 0.01, color = 'cyan')

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

plt.show()




