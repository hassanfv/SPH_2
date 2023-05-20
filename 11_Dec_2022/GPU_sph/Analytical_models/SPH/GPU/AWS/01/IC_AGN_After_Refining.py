
import numpy as np
import matplotlib.pyplot as plt
import time
from libsx import *
#from shear import *
import pandas as pd
import struct


def read_arrays_from_binary(filename):
    with open(filename, 'rb') as file:
        # Read N and NG from the file
        N = struct.unpack('i', file.read(4))[0]
        NG = struct.unpack('i', file.read(4))[0]

        # Read the arrays from the file
        x = struct.unpack(f'{N}f', file.read(4 * N))
        y = struct.unpack(f'{N}f', file.read(4 * N))
        z = struct.unpack(f'{N}f', file.read(4 * N))
        vx = struct.unpack(f'{N}f', file.read(4 * N))
        vy = struct.unpack(f'{N}f', file.read(4 * N))
        vz = struct.unpack(f'{N}f', file.read(4 * N))
        rho = struct.unpack(f'{NG}f', file.read(4 * NG))
        h = struct.unpack(f'{NG}f', file.read(4 * NG))
        u = struct.unpack(f'{NG}f', file.read(4 * NG))

    return x, y, z, vx, vy, vz, rho, h, u


filename = './Outputs/G-0.000000.bin'
x, y, z, vx, vy, vz, rho, h, u = read_arrays_from_binary(filename)

x = np.array(x)
y = np.array(y)
z = np.array(z)

#x_com = np.mean(x)
#y_com = np.mean(y)
#z_com = np.mean(z)

#x = x - x_com
#y = y - y_com
#z = z - z_com

vx = np.array(vx)
vy = np.array(vy)
vz = np.array(vz)
rho = np.array(rho)
h = np.array(h)
u = np.array(u)

N = len(x)
N_gas = len(u)
N_dm = N - N_gas

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

# Derived parameters
R_vir = 162.62 * 3.086e21  # Virial radius in cm
a = 28.06 * 3.086e21  # Scale length in cm
M_gas = M_halo * f_gas
M_dm = M_halo * (1 - f_gas)
rho0 = M_halo / (2 * np.pi * a**3)



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


epsilon = epsilon / unitLength_in_cm

N = N_gas + N_dm
M = M_gas + M_dm

m = np.full(N, M / N) / unitMass_in_g

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
print('NGas = ', N_gas)
print('N_dm = ', N_dm)
print('N = ', N)


epsilon = np.hstack((h, epsilon))

x = np.hstack((x, np.array(0)))
y = np.hstack((y, np.array(0)))
z = np.hstack((z, np.array(0)))

vx = np.hstack((vx, np.array(0)))
vy = np.hstack((vy, np.array(0)))
vz = np.hstack((vz, np.array(0)))

m = np.hstack((m, np.array([mBH])))

epsilon = np.hstack((epsilon, np.array(epsilon[-1])))

N = len(x)  # Black hole is counted in this N!

r = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
rBH = np.array([[0, 0, 0]])
r = np.vstack((r, rBH))
hBH = smoothing_BH(rBH, r[:N_gas])

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
h = h.astype(np.float32)
epsilon = epsilon.astype(np.float32)

k_B = 1.380649e-16  # Boltzmann constant in erg/K
m_p = 1.6726219e-24  # Proton mass in grams
T_gas = 1e4  # Gas temperature in K
X = 0.76
Y = 0.24
mu = 1 / (2 * X + 3/4 * Y)
u = (3/2) * k_B * T_gas / mu / m_p / Unit_u_in_cgs + np.zeros_like(u)

u = u.astype(np.float32)

print(u)

print('hBH = ', hBH)
#-------------------------------------

# Save the arrays to a binary file
num = str(int(np.floor(N/1000)))
with open('IC_AGN_Evolved_0.00Gyr_' + num + 'k.bin', "wb") as file:
    file.write(x.tobytes())
    file.write(y.tobytes())
    file.write(z.tobytes())
    
    file.write(vx.tobytes())
    file.write(vy.tobytes())
    file.write(vz.tobytes())
    
    file.write(m.tobytes())
    file.write(h.tobytes())
    file.write(epsilon.tobytes())
    file.write(u.tobytes())
   


G = 1.0
eps_AGN = 0.05

# Saving the parameters and constants!
with open('params.txt', 'w') as f:
    # Write each variable on its own line
    f.write(f'{N}\n')
    f.write(f'{N_dm}\n')
    f.write(f'{N_gas}\n')
    f.write(f'{G}\n')
    f.write(f'{hBH}\n')
    f.write(f'{eps_AGN}\n')
    f.write(f'{L_Edd}\n')


xy = 0.25

print('x shape = ', x.shape)

plt.hist(x, bins = np.arange(-1, 1, 0.1))
plt.show()

plt.scatter(x[:N_gas], y[:N_gas], s = 0.01, color = 'k')
plt.scatter(x[N_gas:], y[N_gas:], s = 0.01, color = 'cyan')

plt.xlim(-xy, xy)
plt.ylim(-xy, xy)

plt.show()




