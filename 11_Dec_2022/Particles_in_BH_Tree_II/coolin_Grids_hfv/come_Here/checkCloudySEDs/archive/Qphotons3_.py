import numpy as np

# Constants
h = 6.6261e-27  # Planck constant [erg.s]
c = 3.0e10      # Speed of light [Å/s]
hc = h * c

# Luminosity
L_lamb912 = 3.25e42  # erg/s/Å

# Wavelengths
lamb0 = 45.0   # Angstrom [i.e. 20 Ryd]
lamb1 = 912.0  # Angstrom [i.e. 1 Ryd]

# Wavelength grid
wgrid = np.arange(lamb0, lamb1 + 0.01, 0.01)  # Includes endpoint
n = len(wgrid)

# Delta lambda
dlamb = (wgrid[1] - wgrid[0]) * 1e-8

# Constant Luminosity along the wavelength
L_lamb = np.full(n, L_lamb912)

# Integration
s = 0.0
for i in range(n - 1):
    s += (dlamb * wgrid[i]  * 1e-8 * L_lamb[i]) / hc

print(f'Total number of photons: {s}')

