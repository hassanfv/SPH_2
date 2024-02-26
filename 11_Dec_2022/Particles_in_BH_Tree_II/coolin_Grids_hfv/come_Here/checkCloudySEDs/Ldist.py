import math
import numpy as np

import math

def luminosity_distance(z, H0=71, WM=0.27, WV=None):
    # Constants
    c = 299792.458  # Speed of light in km/s
    
    # Check if WV is given, else calculate it based on WM and H0
    if WV is None:
        WV = 1.0 - WM - 0.4165 / (H0 * H0)
    
    # Omega(radiation) and Omega(curvature)
    WR = 4.165E-5 / (H0 / 100.0)**2
    WK = 1 - WM - WR - WV

    # Scale factor
    az = 1.0 / (1 + z)
    
    # Initialize variables for integration
    n = 1000  # Number of integration steps
    DCMR = 0.0  # Comoving radial distance

    # Integrate to compute DCMR
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = math.sqrt(WK + (WM / a) + (WR / (a * a)) + (WV * a * a))
        DCMR += 1.0 / (a * adot)
    
    DCMR *= (1.0 - az) / n
    
    # Calculate DL in Mpc
    DL_Mpc = DCMR * (1 + z) * (c / H0)

    return DL_Mpc




# Example usage:
z = 3.19  # Redshift
DL = luminosity_distance(z)
DL_cm = DL * 1e6 * 3.086e18
print(f"Luminosity Distance for z={z}: {DL:.2f} Mpc or {DL_cm:.4E} cm")

fobs = 3.40e-17 # erg/s/cm^2/A or /Hz

L = 4.0 * np.pi * DL_cm**2 * fobs

print(f'L = {L:.3E}')


