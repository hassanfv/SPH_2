import numpy as np
import pandas as pd
from numba import njit

@njit
def calculate_accelerations(x, y, z, mass):
    n = len(x)  # Number of particles
    eps2 = 0.025

    # Initialize acceleration arrays
    acc_x = np.zeros(n)
    acc_y = np.zeros(n)
    acc_z = np.zeros(n)

    # Calculate accelerations
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dz = z[j] - z[i]
                r = np.sqrt(dx*dx + dy*dy + dz*dz + eps2)
                
                # Avoid division by zero
                if r != 0:
                    F = mass[j] / (r*r*r)
                    acc_x[i] += F * dx
                    acc_y[i] += F * dy
                    acc_z[i] += F * dz

    return acc_x, acc_y, acc_z


df = pd.read_csv('data.csv')


x = df['x'].values
y = df['y'].values
z = df['z'].values
mass = df['m'].values

acc_x, acc_y, acc_z = calculate_accelerations(x, y, z, mass)
print("Acceleration in y-direction:", np.sort(acc_y))
#print("Acceleration in y-direction:", acc_y)






