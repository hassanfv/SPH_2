import numpy as np
import pandas as pd
from numba import njit

@njit
def calculate_accelerations(x, y, mass):
    n = len(x)  # Number of particles
    eps2 = 0.025

    # Initialize acceleration arrays
    acc_x = np.zeros(n)
    acc_y = np.zeros(n)

    # Calculate accelerations
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                r = np.sqrt(dx*dx + dy*dy + eps2)
                
                # Avoid division by zero
                if r != 0:
                    F = mass[j] / (r*r*r)
                    acc_x[i] += F * dx
                    acc_y[i] += F * dy

    return acc_x, acc_y


df = pd.read_csv('data.csv')


x = df['x'].values
y = df['y'].values
mass = df['m'].values

acc_x, acc_y = calculate_accelerations(x, y, mass)
print("Acceleration in x-direction:", np.sort(acc_x))
#print("Acceleration in y-direction:", acc_y)

