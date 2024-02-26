
import numpy as np


def interpolate_4d_hypercube(p, x, y, z, w):
    """
    Perform 4D interpolation on point p within a hypercube defined by the coordinates x, y, z, w.
    
    :param p: An array of shape (16,) representing the values at the 16 points of the hypercube
              following the binary encoding convention for indexing.
    :param x: The x-coordinate for interpolation, normalized between 0 and 1.
    :param y: The y-coordinate for interpolation, normalized between 0 and 1.
    :param z: The z-coordinate for interpolation, normalized between 0 and 1.
    :param w: The w-coordinate for interpolation, normalized between 0 and 1.
    :return: The interpolated value.
    """
    
    # Interpolate along the x-axis
    c00 = p[0] * (1 - x) + p[8] * x
    c01 = p[1] * (1 - x) + p[9] * x
    c10 = p[2] * (1 - x) + p[10] * x
    c11 = p[3] * (1 - x) + p[11] * x
    c20 = p[4] * (1 - x) + p[12] * x
    c21 = p[5] * (1 - x) + p[13] * x
    c30 = p[6] * (1 - x) + p[14] * x
    c31 = p[7] * (1 - x) + p[15] * x

    # Correct interpolation along the y-axis
    c0 = c00 * (1 - y) + c10 * y
    c1 = c01 * (1 - y) + c11 * y
    c2 = c20 * (1 - y) + c30 * y
    c3 = c21 * (1 - y) + c31 * y

    # Interpolate along the z-axis
    c = c0 * (1 - z) + c2 * z
    d = c1 * (1 - z) + c3 * z

    # Finally, interpolate along the w-axis
    c_final = c * (1 - w) + d * w

    return c_final



nxnH0 = 0
nxT0  = 0
nxr0  = 0
nxNH0 = 0

nxnH1 = nxnH0 + 1
nxT1  = nxT0 + 1
nxr1  = nxr0 + 1
nxNH1 = nxNH0 + 1

for i in [nxnH0, nxnH1]:
  for j in [nxT0, nxT1]:
    for k in [nxr0, nxr1]:
      for l in [nxNH0, nxNH1]:
        print(i, j, k, l)
        #P[s] = u[i, j, k, l]

# Example usage:
# The values at the 16 points of the hypercube arranged using the binary encoding convention
points = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

# The coordinates for interpolation
x_d, y_d, z_d, w_d = 0.5, 0.5, 0.5, 0.5

# Perform the 4D interpolation
interpolated_value = interpolate_4d_hypercube(points, x_d, y_d, z_d, w_d)

print(interpolated_value)







