
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from numba import njit
from libsx import *
import time


#===== smooth_hX (non-parallel)
@njit
def do_smoothingX(poz):

    pos = poz[0]
    subpos = poz[1]

    N = pos.shape[0]
    M = subpos.shape[0]
    hres = []

    for i in range(M):
        dist = np.zeros(N)
        for j in range(N):
        
            dx = pos[j, 0] - subpos[i, 0]
            dy = pos[j, 1] - subpos[i, 1]
            dz = pos[j, 2] - subpos[i, 2]
            dist[j] = (dx**2 + dy**2 + dz**2)**0.5

        hres.append(np.sort(dist)[50])

    return np.array(hres) * 0.5


def compute_smoothing_lengths(positions, target_neighbors=64, tolerance=5):
    """
    Compute smoothing lengths for each particle in positions.
    
    :param positions: (N, 3) array of particle positions.
    :param target_neighbors: Desired number of neighbors.
    :param tolerance: Acceptable deviation from target_neighbors.
    :return: Array of smoothing lengths for each particle.
    """
    tree = KDTree(positions)
    lower_bound = target_neighbors - tolerance
    upper_bound = target_neighbors + tolerance

    smoothing_lengths = np.empty(len(positions))
    
    for i, pos in enumerate(positions):
        # Initial guess
        h = 0.1  # or some other reasonable starting value based on your system's scale
        neighbors = tree.query_ball_point(pos, h)
        
        # Refinement loop: Adjust h until the number of neighbors is within the acceptable range
        while len(neighbors) < lower_bound or len(neighbors) > upper_bound:
            if len(neighbors) < lower_bound:
                h *= 1.1  # Increase h
            else:
                h *= 0.9  # Decrease h

            neighbors = tree.query_ball_point(pos, h)
        
        smoothing_lengths[i] = h

    return smoothing_lengths * 0.5

# Example usage:
r = np.random.rand(10000, 3)  # This is just an example of 1000 random particles

#plt.scatter(r[:, 0], r[:, 1], s = 0.5, color = 'k')
#plt.show()

T1 = time.time()
hs = compute_smoothing_lengths(r)
#print(hs)
print('DKTree time = ', time.time() - T1)

#plt.hist(hs, bins = 20)
#plt.show()

T2 = time.time()
hs2 = do_smoothingX((r, r))
print('T2 = ', time.time() - T2)

T3 = time.time()
hs3 = h_smooth_fast(r, hs2*1.2)
print('T3 = ', time.time() - T3)

#plt.hist(hs2, bins = 20)
#plt.show()



