import numpy as np

def gaussian_kernel(r, h):
    """
    Gaussian kernel function for SPH

    :param r: distance between particles
    :param h: smoothing length
    :return: value of the Gaussian kernel at r
    """
    # Normalization factor for 3D
    norm = 1 / (h**3 * (np.pi**(3/2)))
    
    # Gaussian function
    return norm * np.exp(-r**2 / h**2)

def calculate_density(positions, masses, h):
    """
    Calculate the density at each point using the SPH Gaussian kernel approach.

    :param positions: array of particle positions
    :param masses: array of particle masses
    :param h: smoothing length
    :return: array of densities at each position
    """
    num_particles = len(positions)
    densities = np.zeros(num_particles)
    
    # Calculate the density at each position
    for i in range(num_particles):
        r_i = positions[i]
        # Sum the kernel contributions from all other particles
        for j in range(num_particles):
            if i != j:  # Avoid self-contribution
                r_j = positions[j]
                r_ij = np.linalg.norm(r_i - r_j)
                densities[i] += masses[j] * gaussian_kernel(r_ij, h)
    
    return densities

# Example usage:
# Assuming we have a list of positions and corresponding masses for the particles,
# and a smoothing length h, we would calculate the density like this:

# Example data (positions must be an array of 3D coordinates, masses is an array of masses, h is the smoothing length)
positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # Placeholder positions
masses = np.array([1, 1, 1])  # Placeholder masses
h = 1.0  # Placeholder smoothing length

# Calculate densities
rho = calculate_density(positions, masses, h)
print(rho)

