

#def find_clumps(particles, cold_particles_indices, T_threshold=500, density_factor=50, min_particles=100):
def find_clumps(particles, cold_particles_indices, min_particles=50):
    """
    Identify clumps in particle data based on temperature and density thresholds.

    Parameters:
    - particles: numpy array of shape (n_particles, 5) where each row represents a particle and
                 columns represent x, y, z coordinates, temperature, and density, respectively.
    - T_threshold: Temperature threshold to consider a particle as cold.
    - density_factor: Factor by which a particle's density must exceed the mean density to be considered.
    - min_particles: Minimum number of particles to consider a group a clump.

    Returns:
    - clumps: A list of lists, where each inner list contains the indices of particles in a clump.
    """
    #mean_density = np.mean(particles[:, 4])
    #cold_particles_indices = np.where((particles[:, 3] < T_threshold) & (particles[:, 4] > density_factor * mean_density))[0]
    #cold_particles_indices = # Indices from pickle file!!!!!!!!!!!!!!!!!!!!!!!!!!!
    in_clump = np.zeros(len(particles), dtype=bool)
    clumps = []

    for i in cold_particles_indices:
        if in_clump[i]:
            continue  # Skip if particle is already in a clump

        # Start a new clump with the current particle
        current_clump = [i]
        in_clump[i] = True

        # Iteratively add nearby particles to the clump
        for j in current_clump:
            for k in cold_particles_indices:
                if in_clump[k]:
                    continue  # Skip if particle is already in a clump

                distance = np.sqrt(np.sum((particles[j, :3] - particles[k, :3]) ** 2))
                #if distance <= particles[k, 4]:  # Using particle's density as a proxy for smoothing length
                if distance <= (particles[j, 3] + particles[j, 3]):  # Using the sum of the smoothing lengths of particles j, and k.
                    current_clump.append(k)
                    in_clump[k] = True

        # Check if the current clump meets the minimum size requirement
        if len(current_clump) >= min_particles:
            clumps.append(current_clump)

    return clumps

# Example usage (assuming 'particles' is a numpy array with the required structure)
# clumps = find_clumps(particles)
# print(f"Found {len(clumps)} clumps")




