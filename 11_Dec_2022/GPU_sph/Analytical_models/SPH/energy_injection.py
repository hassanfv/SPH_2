import numpy as np

def inject_energy(bh_position, gas_positions, gas_internal_energy, gas_smoothing_lengths, epsilon, L_Edd, dt, n_neighbors=64):
    # Calculate distances between the black hole and gas particles
    distances = np.linalg.norm(gas_positions - bh_position, axis=1)
    
    # Find the indices of the nearest n_neighbors gas particles
    nearest_neighbors_indices = np.argpartition(distances, n_neighbors)[:n_neighbors]
    
    # Get the smoothing lengths and internal energy of the nearest neighbors
    neighbor_smoothing_lengths = gas_smoothing_lengths[nearest_neighbors_indices]
    neighbor_internal_energy = gas_internal_energy[nearest_neighbors_indices]
    
    # Calculate the kernel weights for each nearest neighbor
    kernel_weights = (1.0 / (np.pi * neighbor_smoothing_lengths**3)) * (1 - (distances[nearest_neighbors_indices] / neighbor_smoothing_lengths)**2)**3

    # Normalize the kernel weights
    kernel_weights /= np.sum(kernel_weights)
    
    # Calculate the energy injection for each neighbor, taking into account the time step
    energy_injection = epsilon * L_Edd * dt * kernel_weights
    
    # Update the internal energy of the neighboring gas particles
    for i, idx in enumerate(nearest_neighbors_indices):
        gas_internal_energy[idx] += energy_injection[i]

