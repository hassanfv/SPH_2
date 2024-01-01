import numpy as np

# Assume x, y, z, h, T are 1D numpy arrays of the same length
# x, y, z are the position coordinates
# h is the smoothing length
# T is the temperature

def distance(i, j):
    """Calculate the Euclidean distance between two particles."""
    return ((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2) ** 0.5

def friends_of_friends():
    """Identify groups (clumps) of particles using the Friends-of-Friends algorithm."""
    num_particles = len(x)
    groups = []
    grouped_particles = set()

    for i in range(num_particles):
        if i in grouped_particles:
            continue
        
        current_group = {i}
        group_members = [i]
        
        while group_members:
            new_members = []
            for j in group_members:
                for k in range(num_particles):
                    if k not in current_group:
                        if distance(j, k) <= min(h[j], h[k]):
                            new_members.append(k)
                            current_group.add(k)
            group_members = new_members

        if len(current_group) >= 50:
            groups.append(current_group)
            grouped_particles.update(current_group)

    return groups


clumps = friends_of_friends()

print(f"Found {len(clumps)} clumps with at least 100 particles each")






