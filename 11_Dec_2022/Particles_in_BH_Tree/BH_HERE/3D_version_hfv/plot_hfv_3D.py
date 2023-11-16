import struct
import matplotlib.pyplot as plt

def read_binary_file(filename):
    bodies = []
    nodes = []
    
    with open(filename, "rb") as file:
        nBodies, nNodes = struct.unpack('ii', file.read(8))
        
        # Read body positions
        for _ in range(nBodies):
            x, y, z = struct.unpack('fff', file.read(12))
            bodies.append((x, y, z))

        # Read node boundaries
        for _ in range(nNodes):
            min_x, min_y, min_z = struct.unpack('fff', file.read(12))
            max_x, max_y, max_z = struct.unpack('fff', file.read(12))
            nodes.append(((min_x, min_y, min_z), (max_x, max_y, max_z)))

    return bodies, nodes

def plot_bodies_and_boundaries(bodies, nodes):
    fig, ax = plt.subplots()
    
    # Plot bodies
    x_vals, y_vals, _ = zip(*bodies)
    ax.scatter(x_vals, y_vals, s=1)

    # Draw boundaries
    for min_corner, max_corner in nodes:
        min_x, min_y, _ = min_corner
        max_x, max_y, _ = max_corner
        rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, color='red')
        ax.add_patch(rect)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Bodies and Octant Boundaries')
    plt.show()

# Replace 'path_to_file.bin' with your actual file path
bodies, nodes = read_binary_file('BH.bin')
plot_bodies_and_boundaries(bodies, nodes)

