import struct
import matplotlib.pyplot as plt

# Function to read the binary file
def read_data(filename):
    with open(filename, 'rb') as file:
        nBodies = struct.unpack('i', file.read(4))[0]
        nNodes = struct.unpack('i', file.read(4))[0]

        positions = []
        for _ in range(nBodies):
            x = struct.unpack('d', file.read(8))[0]
            y = struct.unpack('d', file.read(8))[0]
            positions.append((x, y))

        boundaries = []
        for _ in range(nNodes):
            topLeftX = struct.unpack('d', file.read(8))[0]
            topLeftY = struct.unpack('d', file.read(8))[0]
            botRightX = struct.unpack('d', file.read(8))[0]
            botRightY = struct.unpack('d', file.read(8))[0]
            boundaries.append((topLeftX, topLeftY, botRightX, botRightY))

    return positions, boundaries

# Function to create the scatter plot
def plot_data(positions, boundaries):
    x, y = zip(*positions)
    plt.scatter(x, y, s=5)

    for topLeftX, topLeftY, botRightX, botRightY in boundaries:
        plt.plot([topLeftX, topLeftX], [topLeftY, botRightY], color='r') # left line
        plt.plot([topLeftX, botRightX], [topLeftY, topLeftY], color='r') # top line
        plt.plot([botRightX, botRightX], [topLeftY, botRightY], color='r') # right line
        plt.plot([topLeftX, botRightX], [botRightY, botRightY], color='r') # bottom line

    plt.xlabel('Position X')
    plt.ylabel('Position Y')
    plt.title('Scatter Plot of Bodies and Quadtree Boundaries')
    plt.show()

# Read data and plot
positions, boundaries = read_data('BH.bin')
plot_data(positions, boundaries)

