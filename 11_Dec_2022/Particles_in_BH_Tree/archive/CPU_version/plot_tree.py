import matplotlib.pyplot as plt
import csv

# Load particle positions
particles_x = []
particles_y = []
with open('particles.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        _, x, y = map(float, row)
        particles_x.append(x)
        particles_y.append(y)

# Load cell boundaries
boxes = []
with open('boxes.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        box_id, center_x, center_y, width = map(float, row)
        half_width = width / 2.0
        xmin = center_x - half_width
        xmax = center_x + half_width
        ymin = center_y - half_width
        ymax = center_y + half_width
        boxes.append((xmin, xmax, ymin, ymax))

# Plot cells
for box in boxes:
    xmin, xmax, ymin, ymax = box
    plt.plot([xmin, xmin], [ymin, ymax], 'k-')  # left side
    plt.plot([xmax, xmax], [ymin, ymax], 'k-')  # right side
    plt.plot([xmin, xmax], [ymin, ymin], 'k-')  # bottom
    plt.plot([xmin, xmax], [ymax, ymax], 'k-')  # top

# Plot particles
plt.scatter(particles_x, particles_y, c='blue', s=20)

plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('BH-Tree with Particles and Cells')
plt.show()

