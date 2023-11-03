import numpy as np
import matplotlib.pyplot as plt

# Define particle class
class Particle:
    def __init__(self, x, y, mass=1.0):
        self.x = x
        self.y = y
        self.mass = mass
        self.vx = 0
        self.vy = 0
        self.fx = 0
        self.fy = 0

# Define the Quadtree node class
class Quad:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.children = []
        self.particle = None
        self.center_of_mass_x = 0
        self.center_of_mass_y = 0
        self.total_mass = 0

    def insert(self, particle):
        if not self.contains(particle):
            return False

        if len(self.children) == 0 and self.particle is None:
            self.particle = particle
            self.center_of_mass_x = particle.x
            self.center_of_mass_y = particle.y
            self.total_mass = particle.mass
            return True

        if len(self.children) == 0:
            self.subdivide()

        for child in self.children:
            if child.insert(particle):
                break

        if self.particle:
            self.children[0].insert(self.particle)
            self.particle = None

        self.update_mass_and_center_of_mass()

    def subdivide(self):
        hx = (self.x1 + self.x2) * 0.5
        hy = (self.y1 + self.y2) * 0.5

        self.children.append(Quad(self.x1, self.y1, hx, hy))
        self.children.append(Quad(hx, self.y1, self.x2, hy))
        self.children.append(Quad(self.x1, hy, hx, self.y2))
        self.children.append(Quad(hx, hy, self.x2, self.y2))

    def contains(self, particle):
        return (self.x1 <= particle.x < self.x2) and (self.y1 <= particle.y < self.y2)

    def update_mass_and_center_of_mass(self):
        self.total_mass = 0
        self.center_of_mass_x = 0
        self.center_of_mass_y = 0
        for child in self.children:
            self.total_mass += child.total_mass
            self.center_of_mass_x += child.center_of_mass_x * child.total_mass
            self.center_of_mass_y += child.center_of_mass_y * child.total_mass
        if self.total_mass > 0:
            self.center_of_mass_x /= self.total_mass
            self.center_of_mass_y /= self.total_mass

# Simulation parameters
N = 100
G = 1e-3
particles = [Particle(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(N)]
threshold = 0.5

def compute_force(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    r = np.sqrt(dx*dx + dy*dy)
    if r == 0: return (0, 0)

    force = G * p1.mass * p2.mass / (r**3)
    return dx * force, dy * force

def traverse_tree(node, particle):
    if node.particle and node.particle != particle:
        fx, fy = compute_force(particle, node.particle)
        particle.fx += fx
        particle.fy += fy
    elif len(node.children) > 0:
        dx = node.center_of_mass_x - particle.x
        dy = node.center_of_mass_y - particle.y
        distance = np.sqrt(dx**2 + dy**2)
        if distance == 0: return
        size = node.x2 - node.x1

        if size/distance < threshold:
            fx, fy = compute_force(particle, Particle(node.center_of_mass_x, node.center_of_mass_y, node.total_mass))
            particle.fx += fx
            particle.fy += fy
        else:
            for child in node.children:
                traverse_tree(child, particle)

def update_particles(dt):
    for p in particles:
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.vx += p.fx * dt
        p.vy += p.fy * dt
        p.fx = 0
        p.fy = 0

def run_simulation(timesteps, dt):
    for _ in range(timesteps):
        root = Quad(-1, -1, 1, 1)
        for p in particles:
            root.insert(p)

        for p in particles:
            traverse_tree(root, p)

        update_particles(dt)

        plt.scatter([p.x for p in particles], [p.y for p in particles], s=5)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()

run_simulation(timesteps=5, dt=0.01)

