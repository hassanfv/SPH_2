import numpy as np

# Simulation parameters
N = 100

def create_particle(x, y, mass=1.0):
    return {
        'x': x,
        'y': y,
        'mass': mass
    }

def create_quad(x1, y1, x2, y2):
    return {
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'children': [],
        'particle': None,
        'center_of_mass_x': 0,
        'center_of_mass_y': 0,
        'total_mass': 0
    }

def insert_to_quad(quad, particle):
    if not contains(quad, particle):
        return False

    if not quad['children'] and quad['particle'] is None:
        quad['particle'] = particle
        quad['center_of_mass_x'] = particle['x']
        quad['center_of_mass_y'] = particle['y']
        quad['total_mass'] = particle['mass']
        return True

    if not quad['children']:
        subdivide(quad)

    for child in quad['children']:
        if insert_to_quad(child, particle):
            break

    if quad['particle']:
        insert_to_quad(quad['children'][0], quad['particle'])
        quad['particle'] = None

    update_mass_and_center_of_mass(quad)

def contains(quad, particle):
    return (quad['x1'] <= particle['x'] < quad['x2']) and (quad['y1'] <= particle['y'] < quad['y2'])

def subdivide(quad):
    hx, hy = (quad['x1'] + quad['x2']) / 2, (quad['y1'] + quad['y2']) / 2
    quad['children'].append(create_quad(quad['x1'], quad['y1'], hx, hy))
    quad['children'].append(create_quad(hx, quad['y1'], quad['x2'], hy))
    quad['children'].append(create_quad(quad['x1'], hy, hx, quad['y2']))
    quad['children'].append(create_quad(hx, hy, quad['x2'], quad['y2']))

def update_mass_and_center_of_mass(quad):
    quad['total_mass'] = sum(child['total_mass'] for child in quad['children'])
    quad['center_of_mass_x'] = sum(child['center_of_mass_x'] * child['total_mass'] for child in quad['children']) / quad['total_mass']
    quad['center_of_mass_y'] = sum(child['center_of_mass_y'] * child['total_mass'] for child in quad['children']) / quad['total_mass']

def build_tree(particles):
    root = create_quad(-1, -1, 1, 1)
    for p in particles:
        insert_to_quad(root, p)
    return root

particles = [create_particle(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(N)]
quadtree = build_tree(particles)

