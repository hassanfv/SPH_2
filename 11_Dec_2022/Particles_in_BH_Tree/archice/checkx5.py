import numpy as np
import readchar


np.random.seed(42)

#===== create_particle
def create_particle(x, y, mass=1.0):
  return {
    'x': x,
    'y': y,
    'mass': mass
  }


#===== create_quad
def create_quad(x1, y1, x2, y2):
  return {
    'x1': x1,
    'y1': y1,
    'x2': x2,
    'y2': y2,
    'particles': [],
    'children': [],
    'center_of_mass_x': 0,
    'center_of_mass_y': 0,
    'total_mass': 0,
    'particle_count': 0  # Add this field to track the number of particles
  }



#===== contains
def contains(quad, particle):
  return (quad['x1'] <= particle['x'] < quad['x2']) and (quad['y1'] <= particle['y'] < quad['y2'])



#===== insert_to_quad
def insert_to_quad(quad, particle):
  if not contains(quad, particle):  # Check if the particle is within the quad boundary.
    return False

  # If the quad has no children and particle_count is less than 2, insert the particle in the quad.
  if not quad['children'] and quad['particle_count'] < 2:
    quad['particles'].append(particle)
    quad['particle_count'] += 1
        
    # Recalculate center of mass and total mass
    total_mass = sum(p['mass'] for p in quad['particles'])
    center_of_mass_x = sum(p['x'] * p['mass'] for p in quad['particles']) / total_mass
    center_of_mass_y = sum(p['y'] * p['mass'] for p in quad['particles']) / total_mass
        
    quad['center_of_mass_x'] = center_of_mass_x
    quad['center_of_mass_y'] = center_of_mass_y
    quad['total_mass'] = total_mass

    return True

  # Subdivide the quad if particle_count is 2 and quad has no children yet.
  if not quad['children'] and quad['particle_count'] == 2:
    subdivide(quad)
        
    # Try to insert the existing particles into the children.
    existing_particles = quad['particles'][:]  # Take a copy of the particles list.
    quad['particles'] = []
    for existing_particle in existing_particles:
      for child in quad['children']:
        if insert_to_quad(child, existing_particle):
          break

  # Try to insert the current particle into the children.
  for child in quad['children']:
    if insert_to_quad(child, particle):
      quad['particle_count'] += 1  # Increment the particle_count of the parent
      return True

  return False




#===== subdivide
def subdivide(quad):
  hx, hy = (quad['x1'] + quad['x2']) / 2, (quad['y1'] + quad['y2']) / 2
  quad['children'].append(create_quad(quad['x1'], quad['y1'], hx, hy))
  quad['children'].append(create_quad(hx, quad['y1'], quad['x2'], hy))
  quad['children'].append(create_quad(quad['x1'], hy, hx, quad['y2']))
  quad['children'].append(create_quad(hx, hy, quad['x2'], quad['y2']))



N = 5

xx = [-0.90, -0.80, -0.40, -0.20]
yy = [0.2, 0.2, 0.2, 0.2]

#particles = [create_particle(np.round(np.random.uniform(-1, 1),2), np.round(np.random.uniform(-1, 1),2)) for _ in range(N)]
particles = [create_particle(xx[i], yy[i]) for i in range(len(xx))]
print(particles)
print('----------------------------------------')
print()

root = create_quad(-1, -1, 1, 1)

for p in particles:
  insert_to_quad(root, p)



def traverse_tree(quad, depth=0):
  """Recursively traverse the Barnes-Hut tree using DFS."""
  
  # Process the current quad (e.g., print its properties)
  print('  ' * depth + f'Quad at depth {depth}: x1={quad["x1"]}, y1={quad["y1"]}, x2={quad["x2"]}, y2={quad["y2"]}')
  print('  ' * depth + f'Center of Mass: ({quad["center_of_mass_x"]}, {quad["center_of_mass_y"]})')
  print('  ' * depth + f'Particles: {quad["particles"]}')
  
  # If the current quad has children, recurse on them
  for child in quad['children']:
    traverse_tree(child, depth + 1)


traverse_tree(root)


