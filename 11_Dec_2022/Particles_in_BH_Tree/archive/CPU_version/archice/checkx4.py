import numpy as np
import readchar
import matplotlib.patches as patches
import matplotlib.pyplot as plt


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
  if not quad['children'] and quad['particle_count'] < Nngb:
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
  if not quad['children'] and quad['particle_count'] == Nngb:
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



#===== find_quad_with_particle
def find_quad_with_particle(quad, particle, min_count, max_count):
    """
    Traverse the Barnes-Hut tree and find the quad containing the given particle with a particle_count in the specified range.
    
    Parameters:
    - quad: The current quad being examined
    - particle: The target particle we want to find
    - min_count, max_count: The range of particle_count we're interested in
    
    Returns:
    - The quad containing the particle with particle_count in the given range, or None if not found.
    """
    
    # Base condition: if the quad doesn't contain the particle, return None
    if not contains(quad, particle):
        return None
    
    # Check if the current quad's particle_count is in the desired range
    if min_count <= quad['particle_count'] <= max_count:
        for p in quad['particles']:
            if p == particle:
                return quad
    
    # Recursively search in the child quads
    for child in quad['children']:
        result = find_quad_with_particle(child, particle, min_count, max_count)
        if result:
            return result

    return None



#===== traverse_tree
def traverse_tree(quad, depth=0):
  """Recursively traverse the Barnes-Hut tree using DFS."""
  
  # Process the current quad (e.g., print its properties)
  print('  ' * depth + f'Quad at depth {depth}: x1={quad["x1"]}, y1={quad["y1"]}, x2={quad["x2"]}, y2={quad["y2"]}')
  print('  ' * depth + f'Center of Mass: ({quad["center_of_mass_x"]}, {quad["center_of_mass_y"]})')
  print('  ' * depth + f'Particles: {quad["particles"]}')
  
  xL = quad['x1']
  yL = quad['y1']
  
  xR = quad['x2']
  yR = quad['y2']
  
  # Add a rectangle to the plot
  rectangle = patches.Rectangle((xL, yL), xR - xL, yR - yL, edgecolor='blue', facecolor='none')
  ax.add_patch(rectangle)
  
  # If the current quad has children, recurse on them
  for child in quad['children']:
    traverse_tree(child, depth + 1)




N = 1000

Nngb = 5

#xx = [-0.90, -0.80, -0.40, -0.20]
#yy = [0.2, 0.2, 0.2, 0.2]

xx = [np.round(np.random.uniform(-1, 1),2) for _ in range(N)]
yy = [np.round(np.random.uniform(-1, 1),2) for _ in range(N)]

#particles = [create_particle(np.round(np.random.uniform(-1, 1),2), np.round(np.random.uniform(-1, 1),2)) for _ in range(N)]
particles = [create_particle(xx[i], yy[i]) for i in range(len(xx))]
print(particles)
print('----------------------------------------')
print()

# Create a figure and axis
fig, ax = plt.subplots()

ax.scatter(xx, yy, s = 20, color = 'k')

root = create_quad(-1, -1, 1, 1)

for p in particles:
  insert_to_quad(root, p)

traverse_tree(root)

i_p = 50  # for example
target_particle = create_particle(xx[i_p], yy[i_p])
quadx = find_quad_with_particle(root, target_particle, 1, 100)

xL = quadx['x1']
yL = quadx['y1']

xR = quadx['x2']
yR = quadx['y2']

# Add a rectangle to the plot
rectangle = patches.Rectangle((xL, yL), xR - xL, yR - yL, edgecolor='red', facecolor='none')
ax.add_patch(rectangle)

print('====================')
print(quadx)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.show()







