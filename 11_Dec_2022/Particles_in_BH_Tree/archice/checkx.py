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
    'particle': None,
    'children': [],
    'center_of_mass_x': 0,
    'center_of_mass_y': 0,
    'total_mass': 0
  }



#===== contains
def contains(quad, particle):
  return (quad['x1'] <= particle['x'] < quad['x2']) and (quad['y1'] <= particle['y'] < quad['y2'])



#===== insert_to_quad
def insert_to_quad(quad, particle):
  if not contains(quad, particle): # checks whether the particle is within the quad boundary!
    return False

  if not quad['children'] and quad['particle'] is None: # Checks if the quad has no children and also no particle. If so, inserts the particle there!
    quad['particle'] = particle # inserting the particle.
    quad['center_of_mass_x'] = particle['x'] # Since we have one article therefore the center_of_mass_x = x
    quad['center_of_mass_y'] = particle['y'] # Since we have one article therefore the center_of_mass_y = y
    quad['total_mass'] = particle['mass'] # Since we have one partice in the quad, therefore, total_mass = particle mass.
    return True

  # This will be activated if we already have a particle inside the quad. And since we are adding another particle
  # inside this quad, therefore, we need to subdivide it as only one particle can reside in  each quad.
  # Also this will only be activated if the quad has no children. If it has children then it will skip to the next line
  # and tries to find a children "with no particle and no children" to place the particle there.
  if not quad['children']:
    subdivide(quad)

  # Now that the quad has beed subdivided, we should first place the already existing particle in the right sub-quad and
  # reset quad['particle'] = None. Only then we can try to insert the new particle as we do it later after this if condition.
  if quad['particle']:
    existing_particle = quad['particle']
    quad['particle'] = None
    for child in quad['children']:
      if insert_to_quad(child, existing_particle):
        break

  # Now that the existing particle is placed in the right quad, we can go ahead and try to insert the current particle.
  for child in quad['children']:
    if insert_to_quad(child, particle):
      break

  update_mass_and_center_of_mass(quad)


#===== subdivide
def subdivide(quad):
  hx, hy = (quad['x1'] + quad['x2']) / 2, (quad['y1'] + quad['y2']) / 2
  quad['children'].append(create_quad(quad['x1'], quad['y1'], hx, hy))
  quad['children'].append(create_quad(hx, quad['y1'], quad['x2'], hy))
  quad['children'].append(create_quad(quad['x1'], hy, hx, quad['y2']))
  quad['children'].append(create_quad(hx, hy, quad['x2'], quad['y2']))



#===== update_mass_and_center_of_mass
def update_mass_and_center_of_mass(quad):
  quad['total_mass'] = sum(child['total_mass'] for child in quad['children'])
  quad['center_of_mass_x'] = sum(child['center_of_mass_x'] * child['total_mass'] for child in quad['children']) / quad['total_mass']
  quad['center_of_mass_y'] = sum(child['center_of_mass_y'] * child['total_mass'] for child in quad['children']) / quad['total_mass']



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
  print()
  print(root)
  
  kb = readchar.readkey()
  
  if kb == 'q':
    break






