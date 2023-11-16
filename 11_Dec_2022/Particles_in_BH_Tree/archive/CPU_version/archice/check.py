
import random
import readchar
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)


#===== contains
def contains(node, x, y):
  return node['x0'] <= x < node['x1'] and node['y0'] <= y <= node['y1']


#===== subdivide
def subdivide(node):
    midX = (node['x0'] + node['x1']) / 2
    midY = (node['y0'] + node['y1']) / 2

    topLeft = {'x0': node['x0'], 'y0': node['y0'], 'x1': midX, 'y1': midY, 'points': [], 'children': []}
    topRight = {'x0': midX, 'y0': node['y0'], 'x1': node['x1'], 'y1': midY, 'points': [], 'children': []}
    bottomLeft = {'x0': node['x0'], 'y0': midY, 'x1': midX, 'y1': node['y1'], 'points': [], 'children': []}
    bottomRight = {'x0': midX, 'y0': midY, 'x1': node['x1'], 'y1': node['y1'], 'points': [], 'children': []}

    node['children'] = [topLeft, topRight, bottomLeft, bottomRight]


#===== insert
def insert(node, x, y):
    if not contains(node, x, y):
        return False

    if len(node['points']) < 1 and not node['children']:
        node['points'].append((x, y))
        return True

    if not node['children']:
        subdivide(node)

    for child in node['children']:
        if insert(child, x, y):
            return True
    return False






root = {'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'points': [], 'children': []}
print(root)


# Generate 1000 random particles within the specified range and insert them

x = [-0.90, -0.80, -0.40, -0.20]
y = [0.2, 0.2, 0.2, 0.2]

res = []
for i in range(len(x)):
  #x = random.uniform(-1, 1)
  #y = random.uniform(-1, 1)
  insert(root, np.round(x[i], 2), np.round(y[i], 2))
  
  res.append([x, y])
  
  print()
  print(root)
  
  kb = readchar.readkey()
  
  if kb == 'q':
    break


res = np.array(res)
xx = res[:, 0]
yy = res[:, 1]

plt.scatter(xx, yy, s = 10, color = 'k')
xy = 1
plt.xlim(-xy, xy)
plt.ylim(-xy, xy)
plt.show()




