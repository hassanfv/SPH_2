import random

def contains(node, x, y):
    return node['x0'] <= x < node['x1'] and node['y0'] <= y < node['y1']

def subdivide(node):
    midX = (node['x0'] + node['x1']) / 2
    midY = (node['y0'] + node['y1']) / 2

    topLeft = {'x0': node['x0'], 'y0': node['y0'], 'x1': midX, 'y1': midY, 'points': [], 'children': []}
    topRight = {'x0': midX, 'y0': node['y0'], 'x1': node['x1'], 'y1': midY, 'points': [], 'children': []}
    bottomLeft = {'x0': node['x0'], 'y0': midY, 'x1': midX, 'y1': node['y1'], 'points': [], 'children': []}
    bottomRight = {'x0': midX, 'y0': midY, 'x1': node['x1'], 'y1': node['y1'], 'points': [], 'children': []}

    node['children'] = [topLeft, topRight, bottomLeft, bottomRight]

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

def main():
    # Create root node for range [-1, 1] in both x and y
    root = {'x0': -1, 'y0': -1, 'x1': 1, 'y1': 1, 'points': [], 'children': []}

    # Generate 1000 random particles within the specified range and insert them
    for _ in range(1000):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        insert(root, x, y)

    # Display the Quadtree structure
    nodes = [root]
    while nodes:
        current = nodes.pop()
        print(f"Node({current['x0']}, {current['y0']}, {current['x1']}, {current['y1']}) with {len(current['points'])} points")
        nodes.extend(current['children'])

if __name__ == "__main__":
    main()




