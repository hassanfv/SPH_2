import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def read_binary_file(filename):
    with open(filename, 'rb') as file:
        N = np.fromfile(file, dtype=np.int32, count=1)[0]
        expected_number_of_nodes = np.fromfile(file, dtype=np.int32, count=1)[0]
        unc_pos = np.fromfile(file, dtype=np.float32, count=2*N).reshape(N, 2)
        all_nodes = np.fromfile(file, dtype=np.float32, count=expected_number_of_nodes*8).reshape(expected_number_of_nodes, 8)
    return unc_pos, all_nodes

def plot_tree(unc_pos, all_nodes):
    fig, ax = plt.subplots()
    ax.scatter(unc_pos[:, 0], unc_pos[:, 1], s=10)  # Plot particles
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)

    node_gen = (node for node in all_nodes if node[2] > 0)  # Generator for valid nodes

    def on_key(event):
        try:
            node = next(node_gen)  # Get the next node
            center_x, center_y, width = node[0], node[1], node[2]
            lower_left_x = center_x - width / 2
            lower_left_y = center_y - width / 2
            rect = patches.Rectangle((lower_left_x, lower_left_y), width, width, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.draw()  # Redraw the plot
        except StopIteration:
            return  # No more nodes to add

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

def main():
    unc_pos, all_nodes = read_binary_file('bh_tree_data.bin')
    plot_tree(unc_pos, all_nodes)

if __name__ == "__main__":
    main()

