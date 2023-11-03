import numpy as np
import matplotlib.pyplot as plt

def read_binary_file(filename):
    with open(filename, 'rb') as f:
        # Read N_tot from the start of the file
        N_tot = np.fromfile(f, dtype=np.int32, count=1)[0]
        
        # Read the other arrays
        Typnew = np.fromfile(f, dtype=np.int32, count=N_tot)
        xnew = np.fromfile(f, dtype=np.float32, count=N_tot)
        ynew = np.fromfile(f, dtype=np.float32, count=N_tot)
        znew = np.fromfile(f, dtype=np.float32, count=N_tot)
        vxnew = np.fromfile(f, dtype=np.float32, count=N_tot)
        vynew = np.fromfile(f, dtype=np.float32, count=N_tot)
        vznew = np.fromfile(f, dtype=np.float32, count=N_tot)
        unew = np.fromfile(f, dtype=np.float32, count=N_tot)
        hnew = np.fromfile(f, dtype=np.float32, count=N_tot)
        epsnew = np.fromfile(f, dtype=np.float32, count=N_tot)
        massnew = np.fromfile(f, dtype=np.float32, count=N_tot)

    return xnew, ynew

def plot_scatter(x, y):
    plt.scatter(x, y, marker='.', s=1)  # s=1 for small marker size
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter plot of x vs y")
    plt.show()

if __name__ == "__main__":
    filename = "IC_R_1334k.bin"  # replace YOURVALUE with the correct value
    x, y = read_binary_file(filename)
    plot_scatter(x, y)

