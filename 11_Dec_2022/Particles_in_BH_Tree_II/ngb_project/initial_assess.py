import struct
import numpy as np

filename = "h_child.bin"

with open(filename, 'rb') as file:
    # Read the entire file into a bytes object
    data = file.read()

# Calculate the number of integers in the file
num_ints = len(data) // 4  # 4 bytes per int

# Unpack the data into integers
h_child = np.array(struct.unpack(f'{num_ints}i', data))

print("h_child.shape = ", h_child.shape)

nx = np.where(h_child != -1)[0]

print("len(nx) = ", len(nx))
print()
print("nx = ", nx)

