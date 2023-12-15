import struct
import numpy as np

filename = "h_x.bin"

with open(filename, 'rb') as file:
    # Read the entire file into a bytes object
    data = file.read()

# Calculate the number of integers in the file
num_floats = len(data) // 4  # 4 bytes per float

# Unpack the data into integers
h_x = np.array(struct.unpack(f'{num_floats}i', data))

print("h_x.shape = ", h_x.shape)

nx = np.where(h_x != 0)[0]

print("len(nx) = ", len(nx))
print()
print("nx = ", nx)

