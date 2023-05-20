import numpy as np

# Create the arrays
array1 = np.random.random(1000).astype(np.float32)
array2 = np.random.random(2000).astype(np.float32)
array3 = np.random.random(3000).astype(np.float32)
array4 = np.random.random(4000).astype(np.float32)

# Save the arrays to a binary file
with open("arrays.bin", "wb") as file:
    file.write(array1.tobytes())
    file.write(array2.tobytes())
    file.write(array3.tobytes())
    file.write(array4.tobytes())



print(array2[:10])
