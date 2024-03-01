
import numpy as np
import pickle



with open('z-uRes.pkl', 'rb') as f:
  lst = pickle.load(f)

print(len(lst))


with open('float32_uRes.bin', 'wb') as binary_file:
  for element in lst:
  
    tmp = np.round(element, 3)
  
    # Convert the element to np.float32
    float32_element = np.float32(tmp)
    
    # Write the np.float32 element to the binary file
    binary_file.write(float32_element.tobytes())

