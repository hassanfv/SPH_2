import numpy as np
import glob
import readchar
import os

filez = np.sort(glob.glob('./Outputs/*.bin'))

for i in range(1, len(filez), 2):
  
  print(filez[i])
  
  os.remove(filez[i])
  
  kb = readchar.readkey()
  
  if kb == 'q':
    break
  
