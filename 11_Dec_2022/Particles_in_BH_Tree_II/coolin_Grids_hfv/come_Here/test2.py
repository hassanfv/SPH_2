
import numpy as np

Lsh = np.logspace(np.log10(1), np.log10(200), 20)

for x in Lsh:
  print(f'{x:.2f}')



