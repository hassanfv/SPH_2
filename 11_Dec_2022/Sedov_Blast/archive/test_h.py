
import numpy as np
from do_smoothingZ import *
import time


pos = np.random.normal(0, 1, (8000, 3))


print(pos.shape)

TA = time.time()

h = do_smoothingZ(pos)


print('Elapsed time = ', time.time() - TA)







