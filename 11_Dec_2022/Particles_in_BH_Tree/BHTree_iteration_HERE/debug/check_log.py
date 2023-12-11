import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('log.csv', header = None)


ndx = df.values
ndx = ndx.reshape(1, -1)[0]

print(len(ndx))
print(ndx)

nt = np.where(ndx != -1)[0]

print(nt)  # !!! Note that ordering of this array is not important!!! Its length is VERY IMPORTANT and must be the same every time we run!!!!
print(len(nt)) #!!!! MUST be independent of the run for the same particle distribution!!!!!

