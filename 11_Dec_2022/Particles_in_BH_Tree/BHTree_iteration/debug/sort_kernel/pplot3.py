import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('data.csv')

# Extract x, y, and ndx values
x = df['x']
y = df['y']
ndx = df['ndx']


# Create the scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(x, y, s=30, color='k')

#----- CM of one particle -----
#cm = [0.791765, 0.417585]
#plt.scatter([cm[0]], [cm[1]], s = 20, color = 'red')
#------------------------------

# Annotate each point with its ndx value
for i in range(len(x)):
    plt.text(x[i], y[i], str(ndx[i]), fontsize=11, ha='right', va='bottom')

'''
for i in range(16+1):
  plt.axvline(x = i*0.0625, linestyle = '--', color = 'pink', linewidth = 1)
  plt.axhline(y = i*0.0625, linestyle = '--', color = 'pink', linewidth = 1)
'''

for i in range(8+1):
  plt.axvline(x = i*0.125, linestyle = '--', color = 'orange', linewidth = 2)
  plt.axhline(y = i*0.125, linestyle = '--', color = 'orange', linewidth = 2 )

plt.axvline(x = 0.5, linestyle = '--', color = 'k', linewidth = 4)
plt.axhline(y = 0.5, linestyle = '--', color = 'k', linewidth = 4)

plt.axvline(x = 0.25, linestyle = '--', color = 'b', linewidth = 3)
plt.axhline(y = 0.25, linestyle = '--', color = 'b', linewidth = 3)
plt.axvline(x = 0.75, linestyle = '--', color = 'b', linewidth = 3)
plt.axhline(y = 0.75, linestyle = '--', color = 'b', linewidth = 3)


plt.xlim(0, 1)
plt.ylim(0, 1)

plt.show()

