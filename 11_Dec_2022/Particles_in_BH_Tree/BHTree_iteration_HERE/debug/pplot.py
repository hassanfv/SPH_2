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

# Annotate each point with its ndx value
for i in range(len(x)):
    plt.text(x[i], y[i], str(ndx[i]), fontsize=11, ha='right', va='bottom')

plt.axvline(x = 0.5, linestyle = '--', color = 'k')
plt.axhline(y = 0.5, linestyle = '--', color = 'k')

plt.axvline(x = 0.25, linestyle = '--', color = 'b')
plt.axhline(y = 0.25, linestyle = '--', color = 'b')

plt.axvline(x = 0.75, linestyle = '--', color = 'b')
plt.axhline(y = 0.75, linestyle = '--', color = 'b')

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.show()

