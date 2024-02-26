import pandas as pd
import matplotlib.pyplot as plt

# Read the file into a DataFrame, assuming it's tab-separated
data = pd.read_csv('xxxxx.sed', sep='\t', comment='#', header = None)

print(data.head())

# Define the columns to be used for plotting by their index positions
x = data.iloc[:, 0]  # First column as independent variable
y1 = data.iloc[:, 1]  # Second column as first dependent variable
y2 = data.iloc[:, 4]  # Fifth column as second dependent variable

# Create the plots with logarithmic axes
plt.figure(figsize=(14, 6))

# Plotting the second column as a function of the first column with logarithmic axes
plt.subplot(1, 2, 1)
plt.loglog(x, y1, label='Incident radiation')
plt.xlabel('E [Ryd]')
plt.ylabel('Incident radiation')
plt.legend()

# Plotting the fifth column as a function of the first column with logarithmic axes
plt.subplot(1, 2, 2)
plt.loglog(x, y2, label='Transmitted radiation', color='orange')
plt.xlabel('E [Ryd]')
plt.ylabel('Transmitted radiation')
plt.legend()

plt.tight_layout()

#plt.savefig('sed_Notextinguished.png')
plt.savefig('sed_Extinguished.png')

plt.show()



