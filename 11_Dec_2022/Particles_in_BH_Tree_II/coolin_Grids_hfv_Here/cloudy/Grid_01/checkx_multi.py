import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
file_path = 'test3.het'
data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None)

# Correcting the columns based on user input
temperature = np.log10(data[1])
heating_rate = np.log10(data[2])

# Identifying the indexes where the temperature value resets
indexes = temperature[temperature.diff() < 0].index

# Splitting the data into separate sequences
sequences = []
start_idx = 0
for end_idx in indexes:
    sequences.append((temperature[start_idx:end_idx], heating_rate[start_idx:end_idx]))
    start_idx = end_idx
# Adding the last sequence
sequences.append((temperature[start_idx:], heating_rate[start_idx:]))

# Plotting each sequence
plt.figure(figsize=(12, 8))
for i, (temp, heat) in enumerate(sequences):
    plt.plot(temp, heat, label=f'Sequence {i + 1}')

plt.xlabel('Temperature (K)')
plt.ylabel('Heating Rate')
plt.title('Heating Rate vs Temperature for Each Sequence')
plt.legend()
plt.grid(True)
plt.show()

