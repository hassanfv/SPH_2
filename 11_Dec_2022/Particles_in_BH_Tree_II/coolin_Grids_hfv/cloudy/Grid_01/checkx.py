# Python code to read the .het file and plot the heating rates vs temperature for each temperature value separately

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the file
file_path = 'test3.het'

# Read the file with custom parsing due to its complex structure
# Define a function to parse each line
def parse_line(line):
    parts = line.split()
    if len(parts) < 3:
        return None  # Skip lines that don't have enough data
    try:
        temp = float(parts[1])  # Temperature is the second item
        heating_rate = float(parts[2])  # Heating rate is the third item
        cooling_rate = float(parts[3])  # Heating rate is the third item
        return temp, heating_rate, cooling_rate
    except ValueError:
        return None

# Parse the file
temps = []
heating_rates = []
cooling_rates = []
with open(file_path, 'r') as file:
    for line in file:
        parsed = parse_line(line)
        if parsed:
            temps.append(parsed[0])
            heating_rates.append(parsed[1])
            cooling_rates.append(parsed[2])


nH = 3.0 # in log
nH = 10**nH

# Convert to pandas DataFrame for easier handling

heating_rates = [tmp/nH/nH for tmp in heating_rates]
cooling_rates = [tmp/nH/nH for tmp in cooling_rates]

df = pd.DataFrame({
    'Temperature': np.log10(temps),
    'HeatingRate': np.log10(heating_rates),
    'CoolingRate': np.log10(cooling_rates)
})


df.to_csv('cloudyHCool.csv', index = False)


# Group by temperature and plot each group
grouped = df.groupby('Temperature')


plt.figure(figsize=(12, 8))
for name, group in grouped:
    plt.plot(group['Temperature'], group['HeatingRate'], marker='o', linestyle='', label=f'Temp {name} K')
    plt.plot(group['Temperature'], group['CoolingRate'], marker='*', linestyle='', label=f'Temp {name} K')
    

plt.xlabel('Temperature (K)')
plt.ylabel('Heating Rate (erg/cm³/s)')
plt.title('Heating Rates vs Temperature for Each Temperature Value')
#plt.legend()
plt.grid(True)

plt.savefig('HCool.png')

plt.show()

