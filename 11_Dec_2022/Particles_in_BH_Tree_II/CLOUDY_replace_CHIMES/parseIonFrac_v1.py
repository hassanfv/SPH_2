import numpy as np
import re
import pandas as pd

def ionFractions(file_path, nelec):

  # Function to extract numbers from a line, ignoring non-numeric characters after the numbers
  def extract_numbers(line):
      line = re.sub(r'^[A-Za-z]+', '', line)  # Remove element name
      line = re.sub(r'\([^\)]*\).*', '', line)  # Remove everything after first non-numeric character
      numbers_str = re.findall(r'-?\d+\.\d+', line)  # Extract numbers, handling cases with no spaces
      return [float(num) for num in numbers_str]

  # Initialize the numpy array with dimensions (30, 17) filled with -30.0
  ionization_array = np.full((30, 17), -30.0)

  # Read the file and process each line
  with open(file_path, 'r') as file:
      lines = file.readlines()
      element_count = 0  # Counter for the number of elements processed

      for line in lines:
          if not line.strip() or line.startswith('#') or element_count >= 30:
              continue  # Skip empty lines, comments, and stop after 30 elements

          numbers = extract_numbers(line)
          # Fill the corresponding row in the array with extracted numbers
          ionization_array[element_count, :len(numbers)] = numbers
          element_count += 1  # Increment the element counter

  # The ionization_array is now populated with the extracted numbers
  # You can now use this array for further analysis or processing as needed

  df = pd.DataFrame(ionization_array)
  #print(df)

  ElmId = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 
           'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
           'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

  SolAbund = [12.0, 10.93, 1.05, 1.38, 2.70, 8.43, 7.83, 8.69, 4.56, 7.93,
              6.24, 7.60, 6.45, 7.51, 5.41, 7.12, 5.50, 6.40, 5.03, 6.34,
              3.15, 4.95, 3.93, 5.64, 5.43, 7.50, 4.99, 6.22, 4.19, 4.56]

  SolAbund = np.array(SolAbund)
  SolAbund = SolAbund - 12.0

  Z = -1.0 # metallicity

  ElmAbund = 10**Z * 10**SolAbund
  ElmAbund[0] = 10**0.0 # Correcting for H as only metal abundances scale with Z!
  ElmAbund[1] = 10**(-1.07) # Also correcting for He

  ElmAbund = ElmAbund[:, np.newaxis] #---> convert to column vector
  ionFrac = 10**ionization_array * ElmAbund #---> Now each row represents the elemental abundance relative to H (similar to what CHIMES outputs)!
  
  #----> Mean molecular weight (mu) <------
  AtomicNumber = np.arange(1, 31)
  AtomicNumber = AtomicNumber[:, np.newaxis]
  
  njAj = ionFrac * AtomicNumber
  
  mu = np.sum(njAj) / (np.sum(ionFrac) + nelec)

  
  return ionFrac, mu






