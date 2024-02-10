import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('B87_from_chimes.csv')

E = df['E_ryd'].values
Jnu = df['J_nu'].values

plt.scatter(E, Jnu, s = 10)
plt.show()


