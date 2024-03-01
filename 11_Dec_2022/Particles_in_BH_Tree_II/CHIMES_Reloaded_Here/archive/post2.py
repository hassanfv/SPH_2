
import numpy as np
import pickle


with open('CHIMES_EC2_hfvx.pkl', 'rb') as f:
  df = pickle.load(f)

Tres = df['Tres']
uRes = df['uRes']
muRes = df['muRes']
AbRes = df['AbRes']

print(AbRes)


