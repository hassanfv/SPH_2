
import numpy as np
import pickle


with open('data.pkl', 'rb') as f:
  lst = pickle.load(f)

#lst = [round(_, 2) for _ in lst]

print(lst)



