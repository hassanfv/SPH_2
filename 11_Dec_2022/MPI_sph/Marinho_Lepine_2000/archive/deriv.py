
import numpy as np
import pandas as pd


def func(x):

	return np.exp(2.*x)



def deriv_h(x, h):

	df = (func(x+h) - func(x)) / h

	return df


def deriv_h2(x, h):

	df = (func(x+h) - func(x-h)) / 2. / h

	return df


x = np.arange(1., 5., 0.16)


h = 0.1

res = []

for xt in x:

	df1 = deriv_h(xt, h)
	df2 = deriv_h2(xt, h)
	
	res.append([xt, 2.*func(xt), df1, df2])


df = pd.DataFrame(res)
df.columns = ['x', 'Analytical', 'df1', 'df2']

print(df)
