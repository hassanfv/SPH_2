import numpy as np 
        
# This routine takes an integer, decomposes it into 
# all combinations of two factors, then selects the 
# pair with the minimum range. 
def decompose_2d_factors(x):
    factor_list= []
    range_list = []

    for i in range(1, x + 1):
        if x % i == 0:
            j = x // i
            r = abs(i - j)
            factor_list.append([i, j])
            range_list.append(r) 

    factor_list = np.array(factor_list)
    range_list = np.array(range_list)

    ind_sort = range_list.argsort()

    return factor_list[ind_sort][0] 
