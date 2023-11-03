
import numpy as np


arr = np.array([-2, 3, 4, 7, 8, 9, 11, 13])


def searchx(arr, target, L = 0, R = None):

  if R is None:
    R = len(arr) - 1
  
  if R < L:
    return -1
  
  mid = int((L + R)/2)

  if target == arr[mid]:
    return mid
  
  elif target < arr[mid]:
    return searchx(arr, target, L, mid-1)
  
  elif target > arr[mid]:
    return searchx(arr, target, mid + 1, R)


target = -4

print(searchx(arr, target))


