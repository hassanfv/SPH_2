


def quick_sort(A):
    # partitioning array
    n = len(A)  # size/length of array i.e. input 
    p = 0       # pointer on pivot - suppose 1st element is our pivot
    i = p + 1   # pointer on partition border i.e. next possible swap

    for j in range(1,n):   
        if A[p] > A[j]:
            A[i], A[j] = A[j], A[i]
            i += 1

    A[p], A[i-1] = A[i-1], A[p]

    # recursive call on left side of pivot (smaller than pivot)
    if len(A[0:(i-1)]) > 1:
        quick_sort(A[0:(i-1)])

    # recursive call on right side of pivot (greater than pivot)    
    if len(A[(i-1):]) > 1:
        quick_sort(A[i:])

    return A
