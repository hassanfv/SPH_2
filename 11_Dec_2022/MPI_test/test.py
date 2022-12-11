
import numpy
from mpi4py import MPI
import time
import os


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
PID = os.getpid()

#print(comm)
#print(rank)
#print(size)
#print(PID)

print(f'rank: {rank} has PID: {PID}.')


