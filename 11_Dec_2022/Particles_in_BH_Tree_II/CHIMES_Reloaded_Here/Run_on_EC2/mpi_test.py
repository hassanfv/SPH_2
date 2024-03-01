# Save this as mpi_test.py
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # Total number of processes
rank = comm.Get_rank()  # The rank of this process

print(f"Hello from rank {rank} out of {size}")

