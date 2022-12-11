

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

a = 2.0
b = 3.0

if rank == 0:
	print(f'Result from process {rank}: {a * b}')

if rank == 1:
	print(f'Result from process {rank}: {a + b}')

if rank == 3:
	print(f'Result from process {rank}: {max(a, b)}')




