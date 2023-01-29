#----- div_curlV ------
divV = 0.0
curlV = 0.0
local_div_curlV = div_curlVel_mpi(nbeg, nend, r, v, rho, m, h)

if rank == 0:
	divV, curlV = local_div_curlV
	for i in range(1, nCPUs):
		divV_tmp, curlV_tmp = comm.recv(source = i)
		divV = np.concatenate((divV, divV_tmp))
		curlV = np.concatenate((curlV, curlV_tmp))
else:
	comm.send(local_div_curlV, dest = 0)

divV = comm.bcast(divV, root = 0)
curlV = comm.bcast(curlV, root = 0)
#----------------------

