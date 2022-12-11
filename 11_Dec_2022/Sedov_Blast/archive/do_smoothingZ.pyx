
import numpy
cimport numpy
#from quicksort import sort as qsort


ctypedef numpy.double_t DTYPE_t
def do_smoothingZ(numpy.ndarray[DTYPE_t, ndim=2] pos):

	cdef int k = 0
	cdef double dx, dy, dz
	cdef numpy.ndarray hres, dist

	cdef int N = pos.shape[0]

	hres = numpy.zeros(N*N)

	for i in range(N):

		dist = numpy.empty(N)

		for j in range(N):

			dx = pos[j, 0] - pos[i, 0]
			dy = pos[j, 1] - pos[i, 1]
			dz = pos[j, 2] - pos[i, 2]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5

		hres[k] = numpy.sort(dist)[64]
		#qsort(dist)
		#hres[k] = qsort(dist)[64]
		k += 1

	return hres * 0.5










