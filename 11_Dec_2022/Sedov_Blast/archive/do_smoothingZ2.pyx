
import numpy
cimport numpy


ctypedef numpy.double_t DTYPE_t
def do_smoothingZ2(numpy.ndarray[DTYPE_t, ndim=1] x, numpy.ndarray[DTYPE_t, ndim=1] y, numpy.ndarray[DTYPE_t, ndim=1] z):

	cdef int k = 0
	cdef double dx, dy, dz
	cdef numpy.ndarray hres, dist

	cdef int N = len(x)

	hres = numpy.zeros(N*N)

	for i in range(N):

		dist = numpy.zeros(N)

		for j in range(N):

			dx = x[j] - x[i]
			dy = y[j] - y[i]
			dz = z[j] - z[i]
			dist[j] = (dx*dx + dy*dy + dz*dz)**0.5

		hres[k] = numpy.sort(dist)[64]
		k += 1

	return hres * 0.5
