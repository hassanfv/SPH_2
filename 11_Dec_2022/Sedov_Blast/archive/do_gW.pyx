
import numpy
cimport numpy


ctypedef numpy.double_t DTYPE_t
def gradW_Ix(numpy.ndarray[DTYPE_t, ndim=2] pos, numpy.ndarray[DTYPE_t, ndim=1] h):

	#pos = posz[0]
	#h = posz[1]
	cdef int N = pos.shape[0]
	
	cdef numpy.ndarray[DTYPE_t, ndim=2] gWx
	cdef numpy.ndarray[DTYPE_t, ndim=2] gWy
	cdef numpy.ndarray[DTYPE_t, ndim=2] gWz
	
	gWx = numpy.zeros((N, N))
	gWy = numpy.zeros((N, N))
	gWz = numpy.zeros((N, N))
	
	cdef int i, j
	cdef double dx, dy, dz, rr, sig, hij, q, nW

	for i in range(N):
		for j in range(N):

			dx = pos[i, 0] - pos[j, 0]
			dy = pos[i, 1] - pos[j, 1]
			dz = pos[i, 2] - pos[j, 2]
			rr = (dx**2 + dy**2 + dz**2)**0.5

			sig = 1.0/numpy.pi
			hij = 0.5 * (h[i] + h[j])
			q = rr / hij
			
			if q <= 1.0:
				nW = sig/hij**5 * (-3.0 + 9.0/4.0 * q)
				gWx[i][j] = nW * dx
				gWy[i][j] = nW * dy
				gWz[i][j] = nW * dz

			if (q > 1.0) & (q <= 2.0):
				nW = -3.0*sig/4.0/hij**5 * (2.0 - q)**2 / (q+1e-20)
				gWx[i][j] = nW * dx
				gWy[i][j] = nW * dy
				gWz[i][j] = nW * dz

	return gWx, gWy, gWz






