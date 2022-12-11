
import numpy
cimport numpy
#from quicksort import sort as qsort
from numba import njit


ctypedef numpy.double_t DTYPE_t
def get_dUx(numpy.ndarray[DTYPE_t, ndim=2] pos, numpy.ndarray[DTYPE_t, ndim=2] v, numpy.ndarray[DTYPE_t, ndim=1] rho, numpy.ndarray[DTYPE_t, ndim=1] P,
	numpy.ndarray[DTYPE_t, ndim=2] PIij, numpy.ndarray[DTYPE_t, ndim=2] gWx, numpy.ndarray[DTYPE_t, ndim=2] gWy,
	numpy.ndarray[DTYPE_t, ndim=2] gWz, numpy.ndarray[DTYPE_t, ndim=1] h, numpy.ndarray[DTYPE_t, ndim=1] m,
	double gama, double eta, double alpha, double beta):

	cdef int N = pos.shape[0]
	
	cdef numpy.ndarray dudt = numpy.zeros(N)
	cdef double vxij, vyij, vzij
	
	cdef double du_t, vij_gWij
	cdef int i, j

	for i in range(N):
		du_t = 0.0
		for j in range(N):
		
			vxij = v[i, 0] - v[j, 0]
			vyij = v[i, 1] - v[j, 1]
			vzij = v[i, 2] - v[j, 2]
			
			vij_gWij = vxij*gWx[i][j] + vyij*gWy[i][j] + vzij*gWz[i][j]
			
			du_t += m[j] * (P[i]/rho[i]**2 + PIij[i][j]/2.) * vij_gWij

		dudt[i] = du_t
		
	return dudt









