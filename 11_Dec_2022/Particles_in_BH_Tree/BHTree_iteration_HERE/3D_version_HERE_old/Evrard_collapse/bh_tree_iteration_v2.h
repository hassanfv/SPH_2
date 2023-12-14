#ifndef BH_TREE_ITERATION_H
#define BH_TREE_ITERATION_H

//__device__ const int blockSize = 256;
__device__ const int warp = 32;
__device__ const int stackSize = 64;
__device__ const float eps2 = 0.025;
__device__ const float theta = 0.5;

//#define gridSize 4
#define blockSize 256
#define G 1.0

// ==========================================================================================
// CUDA ERROR CHECKING CODE
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) getchar();
   }
}
// ==========================================================================================


//===== reset_arrays_kernel
__global__ void reset_arrays_kernel(int *mutex, float *x, float *y, float *z, float *mass, int *count, int *start, int *sorted, int *child,
                                    int *index, float *left, float *right, float *bottom, float *top, float *front, float *back, int n, int m)
{
    int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    int offset = 0;

    // reset octree arrays
    while(bodyIndex + offset < m)
    {  
        #pragma unroll 8
        for(int i = 0; i < 8; i++)
        {
            child[(bodyIndex + offset) * 8 + i] = -1; // iterates over the 8 child nodes! -1 indicates the node initially has no children!
        }
        if(bodyIndex + offset < n) // if the node is a leaf node.
        {
            count[bodyIndex + offset] = 1;
        }
        else // indicating an empty internal node.
        {
            x[bodyIndex + offset] = 0;
            y[bodyIndex + offset] = 0;
            z[bodyIndex + offset] = 0;
            mass[bodyIndex + offset] = 0;
            count[bodyIndex + offset] = 0;
        }
        start[bodyIndex + offset] = -1;
        sorted[bodyIndex + offset] = 0;
        offset += stride;
    }

    if(bodyIndex == 0)
    {
        *mutex = 0;
        *index = n;
        *left = 0;
        *right = 0;
        *bottom = 0;
        *top = 0;
        *front = 0;
        *back = 0;
    }
}


//===== compute_bounding_box_kernel
__global__ void compute_bounding_box_kernel(int *mutex, float *x, float *y, float *z, 
                                            float *left, float *right, float *bottom, 
                                            float *top, float *front, float *back, int n)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;

    float x_min = x[index];
    float x_max = x[index];
    float y_min = y[index];
    float y_max = y[index];
    float z_min = z[index];
    float z_max = z[index];

    __shared__ float left_cache[blockSize];
    __shared__ float right_cache[blockSize];
    __shared__ float bottom_cache[blockSize];
    __shared__ float top_cache[blockSize];
    __shared__ float front_cache[blockSize];
    __shared__ float back_cache[blockSize];

    int offset = stride;
    while(index + offset < n)
    {
        x_min = fminf(x_min, x[index + offset]);
        x_max = fmaxf(x_max, x[index + offset]);
        y_min = fminf(y_min, y[index + offset]);
        y_max = fmaxf(y_max, y[index + offset]);
        z_min = fminf(z_min, z[index + offset]);
        z_max = fmaxf(z_max, z[index + offset]);
        offset += stride;
    }

    left_cache[threadIdx.x] = x_min;
    right_cache[threadIdx.x] = x_max;
    bottom_cache[threadIdx.x] = y_min;
    top_cache[threadIdx.x] = y_max;
    front_cache[threadIdx.x] = z_min;
    back_cache[threadIdx.x] = z_max;

    __syncthreads();

    int i = blockDim.x/2;
    while(i != 0)
    {
        if(threadIdx.x < i)
        {
            left_cache[threadIdx.x] = fminf(left_cache[threadIdx.x], left_cache[threadIdx.x + i]);
            right_cache[threadIdx.x] = fmaxf(right_cache[threadIdx.x], right_cache[threadIdx.x + i]);
            bottom_cache[threadIdx.x] = fminf(bottom_cache[threadIdx.x], bottom_cache[threadIdx.x + i]);
            top_cache[threadIdx.x] = fmaxf(top_cache[threadIdx.x], top_cache[threadIdx.x + i]);
            front_cache[threadIdx.x] = fminf(front_cache[threadIdx.x], front_cache[threadIdx.x + i]);
            back_cache[threadIdx.x] = fmaxf(back_cache[threadIdx.x], back_cache[threadIdx.x + i]);
        }
        __syncthreads();
        i /= 2;
    }

    if(threadIdx.x == 0)
    {
        while (atomicCAS(mutex, 0 ,1) != 0); // lock
        *left = fminf(*left, left_cache[0]);
        *right = fmaxf(*right, right_cache[0]);
        *bottom = fminf(*bottom, bottom_cache[0]);
        *top = fmaxf(*top, top_cache[0]);
        *front = fminf(*front, front_cache[0]);
        *back = fmaxf(*back, back_cache[0]);
        atomicExch(mutex, 0); // unlock
    }
}


//===== build_tree_kernel
__global__ void build_tree_kernel(float *x, float *y, float *z, float *mass, int *count, int *start, int *child, int *index,
                                  float *left, float *right, float *bottom, float *top, float *front, float *back, int n, int m)
{
	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;
	bool newBody = true;

	// build quadtree
	float l; 
	float r; 
	float b; 
	float t;
	float f; 
	float ba;
	
	int childPath;
	int temp;
	offset = 0;
	while((bodyIndex + offset) < n)
	{
	
		if(newBody)
		{
			newBody = false;

			l = *left;
			r = *right;
			b = *bottom;
			t = *top;
			f = *front;
      ba = *back;			

			temp = 0;
			childPath = 0;
			if(x[bodyIndex + offset] < 0.5*(l+r))
			{
				childPath += 1;
				r = 0.5*(l+r);
			}
			else{
				l = 0.5*(l+r);
			}
			if(y[bodyIndex + offset] < 0.5*(b+t))
			{
				childPath += 2;
				t = 0.5*(t+b);
			}
			else
			{
				b = 0.5*(t+b);
			}
			if(z[bodyIndex + offset] < 0.5f*(f+ba))
      {
          childPath += 4;
          ba = 0.5f*(f+ba);
      }
      else
      {
          f = 0.5f*(f+ba);
      }

		}
		int childIndex = child[temp * 8 + childPath];

		// traverse tree until we hit leaf node
		while(childIndex >= n)
		{
  		temp = childIndex;
			childPath = 0;
			if(x[bodyIndex + offset] < 0.5*(l+r))
			{
				childPath += 1;
				r = 0.5*(l+r);
			}
			else
			{
				l = 0.5*(l+r);
			}
			if(y[bodyIndex + offset] < 0.5*(b+t))
			{
				childPath += 2;
				t = 0.5*(t+b);
			}
			else
			{
				b = 0.5*(t+b);
			}
			if(z[bodyIndex + offset] < 0.5f*(f+ba))
      {
          childPath += 4;
          ba = 0.5f*(f+ba);
      }
      else
      {
          f = 0.5f*(f+ba);
      }

			atomicAdd(&x[temp], mass[bodyIndex + offset]*x[bodyIndex + offset]);
			atomicAdd(&y[temp], mass[bodyIndex + offset]*y[bodyIndex + offset]);
			atomicAdd(&z[temp], mass[bodyIndex + offset]*z[bodyIndex + offset]);
			atomicAdd(&mass[temp], mass[bodyIndex + offset]);
			atomicAdd(&count[temp], 1);
			childIndex = child[8 * temp + childPath];
		}

		if(childIndex != -2)
		{
			int locked = temp * 8 + childPath;
			if(atomicCAS(&child[locked], childIndex, -2) == childIndex)
			{
				if(childIndex == -1)
				{
					child[locked] = bodyIndex + offset;
				}
				else
				{
					int patch = 8 * n;
					while(childIndex >= 0 && childIndex < n)
					{
				 		int cell = atomicAdd(index, 1);
				 		patch = min(patch, cell);
				 		if(patch != cell)
				 		{
				 			child[8 * temp + childPath] = cell;
				 		}

            // insert old particle
				 		childPath = 0;
				 		if(x[childIndex] < 0.5*(l+r))
				 		{
				 			childPath += 1;
            }
				 		if(y[childIndex] < 0.5*(b+t))
				 		{
				 			childPath += 2;
				 		}
				 		if(z[bodyIndex + offset] < 0.5f*(f+ba))
            {
              childPath += 4;
            }
            else
            {
              f = 0.5f*(f+ba);
            }

				 		//if(DEBUG)
				 		if(true)
				 		{
				 			if(cell >= m)
				 			{
				 				printf("%s\n", "error cell index is too large!!");
				 				printf("cell: %d\n", cell);
				 			}
				 		}

				 		x[cell] += mass[childIndex]*x[childIndex];
				 		y[cell] += mass[childIndex]*y[childIndex];
				 		z[cell] += mass[childIndex]*z[childIndex];
				 		mass[cell] += mass[childIndex];
				 		count[cell] += count[childIndex];
				 		child[8 * cell + childPath] = childIndex;

				 		start[cell] = -1;

            // insert new particle
				 		temp = cell;
				 		childPath = 0;
				 		if(x[bodyIndex + offset] < 0.5*(l+r))
				 		{
				 			childPath += 1;
				 			r = 0.5*(l+r);
				 		}
				 		else{
				 			l = 0.5*(l+r);
				 		}
				 		if(y[bodyIndex + offset] < 0.5*(b+t))
				 		{
				 			childPath += 2;
				 			t = 0.5*(t+b);
				 		}
				 		else
				 		{
				 			b = 0.5*(t+b);
				 		}
				 		if(z[bodyIndex + offset] < 0.5f*(f+ba))
            {
                childPath += 4;
                ba = 0.5f*(f+ba);
            }
            else
            {
                f = 0.5f*(f+ba);
            }
				 		x[cell] += mass[bodyIndex + offset]*x[bodyIndex + offset];
				 		y[cell] += mass[bodyIndex + offset]*y[bodyIndex + offset];
				 		z[cell] += mass[bodyIndex + offset]*z[bodyIndex + offset];
				 		mass[cell] += mass[bodyIndex + offset];
				 		count[cell] += count[bodyIndex + offset];
				 		childIndex = child[8 * temp + childPath];				 		
				 	}
				
				 	child[8 * temp + childPath] = bodyIndex + offset;

				 	__threadfence();  // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

				 	child[locked] = patch;

				}

				//__threadfence(); // we have been writing to global memory arrays (child, x, y, mass) thus need to fence

				offset += stride;
				newBody = true;
			}
		}
		__syncthreads(); // not strictly needed 
	}
}





//===== centre_of_mass_kernel
__global__ void centre_of_mass_kernel(float *x, float *y, float *z, float *mass, int *index, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;

	bodyIndex += n;
	while(bodyIndex + offset < *index)
	{
		x[bodyIndex + offset] /= mass[bodyIndex + offset];
		y[bodyIndex + offset] /= mass[bodyIndex + offset];
		z[bodyIndex + offset] /= mass[bodyIndex + offset];

		offset += stride;
	}
}



//===== sort_kernel
__global__ void sort_kernel(int *count, int *start, int *sorted, int *child, int *index, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;

	int s = 0;
	if(threadIdx.x == 0)
	{
		for(int i = 0; i < 8; i++)
		{
			int node = child[i];

			if(node >= n)
			{
			  // not a leaf node
				start[node] = s;
				s += count[node];
			}
			else if(node >= 0)
			{
			  // leaf node
				sorted[s] = node;
				s++;
			}
		}
	}

	int cell = n + bodyIndex;
	int ind = *index;
	while((cell + offset) < ind)
	{
		s = start[cell + offset];
	
		if(s >= 0)
		{
			for(int i = 0; i < 8; i++)
			{
				int node = child[8 * (cell+offset) + i];

				if(node >= n)
				{
				  // not a leaf node
					start[node] = s;
					s += count[node];
				}
				else if(node >= 0)
				{
				  // leaf node
					sorted[s] = node;
					s++;
				}
			}
			offset += stride;
		}
	}
}


//===== compute_forces_kernel
__global__ void compute_forces_kernel(float *x, float *y, float *z, float *ax, float *ay, float *az, float *mass, float *eps,
                                      int *sorted, int *child, float *left, float *right, float *bottom, float *top,
                                      float *front, float *back, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;

	__shared__ float depth[stackSize*blockSize/warp]; 
	__shared__ int stack[stackSize*blockSize/warp];  // stack controled by one thread per warp 

	float rad1 = 0.5*(*right - (*left));
	float rad2 = 0.5*(*bottom - (*top));
	float rad3 = 0.5*(*front - (*back));
	float radius = fmaxf(abs(rad1), fmaxf(abs(rad2), abs(rad3)));
	
	float rr, inv_r3, epsij, q, q2, q3, q4, q5, q6, fk;

	// need this in case some of the first eight entries of child are -1 (otherwise jj = 7)
	int jj = -1;                 
	for(int i = 0; i < 8; i++)
	{       
		if(child[i] != -1)
		{     
			jj++;               
		}                       
	}

	int counter = threadIdx.x % warp;
	int stackStartIndex = stackSize*(threadIdx.x / warp);
	while(bodyIndex + offset < n)
	{
		int sortedIndex = sorted[bodyIndex + offset];

		float pos_x = x[sortedIndex];
		float pos_y = y[sortedIndex];
		float pos_z = z[sortedIndex];
		float acc_x = 0;
		float acc_y = 0;
		float acc_z = 0;

		// initialize stack
		int top = jj + stackStartIndex;
		if(counter == 0)
		{
			int temp = 0;
			for(int i = 0;i < 8; i++)
			{
				if(child[i] != -1)
				{
					stack[stackStartIndex + temp] = child[i];
					depth[stackStartIndex + temp] = radius*radius/theta;
					temp++;
				}
			}
		}

		__syncthreads();

		// while stack is not empty
		while(top >= stackStartIndex)
		{
			int node = stack[top];
			float dp = 0.25*depth[top];
			// float dp = depth[top];
			for(int i = 0; i < 8; i++)
			{
				int ch = child[8*node + i];

				//__threadfence();
			
				if(ch >= 0)
				{
					float dx = x[ch] - pos_x;
  				float dy = y[ch] - pos_y;
  				float dz = z[ch] - pos_z;
    			float r = dx*dx + dy*dy + dz*dz;
    			
    			if(ch < n /*is leaf node*/ || __all_sync(0xffffffff, dp <= r)/*meets criterion*/)
    			{
    			  rr = sqrt(r);
            inv_r3 = 1.0f / (rr * rr * rr + 1e-5);
            epsij = eps[sortedIndex];
            q = rr / epsij;
            q2 = q * q;
            q3 = q2 * q;
            q4 = q3 * q;
            q5 = q4 * q;
            q6 = q5 * q;

            if (q <= 1.0f)
            {
              fk = (1.0f / (epsij * epsij * epsij)) * ((4.0f / 3.0f) - (6.0f / 5.0f) * q2 + (1.0f / 2.0f) * q3);
            }

            if ((q > 1.0f) && (q <= 2.0f))
            {
              fk = inv_r3 * ((-1.0f / 15.0f) + (8.0f / 3.0f) * q3 - 3.0f * q4 + (6.0f / 5.0f) * q5 - (1.0f / 6.0f) * q6);
            }

            if (q > 2.0f)
            {
              fk = inv_r3;
            }

   					acc_x += G * fk * dx * mass[ch];
   					acc_y += G * fk * dy * mass[ch];
   					acc_z += G * fk * dz * mass[ch];
					}
					else
					{
						if(counter == 0)
						{
							stack[top] = ch;
							depth[top] = dp;
							// depth[top] = 0.25*dp;
						}
						top++;
						//__threadfence();
					}
				}
			}

			top--;
		}

		ax[sortedIndex] = acc_x;
   	ay[sortedIndex] = acc_y;
   	az[sortedIndex] = acc_z;

   	offset += stride;

   	__syncthreads();
   	}
}


#endif 
