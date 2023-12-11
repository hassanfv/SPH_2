%%writefile test.cu

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <chrono>


using namespace std;


#define gridSize 8
#define blockSize 8

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
__global__ void reset_arrays_kernel(int *mutex, float *x, float *y, float *mass, int *count, int *start, int *sorted, int *child,
                                    int *index, float *left, float *right, float *bottom, float *top, int n, int m)
{
	int bodyIndex = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	int offset = 0;

	// reset quadtree arrays
	while(bodyIndex + offset < m)
	{  
#pragma unroll 4
		for(int i = 0; i < 4; i++)
		{
			child[(bodyIndex + offset)*4 + i] = -1; // iterates over the 4 child nodes! -1 indicates the node initially has no children!
		}
		if(bodyIndex + offset < n) // if the node is a leaf node.
		{
			count[bodyIndex + offset] = 1;
		}
		else // indicating an empty internal node.
		{
			x[bodyIndex + offset] = 0;
			y[bodyIndex + offset] = 0;
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
	}
}



//===== compute_bounding_box_kernel
__global__ void compute_bounding_box_kernel(int *mutex, float *x, float *y, float *left, float *right, float *bottom, float *top, int n)
{
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	float x_min = x[index];
	float x_max = x[index];
	float y_min = y[index];
	float y_max = y[index];
	
	__shared__ float left_cache[blockSize];
	__shared__ float right_cache[blockSize];
	__shared__ float bottom_cache[blockSize];
	__shared__ float top_cache[blockSize];


	int offset = stride;
	while(index + offset < n)
	{
		x_min = fminf(x_min, x[index + offset]);
		x_max = fmaxf(x_max, x[index + offset]);
		y_min = fminf(y_min, y[index + offset]);
		y_max = fmaxf(y_max, y[index + offset]);
		offset += stride;
	}

	left_cache[threadIdx.x] = x_min;
	right_cache[threadIdx.x] = x_max;
	bottom_cache[threadIdx.x] = y_min;
	top_cache[threadIdx.x] = y_max;

	__syncthreads();

	// assumes blockDim.x is a power of 2!
	int i = blockDim.x/2;
	while(i != 0)
	{
		if(threadIdx.x < i)
		{
			left_cache[threadIdx.x] = fminf(left_cache[threadIdx.x], left_cache[threadIdx.x + i]);
			right_cache[threadIdx.x] = fmaxf(right_cache[threadIdx.x], right_cache[threadIdx.x + i]);
			bottom_cache[threadIdx.x] = fminf(bottom_cache[threadIdx.x], bottom_cache[threadIdx.x + i]);
			top_cache[threadIdx.x] = fmaxf(top_cache[threadIdx.x], top_cache[threadIdx.x + i]);
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
		atomicExch(mutex, 0); // unlock
	}
}




//===== build_tree_kernel
__global__ void build_tree_kernel(float *x, float *y, float *mass, int *count, int *start, int *child, int *index,
                                  float *left, float *right, float *bottom, float *top, int n, int m)
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
			
			//!!!!!!!!!!!!!!!!!!!!! DANGER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			l = 0.0f;
			r = 1.0f;
			b = 0.0f;
			t = 1.0f;
			//!!!!!!!!!!!!!!!!!!!!! DANGER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			

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

		}
		int childIndex = child[temp*4 + childPath];

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

			atomicAdd(&x[temp], mass[bodyIndex + offset]*x[bodyIndex + offset]);
			atomicAdd(&y[temp], mass[bodyIndex + offset]*y[bodyIndex + offset]);
			atomicAdd(&mass[temp], mass[bodyIndex + offset]);
			atomicAdd(&count[temp], 1);
			childIndex = child[4*temp + childPath];
		}

		if(childIndex != -2)
		{
			int locked = temp*4 + childPath;
			if(atomicCAS(&child[locked], childIndex, -2) == childIndex)
			{
				if(childIndex == -1)
				{
					child[locked] = bodyIndex + offset;
				}
				else
				{
					//int patch = 2 * n;
					int patch = 4 * n;
					while(childIndex >= 0 && childIndex < n)
					{
				 		int cell = atomicAdd(index, 1);
				 		patch = min(patch, cell);
				 		if(patch != cell)
				 		{
				 			child[4*temp + childPath] = cell;
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

				 		//if(DEBUG)
				 		if(true)
				 		{
				 			// if(cell >= 2*n){
				 			if(cell >= m)
				 			{
				 				printf("%s\n", "error cell index is too large!!");
				 				printf("cell: %d\n", cell);
				 			}
				 		}

				 		x[cell] += mass[childIndex]*x[childIndex];
				 		y[cell] += mass[childIndex]*y[childIndex];
				 		mass[cell] += mass[childIndex];
				 		count[cell] += count[childIndex];
				 		child[4*cell + childPath] = childIndex;

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
				 		x[cell] += mass[bodyIndex + offset]*x[bodyIndex + offset];
				 		y[cell] += mass[bodyIndex + offset]*y[bodyIndex + offset];
				 		mass[cell] += mass[bodyIndex + offset];
				 		count[cell] += count[bodyIndex + offset];
				 		childIndex = child[4*temp + childPath];				 		
				 	}
				
				 	child[4*temp + childPath] = bodyIndex + offset;

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
__global__ void centre_of_mass_kernel(float *x, float *y, float *mass, int *index, int n)
{
	int bodyIndex = threadIdx.x + blockIdx.x*blockDim.x;
	int stride = blockDim.x*gridDim.x;
	int offset = 0;

	bodyIndex += n;
	while(bodyIndex + offset < *index)
	{
		x[bodyIndex + offset] /= mass[bodyIndex + offset];
		y[bodyIndex + offset] /= mass[bodyIndex + offset];

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
		for(int i = 0; i < 4; i++)
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
			for(int i=0; i<4; i++)
			{
				int node = child[4*(cell+offset) + i];

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






int main()
{

  int numParticles;
	int numNodes;

	float *h_left;
	float *h_right;
	float *h_bottom;
	float *h_top;

	float *h_mass;
	float *h_x;
	float *h_y;
	float *h_ax;
	float *h_ay;

	int *h_child;
	int *h_start;
	int *h_sorted;
	int *h_count;

	float *d_left;
	float *d_right;
	float *d_bottom;
	float *d_top;
	
	float *d_mass;
	float *d_x;
	float *d_y;
	float *d_ax;
	float *d_ay;
	
	int *d_index;
	int *d_child;
	int *d_start;
	int *d_sorted;
	int *d_count;

	int *d_mutex;  //used for locking 

	cudaEvent_t start, stop; // used for timing

	float *h_output;  //host output array for visualization
	float *d_output;  //device output array for visualization

  //parameters = p;
	//step = 0;
	
	int n = pow(2, 6);
	
	numParticles = n;
	numNodes = 4 * n + 500;
	
	int m = numNodes;

	// allocate host data
	h_left = new float;
	h_right = new float;
	h_bottom = new float;
	h_top = new float;
	h_mass = new float[numNodes];
	h_x = new float[numNodes];
	h_y = new float[numNodes];
	h_ax = new float[numNodes];
	h_ay = new float[numNodes];
	h_child = new int[4*numNodes];
	h_start = new int[numNodes];
	h_sorted = new int[numNodes];
	h_count = new int[numNodes];
	h_output = new float[2*numNodes];

	// allocate device data
	gpuErrchk(cudaMalloc((void**)&d_left, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_right, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_bottom, sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_top, sizeof(float)));
	gpuErrchk(cudaMemset(d_left, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_right, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_bottom, 0, sizeof(float)));
	gpuErrchk(cudaMemset(d_top, 0, sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_mass, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_x, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_y, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ax, numNodes*sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&d_ay, numNodes*sizeof(float)));

	gpuErrchk(cudaMalloc((void**)&d_index, sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_child, 4*numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
	gpuErrchk(cudaMalloc((void**)&d_mutex, sizeof(int))); 

	gpuErrchk(cudaMemset(d_start, -1, numNodes*sizeof(int)));
	gpuErrchk(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

	int memSize = sizeof(float) * 2 * numParticles;

	gpuErrchk(cudaMalloc((void**)&d_output, 2*numNodes*sizeof(float)));


    
  reset_arrays_kernel<<< gridSize, blockSize >>>(d_mutex, d_x, d_y, d_mass, d_count, d_start, d_sorted, d_child, d_index,
                                                 d_left, d_right, d_bottom, d_top, n, m);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
  }


  // initializing x, y, mass -----
  mt19937 engine(42);
  uniform_real_distribution<float> distribution(0.0, 1.0);
  
  
  //float xtmp[] = {0.1, 0.1, 0.7, 0.7, 0.4, 0.6};
  //float ytmp[] = {0.1, 0.7, 0.1, 0.7, 0.3, 0.6};
  
  for (int i = 0; i < numParticles; i++)
  {
    //h_x[i] = xtmp[i]; // distribution(engine);
    //h_y[i] = ytmp[i]; //distribution(engine);
    h_x[i] = distribution(engine);
    h_y[i] = distribution(engine);
    
    //cout << h_x[i] << "," << h_y[i] << "," << i << endl;
    
    h_mass[i] = 0.5f;
  }
  
  
  cudaMemcpy(d_x, h_x, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass, h_mass, numNodes * sizeof(float), cudaMemcpyHostToDevice);


  // After the kernel call and cudaDeviceSynchronize()
  compute_bounding_box_kernel<<< gridSize, blockSize >>>(d_mutex, d_x, d_y, d_left, d_right, d_bottom, d_top, n);
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
  }

  
  cudaMemcpy(h_left, d_left, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_right, d_right, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_bottom, d_bottom, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_top, d_top, sizeof(float), cudaMemcpyDeviceToHost);
  
  int *h_index = new int;
  cudaMemcpy(h_index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
  printf("\n");
  printf("h_left, h_right, h_bottom, h_top = %f, %f, %f, %f\n", h_left[0], h_right[0], h_bottom[0], h_top[0]);
  printf("\n");
  printf("initial index = %d\n", h_index[0]);
  printf("\n");
  


  auto T_Filling = std::chrono::high_resolution_clock::now();
  //gridSize, blockSize
  
  build_tree_kernel<<< 1, 1 >>>(d_x, d_y, d_mass, d_count, d_start, d_child, d_index, d_left, d_right, d_bottom, d_top, n, m);
  cudaDeviceSynchronize();
  
  auto end_Filling = std::chrono::high_resolution_clock::now();
  auto elapsed_Filling = std::chrono::duration_cast<std::chrono::nanoseconds>(end_Filling - T_Filling);
  cout << "Elapsed time = " << elapsed_Filling.count() * 1e-9 << endl;
  

  cudaMemcpy(h_index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
  printf("\n");
  printf("Final index = %d\n", h_index[0]);
  printf("\n");
  
  
  cudaMemcpy(h_x, d_x, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_y, d_y, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < numNodes; i++)
  {
    cout << "i, CM = " << i << ", " << h_x[i] << ", " << h_y[i] << endl;
  }
  
  /*
  cudaMemcpy(h_count, d_count, numNodes * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numNodes; i++)
  {
    cout << "count[" << i << "] = " << h_count[i] << endl;
  }
  */

  
  /*
  cudaMemcpy(h_child, d_child, 4 * numNodes * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numNodes; i++)
  {
    cout << i << ", " << h_child[i] << endl;
    //cout << "child[" << i << "] = " << h_child[i] << endl;
  }
  */



}



