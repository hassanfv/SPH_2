%%writefile test.cu
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>

using namespace std;

#define BLOCK_DIM 256
typedef float FLOAT;


const int N = 1000000;

struct FLOAT3
{
  float x, y, z;
};

// Define a helper function to generate random floats in [-1, 1]
float random_float()
{
    return 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
}



// Assuming BLOCK_DIM is defined and is the number of threads per block.
__global__ void maxCoord(const FLOAT3 *dev_unc_pos, FLOAT *dev_result, int num_points)
{
  extern __shared__ FLOAT shared_max[]; // Shared memory for inter-thread communication within a block
  int tid = threadIdx.x; // Thread ID within the block
  int i = blockIdx.x * blockDim.x + tid; // Global index for the entire grid
  
  FLOAT max_val = -FLT_MAX; // Initialize with the smallest float value

  // Iterate over all elements assigned to this thread in the grid, striding by the total number of threads
  while(i < num_points)
  {
    FLOAT3 pos = dev_unc_pos[i]; // Get the position at index i
    // Determine the maximum value across all dimensions
    FLOAT max_coord = fmaxf(fmaxf(pos.x, pos.y), pos.z);
    max_val = fmaxf(max_val, max_coord); // Update the current max_val
    i += blockDim.x * gridDim.x; // Move to the next element
  }

  // Store the found max value in shared memory
  shared_max[tid] = max_val;

  __syncthreads(); // Synchronize threads within the block

  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
    }
    __syncthreads(); // Ensure all accesses to shared memory have completed
  }

  // The first thread in the block writes the result to the global memory
  if (tid == 0)
  {
    dev_result[blockIdx.x] = shared_max[0];
  }
}



int main()
{

  int nbead = N;

  dim3 threads(BLOCK_DIM, 1, 1);

  dim3 grid((int) ceil((nbead + 1.0)/(FLOAT) threads.x), 1, 1);

  int GRID_DIM = grid.x;

  FLOAT *dev_result;
  
  FLOAT result[GRID_DIM];
  
  FLOAT init_value = -100;
  for (int i = 0; i < GRID_DIM; i++)
  {
    result[i] = init_value;
  }

  cudaMalloc((void **)&dev_result, GRID_DIM*sizeof(FLOAT));
  
  cudaMemcpy(dev_result, result, GRID_DIM * sizeof(FLOAT), cudaMemcpyHostToDevice);


  cout << "threads.x = " << threads.x << endl;
  cout << "grid.x = " << grid.x << endl;

  srand(time(0)); // Seed for random number generation

  // Allocate host array
  FLOAT3* unc_pos = new FLOAT3[N];

  // Initialize the array with random numbers
  for (int i = 0; i < N; ++i)
  {
    unc_pos[i].x = random_float();
    unc_pos[i].y = random_float();
    unc_pos[i].z = random_float();
  }

  // Allocate memory on the device
  FLOAT3* dev_unc_pos;
  cudaMalloc((void**)&dev_unc_pos, N * sizeof(FLOAT3));

  // Copy data from host to device
  cudaMemcpy(dev_unc_pos, unc_pos, N * sizeof(FLOAT3), cudaMemcpyHostToDevice);


  maxCoord<<<grid, threads, GRID_DIM*sizeof(FLOAT)>>>(dev_unc_pos, dev_result, nbead);
  cudaDeviceSynchronize();
  
  cudaMemcpy(result, dev_result, GRID_DIM*sizeof(FLOAT), cudaMemcpyDeviceToHost);

  // Perform the final reduction on the host
  FLOAT global_max = init_value;
  for (int i = 0; i < grid.x; ++i)
  {
    global_max = max(global_max, result[i]);
  }
  cout << "Global max = " << global_max << endl;

  cudaFree(dev_unc_pos);
  delete[] unc_pos;

}

