


// Assuming BLOCK_DIM is defined and is the number of threads per block.
__global__ void maxCoord(const FLOAT3* dev_unc_pos, FLOAT* dev_result, int num_points)
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




