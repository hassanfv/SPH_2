#ifndef NGB_H
#define NGB_H


struct Cell {
  int row;
  int col;
  float xcen;
  float ycen;
  int start = -1;
  int end = -1;
};


//===== maxCoord
__global__ void maxCoord(float *d_x, float *d_y, float *d_z, int *d_Typ, float *d_max, int N)
{
    extern __shared__ float shared_max[]; // Shared memory for inter-thread communication within a block
    int tid = threadIdx.x; // Thread ID within the block
    int i = blockIdx.x * blockDim.x + tid; // Global index for the entire grid
    
    float max_val = -1000000.0; // Initialize with the smallest float value

    // Iterate over all elements assigned to this thread in the grid, striding by the total number of threads
    while((i < N) && (d_Typ[i] != -1))
    {
        // Determine the maximum value across all dimensions
        float max_coord = fmaxf(d_x[i], fmaxf(d_y[i], d_z[i]));
        max_val = fmaxf(max_val, max_coord);
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
        atomicMax((int*)d_max, __float_as_int(shared_max[0]));
    }
}



//===== getCelliD
__host__ __device__ int getCelliD(float x, float y, float z, float x_min, float y_min, float z_min, float Wcell, int nSplit)
{
    int col = static_cast<int>((x - x_min) / Wcell);
    int row = static_cast<int>((y - y_min) / Wcell);
    int depth = static_cast<int>((z - z_min) / Wcell);

    return depth * nSplit * nSplit + row * nSplit + col;
}


//===== CountBodies
__global__ void CountBodies(float *d_x, float *d_y, float *d_z, float x_min, float y_min, float z_min, float Wcell, int nSplit, int *count, int Ncell, int N)
{
  
  int tx = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (tx < Ncell)
    count[tx] = 0;
  __syncthreads();

  for (int i = tx; i < N; i += blockDim.x)
  {
    int cell_iD = getCelliD(d_x[i], d_y[i], d_z[i], x_min, y_min, z_min, Wcell, nSplit);
    atomicAdd(&count[cell_iD], 1);
  }
  __syncthreads();
}



//===== ComputeOffset
__global__ void ComputeOffset(int *count, int Ncell)
{
    int tx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tx < Ncell)
    {
        int offset = 0;
        for (int i = 0; i < tx; i++)
        {
            offset += count[i];
        }
        count[tx + Ncell] = offset;
    }
    __syncthreads();
}


//===== GroupBodies
__global__ void GroupBodies(float *d_x, float *d_y, float *d_z, int *d_groupedIndex, float x_min, float y_min, float z_min,
                            float Wcell, int nSplit, int *count, int Ncell, int N)
{
  int *offsets = &count[Ncell];
  for (int i = threadIdx.x; i < N; i += blockDim.x)
  {
    int cell_iD = getCelliD(d_x[i], d_y[i], d_z[i], x_min, y_min, z_min, Wcell, nSplit);
    int dest = atomicAdd(&offsets[cell_iD], 1);
    d_groupedIndex[dest] = i;
  }
  __syncthreads();
}



//===============================================
//================ ngbDB_new_v2 =================
//===============================================
__global__ void ngbDB_new_v1(int *Typ, float *x, float *y, float *z, float *h,
                             float x_min, float y_min, float z_min, float W_cell,
                             int nSplit, int *offSet, int *groupedIndex,
                             int *ngb, int MAX_ngb, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    //--------- Finding the neighboring cells -------------
    int neighbors[27]; // highly likely will stay in the register of the thread!
    
    float x_i = x[i];
    float y_i = y[i];
    float z_i = z[i];

    int cell_id = getCelliD(x_i, y_i, z_i, x_min, y_min, z_min, W_cell, nSplit);
    
    // Convert linear index (i.e. cell_id) to 3D coordinates
    int zcell = cell_id / (nSplit * nSplit);
    int ycell = (cell_id / nSplit) % nSplit;
    int xcell = cell_id % nSplit;
    
    int count = 0;  // Count of valid neighbors

    // Loop through all neighboring coordinates
    for (int dz = -1; dz <= 1; dz++)
    {
      for (int dy = -1; dy <= 1; dy++)
      {
        for (int dx = -1; dx <= 1; dx++)
        {
          int nx = xcell + dx;
          int ny = ycell + dy;
          int nz = zcell + dz;

          // Check if the neighbor is within bounds
          if (nx >= 0 && nx < nSplit && ny >= 0 && ny < nSplit && nz >= 0 && nz < nSplit)
          {
            // Convert 3D coordinates back to linear index (i.e. cell_id) and add to neighbors
            neighbors[count++] = nz * nSplit * nSplit + ny * nSplit + nx;
          }
        }
      }
    }
    //-----------------------------------------------------

    float coeff = 0.10f;
    float counter = 2.0f; // is used to increase the search radius (i.e. h_new) in case the number of neighbrs is less than 65.
    
    float h_new = 0.0f;

    int k = 0;
    int j = 0;
    int jj = 0;

    float dx, dy, dz;
    float rr;

    float h_i = h[i];

    while (k < 70 || k > 190)
    {

      if (k > 190)
      {
        h_new *= 0.95;
      }
      else
      {
        counter++;
        h_new = (2.0f + (counter + 1.0f) * coeff) * h_i;
      }
      
      k = 0;
      
      for (int q = 0; q < count; q++)
      {
        int starting_nx = offSet[neighbors[q]];
        int ending_nx = offSet[neighbors[q] + 1];
        
        jj = starting_nx;
        
        while (jj < ending_nx && k < MAX_ngb)
        {
          j = groupedIndex[jj];
          
          if (Typ[j] == 0)
          {
            dx = x[j] - x_i;
            dy = y[j] - y_i;
            dz = z[j] - z_i;
            rr = sqrt(dx * dx + dy * dy + dz * dz);

            if (rr <= h_new)
            {
              ngb[i * MAX_ngb + k] = j;
              k++;
            }
          }
          jj++;
        }
      }
    }
  }
}


#endif
