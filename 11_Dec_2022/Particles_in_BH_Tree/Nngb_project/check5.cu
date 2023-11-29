%%writefile test.cu
#include <iostream>
#include <cmath>
#include <random>  // Include this for random number generation
#include <fstream>

using namespace std;

struct Cell {
  int row;
  int col;
  float xcen;
  float ycen;
  int start = -1;
  int end = -1;
};




__device__ int getCelliD(float x, float y, float x_min, float y_min, float Wcell, int nSplit)
{
  int col = static_cast<int>((x - x_min) / Wcell);
  int row = static_cast<int>((y - y_min) / Wcell);
  
  return row * nSplit + col;
}



__device__ void CountBodies(float *d_x, float *d_y, float x_min, float y_min, float Wcell, int nSplit, int *count, int Ncell, int N)
{
  int tx = threadIdx.x;
  if (tx < Ncell)
    count[tx] = 0;
  __syncthreads();

  for (int i = tx; i < N; i += blockDim.x)
  {
    int cell_iD = getCelliD(d_x[i], d_y[i], x_min, y_min, Wcell, nSplit);
    atomicAdd(&count[cell_iD], 1);
  }
  __syncthreads();
}



__device__ void ComputeOffset(int *count, int Ncell)
{
    int tx = threadIdx.x;
    if (tx < Ncell)
    {
        int offset = 0;
        for (int i = 0; i < tx; ++i)
        {
            offset += count[i];
        }
        count[tx + Ncell] = offset;
    }
    __syncthreads();
}



__device__ void GroupBodies(float *d_x, float *d_y, int *d_groupedIndex, float x_min, float y_min, float Wcell, int nSplit, int *count, int Ncell, int N)
{
  int *offsets = &count[Ncell];
  for (int i = threadIdx.x; i < N; i += blockDim.x)
  {
    int cell_iD = getCelliD(d_x[i], d_y[i], x_min, y_min, Wcell, nSplit);
    int dest = atomicAdd(&offsets[cell_iD], 1);
    d_groupedIndex[dest] = i;
  }
    __syncthreads();
}


__global__ void ngbTest(float *d_x, float *d_y, int *d_groupedIndex, int *d_count, float x_min, float y_min, float W_cell, int nSplit, int Ncell, int N)
{
  extern __shared__ int count[];  // Use extern to define a dynamic shared memory array

  CountBodies(d_x, d_y, x_min, y_min, W_cell, nSplit, count, Ncell, N);
  ComputeOffset(count, Ncell);
  GroupBodies(d_x, d_y, d_groupedIndex, x_min, y_min, W_cell, nSplit, count, Ncell, N);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0)
  {
    
    
    for (int j = 0; j < 2 * Ncell; j++)
    {
      printf("j, count = %d, %d\n", j, count[j]);
    }
    
    
    
    for (int j = Ncell; j < 2 * Ncell; j++)
    {
      d_count[j-Ncell] = count[j];
    }
  }
}








int main()
{
  float x_min = -1.0;
  float y_min = -1.0;
  
  float maxCoord = 1.0;

  int nSplit = 30;
  int Ncell = nSplit * nSplit;
  
  float W_cell = ceil(2.0 * maxCoord) / nSplit;
  
  int N = 1000000;

  float *x, *y, *d_x, *d_y;
  int *groupedIndex, *d_groupedIndex, *countx, *d_countx;
  
  x = new float[N];
  y = new float[N];
  
  groupedIndex = new int[N];
  countx = new int[Ncell];
  
  cudaMalloc((void **)&d_x, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));
  
  cudaMalloc((void **)&d_groupedIndex, N * sizeof(int));
  cudaMalloc((void **)&d_countx, (Ncell) * sizeof(int));

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < N; i++)
  {
    x[i] = dis(gen);
    y[i] = dis(gen);
    
    groupedIndex[i] = i; // just a place holder. Its initial values don't matter as it will be replaced anyway!
  }
  
  for (int i = 0; i < (Ncell); i++)
  {
    countx[i] = 0;
  }

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_groupedIndex, groupedIndex, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_countx, countx, (Ncell) * sizeof(int), cudaMemcpyHostToDevice);

  ngbTest<<<1, 1024, 2 * Ncell * sizeof(int)>>>(d_x, d_y, d_groupedIndex, d_countx, x_min, y_min, W_cell, nSplit, Ncell, N);
  cudaDeviceSynchronize();
  
  cudaMemcpy(countx, d_countx, (Ncell) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(groupedIndex, d_groupedIndex, N * sizeof(int), cudaMemcpyDeviceToHost);
  
  int *scanOffset = new int[Ncell+1];
  
  scanOffset[0] = 0;
  for (int i = 0; i < Ncell; i++)
  {
    scanOffset[i+1] = countx[i];
  }
  
  // Writing to a binary file
  std::ofstream outFile("dataX.bin", std::ios::binary);
  if (!outFile) {
    cerr << "Error opening file for writing." << endl;
    return 1;
  }

  // Write the size (N, Ncell, nSplit) and W_cell first
  outFile.write(reinterpret_cast<char*>(&N), sizeof(N));
  outFile.write(reinterpret_cast<char*>(&Ncell), sizeof(Ncell));
  outFile.write(reinterpret_cast<char*>(&nSplit), sizeof(nSplit));
  outFile.write(reinterpret_cast<char*>(&W_cell), sizeof(W_cell));

  // Write x, y, groupedIndex, and count arrays
  outFile.write(reinterpret_cast<char*>(x), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(y), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(groupedIndex), N * sizeof(int));
  outFile.write(reinterpret_cast<char*>(scanOffset), (Ncell+1) * sizeof(int));

  outFile.close();

  // Don't forget to free allocated memory
  delete[] x;
  delete[] y;
  delete[] groupedIndex;
  delete[] countx;
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_groupedIndex);
  cudaFree(d_countx);

  return 0;
}

