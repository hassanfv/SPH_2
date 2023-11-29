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




__global__ void ngbTest(float *d_x, float *d_y, float x_min, float y_min, float W_cell, int nSplit, int Ncell, int N)
{
    extern __shared__ int count[];  // Use extern to define a dynamic shared memory array

    CountBodies(d_x, d_y, x_min, y_min, W_cell, nSplit, count, Ncell, N);
    ComputeOffset(count, Ncell);


    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0)
    {
        for (int j = 0; j < 2 * Ncell; j++)
        {
            printf("i, count = %d, %d\n", j, count[j]);
        } 
    }
}





int main()
{
  float x_min = -1.0;
  float y_min = -1.0;
  
  float maxCoord = 1.0;

  int nSplit = 4;
  int Ncell = nSplit * nSplit;
  
  float W_cell = ceil(2.0 * maxCoord) / nSplit;
  
  int N = 200;
  
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  float *x, *y, *d_x, *d_y;
  
  x = new float[N];
  y = new float[N];
  
  cudaMalloc((void **)&d_x, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++)
  {
    x[i] = dis(gen);
    y[i] = dis(gen);
  }

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  
  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  ngbTest<<<gridSize, blockSize, 2 * Ncell * sizeof(int)>>>(d_x, d_y, x_min, y_min, W_cell, nSplit, Ncell, N);
  cudaDeviceSynchronize();
  
  
  
  // Writing to a binary file
  std::ofstream outFile("data.bin", std::ios::out | std::ios::binary);
  if (!outFile) {
    cerr << "Error opening file for writing." << endl;
    return 1;
  }

  // Write the size (N) first
  outFile.write(reinterpret_cast<char*>(&N), sizeof(N));

  // Write x and y arrays
  outFile.write(reinterpret_cast<char*>(x), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(y), N * sizeof(float));

  outFile.close();
  
  

  // Don't forget to free allocated memory
  delete[] x;
  delete[] y;
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}

