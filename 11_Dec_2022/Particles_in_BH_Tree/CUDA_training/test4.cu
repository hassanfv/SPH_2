%%writefile test.cu
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <random>
#include <cmath>

using namespace std;

#define tileSize 3


__global__ void matMultX(float *A, float *B, float *C, int N)
{

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  
  __shared__ float shareA[tileSize * tileSize];
  __shared__ float shareB[tileSize * tileSize];
  
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  float temp = 0.0;
  
  for (int i = 0; i < N/tileSize; i++)
  {
    shareA[ty * tileSize + tx] = A[row * N + (i * tileSize + threadIdx.x)];
    shareB[ty * tileSize + tx] = B[(i * tileSize + threadIdx.y) * N + col];
    __syncthreads();
    
    for (int k = 0; k < tileSize; k++)
    {
      temp += shareA[ty * tileSize + k] * shareB[k * tileSize + tx];
    }
    __syncthreads();
  }
  
  C[row * N + col] = temp;  
}



__global__ void matMult(float *A, float *B, float *C, int N)
{

  int row = blockIdx.x;
  int col = blockIdx.y;
  
  int k = threadIdx.x;
  
  extern __shared__ float smem[];
  
  float temp = A[row * N + k] * B[k * N + col];
  
  smem[k] = temp;
  __syncthreads();
  
  float s = 0.0f;
  
  if (threadIdx.x == 0)
  {
    for (int i = 0; i < N; i++)
    {
      s += smem[i];
    }
    __syncthreads();
    
    C[row * N + col] = s;
  }
  
}


mt19937 gen(42);
uniform_int_distribution<> dist(1, 9);

float generateRandomFloat()
{
  return static_cast<float>(dist(gen));
}



void create_matrix(float *A, int N)
{
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      A[i * N + j] = generateRandomFloat();
    }
  }
}



void print_matrix(float *D, int N)
{
  for (int i = 1; i <= N * N; i++)
  {
    cout << D[i-1] << "   ";
    
    if ((i % N) == 0)
      {
        cout << endl;
      }
  }
}



int main()
{

  int N = 12;

  float *A, *B;
  
  A = new float[N * N];
  B = new float[N * N];
  
  float *d_A, *d_B, *d_C;
  
  cudaMalloc((void**)&d_A, N * N * sizeof(float));
  cudaMalloc((void**)&d_B, N * N * sizeof(float));
  cudaMalloc((void**)&d_C, N * N * sizeof(float));
  
  create_matrix(A, N);
  create_matrix(B, N);
  
  cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 blockSize(tileSize, tileSize, 1);
  dim3 gridSize(N / tileSize, N / tileSize, 1);
  
  //matMult<<<gridSize, blockSize, N * sizeof(float)>>>(d_A, d_B, d_C, N);
  matMultX<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();
  
  float *C;
  C = new float[N * N];
  
  cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
  
  /*
  print_matrix(A, N);
  cout << endl << endl;
  
  print_matrix(B, N);
  cout << endl << endl;
  */
  
  print_matrix(C, N);
  cout << endl << endl;
  
  /*
  for (int i = 0; i < N*N; i++)
  {
    cout << C[i] << endl;
  }
  */
  



}




