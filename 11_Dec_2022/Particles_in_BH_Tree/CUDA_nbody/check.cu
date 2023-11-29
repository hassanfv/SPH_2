%%writefile test.cu
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;

#define N 1048576
#define THREADS_PER_BLOCK 256
#define EPS2 0.0001


//===== bodyBodyInteraction
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
  float3 r;
  // r_ij  [3 FLOPS]
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
  
  // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
  
  // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f/sqrtf(distSixth);
  
  // s = m_j * invDistCube [1 FLOP]
  float s = bj.w * invDistCube;
  
  // a_i =  a_i + s * r_ij [6 FLOPS]
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
  
  return ai;
}



//===== tile_calculation
__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
  extern __shared__ float4 shPosition[]; // Correct shared memory declaration
  
  for (int i = 0; i < blockDim.x; i++)
  {
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel); 
  }
  
  return accel;
}



//===== calculate_forces
__global__ void calculate_forces(void *devX, void *devA)
{
  extern __shared__ float4 shPosition[];
  
  float4 *globalX = (float4 *)devX;
  float4 *globalA = (float4 *)devA;
  
  float4 myPosition;
  
  int i, tile;
  
  float3 acc = {0.0f, 0.0f, 0.0f};
  
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  
  myPosition = globalX[gtid];
  
  for (i = 0, tile = 0; i < N; i += blockDim.x, tile++)
  {
    int idx = tile * blockDim.x + threadIdx.x;
    
    shPosition[threadIdx.x] = globalX[idx];
    __syncthreads();
    
    acc = tile_calculation(myPosition, acc);
    __syncthreads();
  }
  
  // Save the result in global memory for the integration step.
  float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4;
}





int main()
{
  // Allocate memory on host
  float4 *h_X = (float4*)malloc(N * sizeof(float4));
  float4 *h_A = (float4*)malloc(N * sizeof(float4));
  
  // Initialize random seed
  srand(42);
  // Initialize h_X with random positions
  for (int i = 0; i < N; i++) {
    h_X[i].x = 2.0f * rand() / RAND_MAX - 1.0f;
    h_X[i].y = 2.0f * rand() / RAND_MAX - 1.0f;
    h_X[i].z = 2.0f * rand() / RAND_MAX - 1.0f;
    h_X[i].w = 1.0f; // Assuming w component to be 1.0f, modify as needed
  }
    


  // Allocate memory on device
  float4 *d_X, *d_A;
  cudaMalloc(&d_X, N * sizeof(float4));
  cudaMalloc(&d_A, N * sizeof(float4));

  // Copy data from host to device
  cudaMemcpy(d_X, h_X, N * sizeof(float4), cudaMemcpyHostToDevice);

  auto T_ngb = std::chrono::high_resolution_clock::now();
  int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  calculate_forces<<<numBlocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(float4)>>>(d_X, d_A);
  cudaDeviceSynchronize();
  auto end_ngb = std::chrono::high_resolution_clock::now();
  auto elapsed_ngb = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ngb - T_ngb);
  cout << "T_acc = " << elapsed_ngb.count() * 1e-9 << endl;

  // Copy results back to host
  cudaMemcpy(h_A, d_A, N * sizeof(float4), cudaMemcpyDeviceToHost);

  /*
  for (int i = 0; i < 1000; i++)
  {
    cout << "acc_x = " << h_A[i].x << endl;
    cout << "acc_y = " << h_A[i].y << endl;
  }
  */

  // Clean up
  free(h_X);
  free(h_A);
  cudaFree(d_X);
  cudaFree(d_A);

  return 0;
}


