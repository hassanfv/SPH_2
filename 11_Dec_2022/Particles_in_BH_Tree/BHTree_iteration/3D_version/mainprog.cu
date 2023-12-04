%%writefile test.cu

#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include "bh_tree_iteration_v0.h"

using namespace std;





int main()
{

  int numParticles;
  int numNodes;

  float *h_left;
  float *h_right;
  float *h_bottom;
  float *h_top;
  float *h_front;
  float *h_back;

  float *h_mass;
  float *h_x;
  float *h_y;
  float *h_z;
  float *h_ax;
  float *h_ay;
  float *h_az;

  int *h_child;
  int *h_start;
  int *h_sorted;
  int *h_count;

  float *d_left;
  float *d_right;
  float *d_bottom;
  float *d_top;
  float *d_front;
  float *d_back;

  float *d_mass;
  float *d_x;
  float *d_y;
  float *d_z;
  float *d_ax;
  float *d_ay;
  float *d_az;

  int *d_index;
  int *d_child;
  int *d_start;
  int *d_sorted;
  int *d_count;

  int *d_mutex;  //used for locking 

  cudaEvent_t start, stop; // used for timing

  //parameters = p;
  //step = 0;

  int N = pow(2, 10);

  int blockSize_bh = blockSize;
  int gridSize_bh = (N + blockSize_bh - 1) / blockSize_bh;

  numParticles = N;
  numNodes = 8 * N + 15000;

  //int m = numNodes;

  // allocate host data
  h_left = new float;
  h_right = new float;
  h_bottom = new float;
  h_top = new float;
  h_front = new float;
  h_back = new float;
  h_mass = new float[numNodes];
  h_x = new float[numNodes];
  h_y = new float[numNodes];
  h_z = new float[numNodes];
  h_ax = new float[numNodes];
  h_ay = new float[numNodes];
  h_az = new float[numNodes];
  h_child = new int[8*numNodes];
  h_start = new int[numNodes];
  h_sorted = new int[numNodes];
  h_count = new int[numNodes];

  // allocate device data
  gpuErrchk(cudaMalloc((void**)&d_left, sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_right, sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_bottom, sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_top, sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_front, sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_back, sizeof(float)));

  gpuErrchk(cudaMemset(d_left, 0, sizeof(float)));
  gpuErrchk(cudaMemset(d_right, 0, sizeof(float)));
  gpuErrchk(cudaMemset(d_bottom, 0, sizeof(float)));
  gpuErrchk(cudaMemset(d_top, 0, sizeof(float)));
  gpuErrchk(cudaMemset(d_front, 0, sizeof(float)));
  gpuErrchk(cudaMemset(d_back, 0, sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&d_mass, numNodes*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&d_x, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_y, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_z, numNodes*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&d_ax, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_ay, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&d_az, numNodes*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&d_index, sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_child, 8*numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_mutex, sizeof(int))); 

  gpuErrchk(cudaMemset(d_start, -1, numNodes*sizeof(int)));
  gpuErrchk(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

  //int memSize = sizeof(float) * 2 * numParticles;
    
  reset_arrays_kernel<<< gridSize_bh, blockSize_bh >>>(d_mutex, d_x, d_y, d_z, d_mass, d_count, d_start, d_sorted, d_child, d_index,
                                                 d_left, d_right, d_bottom, d_top, d_front, d_back, N, numNodes);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
  }


  // initializing x, y, z, mass -----
  mt19937 engine(42);
  uniform_real_distribution<float> distribution(0.0, 1.0);
  
  for (int i = 0; i < numParticles; i++)
  {
    h_x[i] = distribution(engine);
    h_y[i] = distribution(engine);
    h_z[i] = distribution(engine);
    
    h_mass[i] = 0.5f;
    
    //cout << h_x[i] << "," << h_y[i] << "," << h_z[i] << "," << h_mass[i] << endl;
  }
  
  
  cudaMemcpy(d_x, h_x, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, h_z, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass, h_mass, numNodes * sizeof(float), cudaMemcpyHostToDevice);

  


  // After the kernel call and cudaDeviceSynchronize()
  compute_bounding_box_kernel<<< gridSize_bh, blockSize_bh >>>(d_mutex, d_x, d_y, d_z, d_left, d_right, d_bottom, d_top, d_front, d_back, N);
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
  cudaMemcpy(h_front, d_front, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_back, d_back, sizeof(float), cudaMemcpyDeviceToHost);
  
  int *h_index = new int;
  cudaMemcpy(h_index, d_index, sizeof(int), cudaMemcpyDeviceToHost);
  printf("\n");
  printf("h_left, h_right, h_bottom, h_top, h_front, h_back = %f, %f, %f, %f, %f, %f\n", h_left[0], h_right[0], h_bottom[0], h_top[0], h_front[0], h_back[0]);
  printf("\n");
  printf("initial index = %d\n", h_index[0]);
  printf("\n");

  auto T_build_tree_kernel = std::chrono::high_resolution_clock::now();
  build_tree_kernel<<< 1, 256 >>>(d_x, d_y, d_z, d_mass, d_count, d_start, d_child, d_index, d_left, d_right, d_bottom, d_top, d_front, d_back, N, numNodes);
  cudaDeviceSynchronize();  
  auto end_build_tree_kernel = std::chrono::high_resolution_clock::now();
  auto elapsed_build_tree_kernel = std::chrono::duration_cast<std::chrono::nanoseconds>(end_build_tree_kernel - T_build_tree_kernel);
  cout << "Elapsed time = " << elapsed_build_tree_kernel.count() * 1e-9 << endl;
  
  
  centre_of_mass_kernel<<<gridSize_bh, blockSize_bh>>>(d_x, d_y, d_z, d_mass, d_index, N);
  cudaDeviceSynchronize();  
  
  
  sort_kernel<<< 1, 256 >>>(d_count, d_start, d_sorted, d_child, d_index, N);
  cudaDeviceSynchronize();  
  
  
  auto T_Force = std::chrono::high_resolution_clock::now();
  compute_forces_kernel<<< gridSize_bh, blockSize_bh >>>(d_x, d_y, d_z, d_ax, d_ay, d_az, d_mass, d_sorted, d_child,
                                                   d_left, d_right, d_bottom, d_top, d_front, d_back, N);
  cudaDeviceSynchronize();
  auto end_Force = std::chrono::high_resolution_clock::now();
  auto elapsed_Force = std::chrono::duration_cast<std::chrono::nanoseconds>(end_Force - T_Force);
  cout << "T_Force = " << elapsed_Force.count() * 1e-9 << endl;
  
  
  cudaMemcpy(h_ax, d_ax, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ay, d_ay, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_az, d_az, numNodes * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < numParticles; i++)
  {
    //cout << "ax[" << i << "] = " << h_ax[i] << endl;
    cout << h_ay[i] << endl;
  }


}
