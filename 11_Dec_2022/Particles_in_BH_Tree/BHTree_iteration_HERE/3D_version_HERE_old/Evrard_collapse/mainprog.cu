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

  int numParticles, numNodes;

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

  float *dev_mass;
  float *dev_x;
  float *dev_y;
  float *dev_z;
  float *dev_ax;
  float *dev_ay;
  float *dev_az;

  int *d_index;
  int *d_child;
  int *d_start;
  int *d_sorted;
  int *d_count;

  int *d_mutex;  //used for locking 

  //int N = pow(2, 10);

  int blockSize_bh = blockSize;
  int gridSize_bh = (N + blockSize_bh - 1) / blockSize_bh;

  numParticles = nBodies; // nBodies is the number of patticles with Typ != -1.
  numNodes = 8 * numParticles + 15000;

  //int m = numNodes;

  // allocate host data
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

  gpuErrchk(cudaMalloc((void**)&dev_mass, numNodes*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&dev_x, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dev_y, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dev_z, numNodes*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&dev_ax, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dev_ay, numNodes*sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&dev_az, numNodes*sizeof(float)));

  gpuErrchk(cudaMalloc((void**)&d_index, sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_child, 8*numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_start, numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_sorted, numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_count, numNodes*sizeof(int)));
  gpuErrchk(cudaMalloc((void**)&d_mutex, sizeof(int))); 

  gpuErrchk(cudaMemset(d_start, -1, numNodes*sizeof(int)));
  gpuErrchk(cudaMemset(d_sorted, 0, numNodes*sizeof(int)));

  //int memSize = sizeof(float) * 2 * numParticles;
    
  reset_arrays_kernel<<< gridSize_bh, blockSize_bh >>>(d_mutex, dev_x, dev_y, dev_z, dev_mass, d_count, d_start, d_sorted, d_child, d_index,
                                                       d_left, d_right, d_bottom, d_top, d_front, d_back, numParticles, numNodes);
  cudaDeviceSynchronize();

  // initializing x, y, z, mass -----
  cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);
    
  for (int i = 0; i < numParticles; i++)
  {
    h_x[i] = x[i];
    h_y[i] = y[i];
    h_z[i] = z[i];
    
    h_mass[i] = mass[i];
  }

  cudaMemcpy(dev_x, h_x, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, h_y, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_z, h_z, numNodes * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mass, h_mass, numNodes * sizeof(float), cudaMemcpyHostToDevice);

  compute_bounding_box_kernel<<< gridSize_bh, blockSize_bh >>>(d_mutex, dev_x, dev_y, dev_z, d_left, d_right, d_bottom, d_top, d_front, d_back, numParticles);
  cudaDeviceSynchronize();
  
  auto T_build_tree_kernel = std::chrono::high_resolution_clock::now();
  build_tree_kernel<<< 1, 256 >>>(dev_x, dev_y, dev_z, dev_mass, d_count, d_start, d_child, d_index, d_left, d_right, d_bottom, d_top, d_front, d_back,
                                  numParticles, numNodes);
  cudaDeviceSynchronize();  
  auto end_build_tree_kernel = std::chrono::high_resolution_clock::now();
  auto elapsed_build_tree_kernel = std::chrono::duration_cast<std::chrono::nanoseconds>(end_build_tree_kernel - T_build_tree_kernel);
  cout << "Elapsed time = " << elapsed_build_tree_kernel.count() * 1e-9 << endl;
  
  
  centre_of_mass_kernel<<<gridSize_bh, blockSize_bh>>>(dev_x, dev_y, dev_z, dev_mass, d_index, numParticles);
  cudaDeviceSynchronize();  
  
  
  sort_kernel<<< 1, 256 >>>(d_count, d_start, d_sorted, d_child, d_index, numParticles);
  cudaDeviceSynchronize();  
  
  
  auto T_Force = std::chrono::high_resolution_clock::now();
  compute_forces_kernel<<< gridSize_bh, blockSize_bh >>>(dev_x, dev_y, dev_z, dev_ax, dev_ay, dev_az, dev_mass, d_sorted, d_child,
                                                         d_left, d_right, d_bottom, d_top, d_front, d_back, numParticles);
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
