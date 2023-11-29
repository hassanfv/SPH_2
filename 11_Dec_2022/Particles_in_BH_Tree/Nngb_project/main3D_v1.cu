%%writefile test.cu
#include <iostream>
#include <cmath>
#include <random>  // Include this for random number generation
#include <fstream>
#include "ngb_v1.h"

using namespace std;



int main()
{

  int N = 1000000;

  int nSplit = 10; // for 3D never go above 10 as 10^3 = 1000 and the maximum blockSize in GPUs is 1024!! IMPORTANT !!
  int Ncell = nSplit * nSplit * nSplit;

  
  float *x, *y, *z, *d_x, *d_y, *d_z;
  int *groupedIndex, *d_groupedIndex, *countx, *d_countx;
  
  x = new float[N];
  y = new float[N];
  z = new float[N];
  
  groupedIndex = new int[N];
  countx = new int[Ncell];
  
  cudaMalloc((void **)&d_x, N * sizeof(float));
  cudaMalloc((void **)&d_y, N * sizeof(float));
  cudaMalloc((void **)&d_z, N * sizeof(float));
  
  cudaMalloc((void **)&d_groupedIndex, N * sizeof(int));
  cudaMalloc((void **)&d_countx, (Ncell) * sizeof(int));

  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  for (int i = 0; i < N; i++)
  {
    x[i] = dis(gen);
    y[i] = dis(gen);
    z[i] = dis(gen);
    
    groupedIndex[i] = i; // just a place holder. Its initial values don't matter as it will be replaced anyway!
  }
  
  for (int i = 0; i < (Ncell); i++)
  {
    countx[i] = 0;
  }

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_groupedIndex, groupedIndex, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_countx, countx, (Ncell) * sizeof(int), cudaMemcpyHostToDevice);
  
  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid
  
  float *maxArr, *d_maxArr;
  maxArr = new float[gridSize];
  cudaMalloc((void **)&d_maxArr, N * sizeof(float));

  maxCoord<<<gridSize, blockSize, gridSize * sizeof(int)>>>(d_x, d_y, d_z, d_maxArr, N);
  cudaDeviceSynchronize();
  cudaMemcpy(maxArr, d_maxArr, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
  
  
  cout << "maxArr[0] = " << maxArr[0] << endl;
  
  //exit(0);
  
  float maxRange = maxArr[0];
  float W_cell = ceil(2.0 * maxRange) / nSplit;
  
  float x_min = -1.0 * maxRange;
  float y_min = -1.0 * maxRange;
  float z_min = -1.0 * maxRange;
  

  ngbFinder<<<1, 1024, 2 * Ncell * sizeof(int)>>>(d_x, d_y, d_z, d_groupedIndex, d_countx, x_min, y_min, z_min, W_cell, nSplit, Ncell, N);
  cudaDeviceSynchronize();
  
  cudaMemcpy(countx, d_countx, (Ncell) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(groupedIndex, d_groupedIndex, N * sizeof(int), cudaMemcpyDeviceToHost);
  
  int *offSet = new int[Ncell+1];
  
  offSet[0] = 0;
  for (int i = 0; i < Ncell; i++)
  {
    offSet[i+1] = countx[i];
  }
  
  
  
  int i = 100;
  
  int cell_id = getCelliD(x[i], y[i], z[i], x_min, y_min, z_min, W_cell, nSplit);
  
  cout << "particle i is at cell = " << cell_id << endl;
  cout << "(x[i], y[i], z[i]) = " << x[i] << ", " << y[i] << ", " << z[i] << endl;
  
  
  
  
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
  outFile.write(reinterpret_cast<char*>(z), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(groupedIndex), N * sizeof(int));
  outFile.write(reinterpret_cast<char*>(offSet), (Ncell+1) * sizeof(int));

  outFile.close();

  // Don't forget to free allocated memory
  delete[] x;
  delete[] y;
  delete[] z;
  delete[] groupedIndex;
  delete[] countx;
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_groupedIndex);
  cudaFree(d_countx);

  return 0;
}

