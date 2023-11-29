%%writefile test.cu

#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include "hfv_BHLibs_v1.h"

using namespace std;


int main()
{

  int n = 4000000; // number of particles.
  int nBodies = n;

  int MAX_NODES = 1000000;
  int N_LEAF = 500000;

  int nNodes = MAX_NODES;
  int leafLimit = MAX_NODES - N_LEAF;
  
  Body *h_b, *d_b, *d_b_buffer;
  Node *h_node, *d_node;
  
  int *d_mutex;
  
  h_b = new Body[n];
  h_node = new Node[nNodes];

  cudaMalloc((void **)&d_b, sizeof(Body) * n);
  cudaMalloc((void **)&d_node, sizeof(Node) * nNodes);
  cudaMalloc((void **)&d_mutex, sizeof(int) * nNodes);
  cudaMalloc((void **)&d_b_buffer, sizeof(Body) * n); // n = nBodies


  //--- preparing bodies (bodies are actually particles!) ---
  //random_device rd;  // Obtain a random number from hardware
  mt19937 eng(42); // Seed the generator
  uniform_real_distribution<> distr(-1.0, 1.0); // Define the range
  
  for (int i = 0; i < n; i++)
  {
    h_b[i].position.x = distr(eng);
    h_b[i].position.y = distr(eng);
    h_b[i].position.z = distr(eng);
    
    h_b[i].mass = 1.0;
  }
  
  //--- copying particles from host to device
  cudaMemcpy(d_b, h_b, nBodies * sizeof(Body), cudaMemcpyHostToDevice);
  
  cout << "h_node.minCorner.x = " << h_node[0].minCorner.x << endl;
  cout << "h_node.minCorner.y = " << h_node[0].minCorner.y << endl;
  cout << "h_node.minCorner.z = " << h_node[0].minCorner.z << endl;
  cout << "h_node.maxCorner.x = " << h_node[0].maxCorner.x << endl;
  cout << "h_node.maxCorner.y = " << h_node[0].maxCorner.y << endl;
  cout << "h_node.maxCorner.z = " << h_node[0].maxCorner.z << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;
  
  int blockSize = BLOCK_SIZE;
  dim3 gridSize = ceil((float)nNodes / blockSize);
  ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);
  cudaDeviceSynchronize();

  cudaMemcpy(h_node, d_node, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
  cout << "h_node.minCorner.x = " << h_node[0].minCorner.x << endl;
  cout << "h_node.minCorner.y = " << h_node[0].minCorner.y << endl;
  cout << "h_node.minCorner.z = " << h_node[0].minCorner.z << endl;
  cout << "h_node.maxCorner.x = " << h_node[0].maxCorner.x << endl;
  cout << "h_node.maxCorner.y = " << h_node[0].maxCorner.y << endl;
  cout << "h_node.maxCorner.z = " << h_node[0].maxCorner.z << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;


  blockSize = BLOCK_SIZE;
  gridSize = ceil((float)nBodies / blockSize);
  ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);
  cudaDeviceSynchronize();

  cudaMemcpy(h_node, d_node, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
  cout << "h_node.minCorner.x = " << h_node[0].minCorner.x << endl;
  cout << "h_node.minCorner.y = " << h_node[0].minCorner.y << endl;
  cout << "h_node.minCorner.z = " << h_node[0].minCorner.z << endl;
  cout << "h_node.maxCorner.x = " << h_node[0].maxCorner.x << endl;
  cout << "h_node.maxCorner.y = " << h_node[0].maxCorner.y << endl;
  cout << "h_node.maxCorner.z = " << h_node[0].maxCorner.z << endl;
  
  cout << "h_node.start = " << h_node[0].start << endl;
  cout << "h_node.end = " << h_node[0].end << endl;
  cout << endl;
  
  blockSize = BLOCK_SIZE;
  gridSize = ceil((float)nBodies / blockSize);
  ConstructQuadTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
  cudaDeviceSynchronize();

  //----- example prints --
  cudaMemcpy(h_node, d_node, sizeof(Node) * nNodes, cudaMemcpyDeviceToHost);
  Node node_1 = h_node[100];
  cout << "start = " << node_1.start << endl;
  cout << "end = " << node_1.end << endl;


  auto T_hh = std::chrono::high_resolution_clock::now();

  blockSize = 32; // perhaps because of the warp use in ComputeForce function!
  gridSize = ceil((float)nBodies / blockSize);
  ComputeForceKernel<<<gridSize, blockSize>>>(d_node, d_b, nNodes, nBodies, leafLimit);
  cudaDeviceSynchronize();
  
  auto end_hh = std::chrono::high_resolution_clock::now();
  auto elapsed_hh = std::chrono::duration_cast<std::chrono::nanoseconds>(end_hh - T_hh);
  cout << "T_h = " << elapsed_hh.count() * 1e-9 << endl;
  
  cudaMemcpy(h_b, d_b, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost);
  
  Body b1 = h_b[100];
  
  float accx = b1.acceleration.x;
  float accy = b1.acceleration.y;
  float accz = b1.acceleration.z;
  
  printf("(accx, accy, accz) = %f, %f, %f\n", accx, accy, accz);
  cout << "b1.position.x = " << b1.position.x << endl;
  cout << "b1.position.y = " << b1.position.y << endl;
  cout << "b1.position.z = " << b1.position.z << endl;


  /*
  // Save h_b to a binary file
  for (int i = 0; i < n; i++)
  {
    h_b[i].acceleration.x = 0.0;
    h_b[i].acceleration.y = 0.0;
    h_b[i].acceleration.z = 0.0;
  }

  saveBodyToFile("h_b.bin", h_b, nBodies);


  //----- Output to a file -----
  saveToFile("BH.bin", h_b, h_node, nBodies, nNodes);
  
  */


}



