%%writefile test.cu
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;



int main()
{

  int nDevice;
  
  cudaGetDeviceCount(&nDevice);
  
  cudaDeviceProp prop;
  
  cudaGetDeviceProperties(&prop, 0);
  
  cout << "nDevice = " << nDevice << endl;
  cout << "Device name: " << prop.name << endl;
  cout << "Total global memory [GB]: " << prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f) << endl;
  cout << "Shared memory per block [KB]: " << prop.sharedMemPerBlock / 1024.0f << endl;
  cout << "Registors per block [KB]: " << prop.regsPerBlock / 1024.0f << endl;
  cout << "Warp size: " << prop.warpSize << endl;
  cout << "Max Threads per block: " << prop.maxThreadsPerBlock << endl;

}
