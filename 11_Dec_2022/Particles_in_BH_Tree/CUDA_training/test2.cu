%%writefile test.cu
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;


__global__ void helloWorld(void)
{

  printf("Hello World! (from Block %d, Thread %d)\n", blockIdx.x, threadIdx.x);

}


int main()
{

  helloWorld<<<3, 4>>>();
  cudaDeviceSynchronize();


}
