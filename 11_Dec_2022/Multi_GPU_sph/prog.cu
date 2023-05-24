#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void multiply_func(float *x, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < N)
    {
        x[i] *= 0.1f;
    }
}

int main()
{

    int N = 1e6;

    float *x = new float[N];

    for (int i = 0; i < N; i++)
    {
        x[i] = static_cast<float>(i);
    }

    // Allocate array in device memory.
    float *d_x;
    cudaMalloc(&d_x, N * sizeof(float));

    // Copy array from host to device memory.
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel.
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    multiply_func<<<blocksPerGrid, threadsPerBlock>>>(d_x, N);

    // Copy results from device memory to host memory
    cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++)
    {
        cout << "i, x = " << i << ", " << x[i] << endl;
    }

    return 0;
}