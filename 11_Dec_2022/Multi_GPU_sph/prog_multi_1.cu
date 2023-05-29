#include <cuda_runtime.h>
#include <iostream>

using namespace std;
__global__ void multiply_by_0_1(float *x, int start, int end)
{
    int index = start + threadIdx.x + blockIdx.x * blockDim.x;
    if (index < end)
    {
        x[index] *= 0.1f;
    }
}

int main()
{
    const int nGPUs = 4;
    const int N = 1e9;
    const int N_per_GPU = N / nGPUs;
    const int remainder = N % nGPUs;
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N_per_GPU + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate unified memory - Shared between All GPUs and also CPU!
    float *d_x;
    cudaMallocManaged(&d_x, N * sizeof(float));

    // Initialize input vector
    for (int i = 0; i < N; ++i)
    {
        d_x[i] = static_cast<float>(i);
    }

    // Determining the beg and end index of the array for each GPU.

    int *beg, *end;

    beg = new int[nGPUs];
    end = new int[nGPUs];

    for (int rank = 0; rank < nGPUs; rank++)
    {
        if (rank < remainder)
        {
            beg[rank] = rank * (N_per_GPU + 1);
            end[rank] = beg[rank] + N_per_GPU + 1;
        }
        else
        {
            beg[rank] = rank * N_per_GPU + remainder;
            end[rank] = beg[rank] + N_per_GPU;
        }
    }

    // Run kernel on each GPU
    for (int i = 0; i < nGPUs; i++)
    {

        cudaSetDevice(i);
        multiply_by_0_1<<<blocksPerGrid, threadsPerBlock>>>(d_x, beg[i], end[i]);
    }

    // Ensure all GPUs have finished
    for (int i = 0; i < nGPUs; i++)
    {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    // Verify result
    for (int i = 0; i < 10; ++i)
    {
        // Check if the result is correct
        cout << "i, d_x = " << i << ", " << d_x[i] << endl;
    }

    for (int i = 999990; i < N; ++i)
    {
        // Check if the result is correct
        cout << "i, d_x = " << i << ", " << d_x[i] << endl;
    }

    // Free unified memory
    cudaFree(d_x);

    return 0;
}
