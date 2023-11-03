#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <ctime>

// Kernel: Compute Pairwise Distances
__global__ void computeDistances(float* d_x, float* d_y, float* d_z, float* d_distances, int numParticles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i >= numParticles || j >= numParticles) return;

    float dx = d_x[i] - d_x[j];
    float dy = d_y[i] - d_y[j];
    float dz = d_z[i] - d_z[j];
    
    d_distances[i * numParticles + j] = sqrtf(dx*dx + dy*dy + dz*dz);
}

// Sort Distances
void sortDistances(float* d_distances, int numParticles)
{
    thrust::device_ptr<float> t_distances = thrust::device_pointer_cast(d_distances);
    for(int i = 0; i < numParticles; i++)
    {
        thrust::sort(t_distances + i*numParticles, t_distances + (i+1)*numParticles);
    }
}

// Kernel: Extract n-th Distance for Smoothing Length
__global__ void extractSmoothingLength(float* d_distances, float* d_smoothingLength, int numParticles, int neighborIndex)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= numParticles) return;

    d_smoothingLength[i] = d_distances[i * numParticles + neighborIndex];
}

int main()
{
    // Initialize random seed
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Number of particles
    int numParticles = 100000; 
    float* h_x = new float[numParticles]; 
    float* h_y = new float[numParticles];
    float* h_z = new float[numParticles];
    
    // Populate arrays with random values between -1 and 1
    for (int i = 0; i < numParticles; i++)
    {
        h_x[i] = 2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f;
        h_y[i] = 2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f;
        h_z[i] = 2.0f * static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 1.0f;
    }

    // Allocate device memory and copy data
    float* d_x, *d_y, *d_z, *d_distances, *d_smoothingLength;
    cudaMalloc(&d_x, numParticles * sizeof(float));
    cudaMalloc(&d_y, numParticles * sizeof(float));
    cudaMalloc(&d_z, numParticles * sizeof(float));
    cudaMalloc(&d_distances, numParticles * numParticles * sizeof(float)); 
    cudaMalloc(&d_smoothingLength, numParticles * sizeof(float));

    cudaMemcpy(d_x, h_x, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, numParticles * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, numParticles * sizeof(float), cudaMemcpyHostToDevice);

    // Compute pairwise distances
    dim3 blockDim(1024, 1024);
    dim3 gridDim((numParticles + blockDim.x - 1) / blockDim.x, (numParticles + blockDim.y - 1) / blockDim.y);
    computeDistances<<<gridDim, blockDim>>>(d_x, d_y, d_z, d_distances, numParticles);

    // Sort distances
    sortDistances(d_distances, numParticles);

    // Extract smoothing length (e.g., 64th nearest neighbor)
    extractSmoothingLength<<<(numParticles + blockDim.x - 1) / blockDim.x, blockDim.x>>>(d_distances, d_smoothingLength, numParticles, 64);

    // You might want to copy the results back and do further processing...

    // Cleanup
    delete[] h_x;
    delete[] h_y;
    delete[] h_z;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_distances);
    cudaFree(d_smoothingLength);

    return 0;
}

