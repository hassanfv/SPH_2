



#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <float.h> // For FLT_MAX

#define BLOCK_SIZE 256 // Adjust this according to your GPU's capability
#define N 1000

// CUDA error checking
static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

// Kernel to compute the bounding box
__global__ void computeBoundingBox(float3 *positions, float3 *minPos, float3 *maxPos, int numBodies) {
    // Shared memory to store intermediate results of reduction within a block
    __shared__ float3 s_min[BLOCK_SIZE];
    __shared__ float3 s_max[BLOCK_SIZE];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    
    // Initialize local min and max with first element or max float values
    float3 local_min = i < numBodies ? positions[i] : make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 local_max = i < numBodies ? positions[i] : make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    // Stride over all elements and find min and max
    for (int stride = i + gridDim.x * blockDim.x; stride < numBodies; stride += gridDim.x * blockDim.x) {
        float3 pos = positions[stride];
        local_min = make_float3(fminf(local_min.x, pos.x), fminf(local_min.y, pos.y), fminf(local_min.z, pos.z));
        local_max = make_float3(fmaxf(local_max.x, pos.x), fmaxf(local_max.y, pos.y), fmaxf(local_max.z, pos.z));
    }
    
    // Each thread puts its local min and max into shared memory
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = make_float3(fminf(s_min[tid].x, s_min[tid + s].x), fminf(s_min[tid].y, s_min[tid + s].y), fminf(s_min[tid].z, s_min[tid + s].z));
            s_max[tid] = make_float3(fmaxf(s_max[tid].x, s_max[tid + s].x), fmaxf(s_max[tid].y, s_max[tid + s].y), fmaxf(s_max[tid].z, s_max[tid + s].z));
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        minPos[blockIdx.x] = s_min[0];
        maxPos[blockIdx.x] = s_max[0];
    }
}

int main() {
    // Allocate memory for N bodies
    float3 *positions;
    HANDLE_ERROR(cudaMallocManaged(&positions, N * sizeof(float3)));

    // Initialize positions with some values
    for (int i = 0; i < N; ++i) {
        positions[i] = make_float3(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
    }

    // Allocate memory for the result
    float3 *minPos, *maxPos;
    HANDLE_ERROR(cudaMallocManaged(&minPos, sizeof(float3)));
    HANDLE_ERROR(cudaMallocManaged(&maxPos, sizeof(float3)));

    // Run the kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeBoundingBox<<<numBlocks, BLOCK_SIZE>>>(positions, minPos, maxPos, N);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Now reduce the results from each block to find the global min and max
    float3 global_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 global_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < numBlocks; ++i) {
        global_min = make_float3(fminf(global_min.x, minPos[i].x), fminf(global_min.y, minPos[i].y), fminf(global_min.z, minPos[i].z));
        global_max = make_float3(fmaxf(global_max.x, maxPos[i].x), fmaxf(global_max.y, maxPos[i].y), fmaxf(global_max.z, maxPos[i].z));
    }

    // Print the global bounding box coordinates
    printf("Global Min: (%f, %f, %f)\n", global_min.x, global_min.y, global_min.z);
    printf("Global Max: (%f, %f, %f)\n", global_max.x, global_max.y, global_max.z);

    // Clean up
    HANDLE_ERROR(cudaFree(positions));
    HANDLE_ERROR(cudaFree(minPos));
    HANDLE_ERROR(cudaFree(maxPos));

    return 0;
}

