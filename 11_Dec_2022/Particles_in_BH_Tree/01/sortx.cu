%%writefile test.cu

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <chrono>

using namespace std;

#define N 1024 // Total number of particles, you can adjust this value
#define K 60   // The Kth most distant particle

struct Particle {
    float x, y, z;
};



//===== computeDistances
__global__ void compute60thDistance(const Particle* particles, float* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float top60Distances[60];
    for (int k = 0; k < 60; k++) {
        top60Distances[k] = FLT_MAX;
    }

    Particle p = particles[i];
    for (int j = 0; j < n; j++) {
        if (i != j) {
            Particle q = particles[j];
            float dx = p.x - q.x;
            float dy = p.y - q.y;
            float dz = p.z - q.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);

            if (dist < top60Distances[59]) {
                top60Distances[59] = dist;
                
                // Use insertion sort to sort the top60Distances array.
                for (int k = 58; k >= 0; k--) {
                    if (top60Distances[k+1] < top60Distances[k]) {
                        float temp = top60Distances[k];
                        top60Distances[k] = top60Distances[k+1];
                        top60Distances[k+1] = temp;
                    } else {
                        break;  // Break early since the rest of the list is sorted.
                    }
                }
            }
        }
    }

    results[i] = top60Distances[59];  // 60th smallest distance (0-indexed)
}




void get60thDistance(const Particle* particles, float* results, int n) {
    float* d_results;
    cudaMalloc(&d_results, sizeof(float) * n);

    dim3 blockDim(256);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);

    compute60thDistance<<<gridDim, blockDim>>>(particles, d_results, n);

    cudaDeviceSynchronize();
    cudaMemcpy(results, d_results, sizeof(float) * n, cudaMemcpyDeviceToHost);

    cudaFree(d_results);
}






int main()
{
    Particle* h_particles = new Particle[N];
    Particle* d_particles;
    float* h_results = new float[N];
    
    // Populate h_particles with random data
    srand(static_cast<unsigned int>(time(NULL)));
    for (int i = 0; i < N; i++)
    {
        h_particles[i].x = static_cast<float>(rand()) / RAND_MAX * 1.0f; // For example: random values between 0 and 1
        h_particles[i].y = static_cast<float>(rand()) / RAND_MAX * 1.0f;
        h_particles[i].z = static_cast<float>(rand()) / RAND_MAX * 1.0f;
    }


    cudaMalloc(&d_particles, sizeof(Particle) * N);
    cudaMemcpy(d_particles, h_particles, sizeof(Particle) * N, cudaMemcpyHostToDevice);

    auto T_dU = std::chrono::high_resolution_clock::now();
    get60thDistance(d_particles, h_results, N);
    auto end_dU = std::chrono::high_resolution_clock::now();
    auto elapsed_dU = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dU - T_dU);
    cout << "T = " << elapsed_dU.count() * 1e-9 << endl;

    // h_results now contains the 60th distance for each particle ...
    
    for (int i = 0; i < 10; i++)
    {
    
      cout << h_results[i] << endl;
    
    }
    

    delete[] h_particles;
    delete[] h_results;
    cudaFree(d_particles);

    return 0;
}

