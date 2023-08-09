//%%writefile test.cu
#include <stdio.h>
#include <cuda.h>

struct Particle {
    float x, y, z;
};

__global__ void powerTwoKernel(Particle *particles, int count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < count) {
        particles[idx].x *= particles[idx].x;
        particles[idx].y *= particles[idx].y;
        particles[idx].z *= particles[idx].z;
    }
}

int main() {
    const int count = 10;
    Particle *h_particles = new Particle[count];
    Particle *d_particles;

    // Initialize particles in host memory
    for (int i = 0; i < count; i++) {
        h_particles[i].x = 1.0f+i;
        h_particles[i].y = 2.0f+i;
        h_particles[i].z = 3.0f+i;
    }

    // Allocate device memory
    cudaMalloc(&d_particles, count * sizeof(Particle));

    // Copy particles to device
    cudaMemcpy(d_particles, h_particles, count * sizeof(Particle), cudaMemcpyHostToDevice);

    // Execute kernel
    powerTwoKernel<<<1, count>>>(d_particles, count);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy particles back to host
    cudaMemcpy(h_particles, d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);

    // Now the particles array contains the particles with x, y, z raised to the power of 2.
    // Print the values of x, y, z for each particle
    for (int i = 0; i < count; i++) {
        printf("Particle %d: x=%f, y=%f, z=%f\n", i, h_particles[i].x, h_particles[i].y, h_particles[i].z);
    }

    // Clean up
    delete[] h_particles;
    cudaFree(d_particles);

    return 0;
}

