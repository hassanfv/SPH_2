%%writefile test.cu

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <random>
#include <tuple>
#include <vector>
#include <map>
#include <chrono>

using namespace std;


//===== get_grid_cell
tuple<int, int, int> get_grid_cell(float x_p, float y_p, float z_p, float max_dist, int GridSize) {
    int cell_x = static_cast<int>(x_p * GridSize / max_dist);
    int cell_y = static_cast<int>(y_p * GridSize / max_dist);
    int cell_z = static_cast<int>(z_p * GridSize / max_dist);

    return make_tuple(cell_x, cell_y, cell_z);
}



//===== generateCellParticlesArrays
void generateCellParticlesArrays(
    const map<tuple<int, int, int>, vector<int>>& cell_particles,
    const int GridSize,
    int* cell_particles_offsets,
    int* cell_particles_values)
{
    cell_particles_offsets[0] = 0;
    int valuesIndex = 0;
    int offsetsIndex = 1;

    for (int i = 0; i < GridSize; ++i) {
        for (int j = 0; j < GridSize; ++j) {
            for (int k = 0; k < GridSize; ++k) {
                tuple<int, int, int> cell = {i, j, k};
                auto iter = cell_particles.find(cell);

                if (iter != cell_particles.end()) {
                    const vector<int>& particles = iter->second;
                    for (int p : particles) {
                        cell_particles_values[valuesIndex++] = p;
                    }
                    cell_particles_offsets[offsetsIndex++] = valuesIndex;
                } else {
                    cell_particles_offsets[offsetsIndex++] = valuesIndex;
                }
            }
        }
    }
}



//===== getNeighborsParticles
__device__ int3 get_grid_cell_device(float x_p, float y_p, float z_p, float max_dist, int GridSize) {
    int cell_x = static_cast<int>(x_p * GridSize / max_dist);
    int cell_y = static_cast<int>(y_p * GridSize / max_dist);
    int cell_z = static_cast<int>(z_p * GridSize / max_dist);
    return make_int3(cell_x, cell_y, cell_z);
}

//======= findNeighbors
__global__ void findNeighbors(float *x, float *y, float *z, float max_dist, int GridSize, int* cell_particles_offsets, 
                              int* cell_particles_values, int* neighboringParticles, const int maxNeighbors, int *ngbCounts, int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
    {
        int3 particle_cell = get_grid_cell_device(x[i], y[i], z[i], max_dist, GridSize);

        int neighborsCount = 0;

        for (int dx = -1; dx <= 1; ++dx) 
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    int neighbor_x = particle_cell.x + dx;
                    int neighbor_y = particle_cell.y + dy;
                    int neighbor_z = particle_cell.z + dz;

                    if (neighbor_x >= 0 && neighbor_x < GridSize &&
                        neighbor_y >= 0 && neighbor_y < GridSize &&
                        neighbor_z >= 0 && neighbor_z < GridSize) 
                    {
                        int cell_idx = neighbor_x * GridSize * GridSize + neighbor_y * GridSize + neighbor_z;

                        if (cell_idx + 1 >= GridSize * GridSize * GridSize)
                        {
                            continue;
                        }

                        int start_offset = cell_particles_offsets[cell_idx];
                        int end_offset = cell_particles_offsets[cell_idx + 1];

                        for (int idx = start_offset; idx < end_offset; ++idx)
                        {
                            if (neighborsCount < maxNeighbors)
                            {
                                neighboringParticles[i * maxNeighbors + neighborsCount++] = cell_particles_values[idx];
                            }
                        }
                    }
                }
            }
        }

        int kk = 0;
        for (int j = 0; j < neighborsCount; j++) 
        {
            int neighborIndex = neighboringParticles[i * maxNeighbors + j];

            kk++;

            // Other calculations...
        }

        ngbCounts[i] = kk;
    }  
}

//====== Helper function to check CUDA errors
void checkCudaErrors(cudaError_t cudaStatus, const char* msg) {
    if (cudaStatus != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " " << cudaGetErrorString(cudaStatus) << endl;
        exit(-1);
    }
}




int main() {
    auto TT = std::chrono::high_resolution_clock::now();

    const int N = 1000000;
    const float beg = -1.0;
    const float end = 1.0;
    const int GridSize = 50;

    // Setting up random number generation
    const unsigned int SEED = 42;
    mt19937 gen(SEED);
    uniform_real_distribution<> dis(beg, end);

    float *x = new float[N];
    float *y = new float[N];
    float *z = new float[N];

    for (int i = 0; i < N; ++i) {
        x[i] = dis(gen);
        y[i] = dis(gen);
        z[i] = dis(gen);
    }

    // Adjusting x, y, and z
    float minX = *min_element(x, x + N);
    float minY = *min_element(y, y + N);
    float minZ = *min_element(z, z + N);

    for (int i = 0; i < N; ++i) {
        x[i] -= minX;
        y[i] -= minY;
        z[i] -= minZ;
    }

    float maxX = *max_element(x, x + N);
    float maxY = *max_element(y, y + N);
    float maxZ = *max_element(z, z + N);

    float max_dist = max({maxX, maxY, maxZ});

    cout << "max_dist: " << max_dist << endl;
    
    map<tuple<int, int, int>, vector<int>> cell_particles_map;
    
    auto T_T = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        tuple<int, int, int> grid_cell = get_grid_cell(x[i], y[i], z[i], max_dist, GridSize);
        cell_particles_map[grid_cell].push_back(i);
    }
    
    auto end_T = std::chrono::high_resolution_clock::now();
    auto elapsed_T = std::chrono::duration_cast<std::chrono::nanoseconds>(end_T - T_T);
    cout << "T_T = " << elapsed_T.count() * 1e-9 << endl;

    cout << "Grid cells populated with particles indices." << endl;

    int *cell_particles_offsets = new int[GridSize * GridSize * GridSize + 1]();
    int *cell_particles_values = new int[N];

    generateCellParticlesArrays(cell_particles_map, GridSize, cell_particles_offsets, cell_particles_values);
    cout << "Offsets and values arrays generated." << endl;
  
    

    // GPU related declarations
    float *d_x, *d_y, *d_z;
    int *d_cell_particles_offsets, *d_cell_particles_values, *d_neighboringParticles, *d_ngbCounts;
    
    const int maxNeighbors = 1000; // You'll need to determine a suitable value for this
    
    int *ngbCounts = new int[N](); // To store neighbors count for each particle

    // Allocate GPU memory
    checkCudaErrors(cudaMalloc((void**)&d_x, N * sizeof(float)), "Allocate d_x");
    checkCudaErrors(cudaMalloc((void**)&d_y, N * sizeof(float)), "Allocate d_y");
    checkCudaErrors(cudaMalloc((void**)&d_z, N * sizeof(float)), "Allocate d_z");
    checkCudaErrors(cudaMalloc((void**)&d_cell_particles_offsets, (GridSize * GridSize * GridSize + 1) * sizeof(int)), "Allocate d_cell_particles_offsets");
    checkCudaErrors(cudaMalloc((void**)&d_cell_particles_values, N * sizeof(int)), "Allocate d_cell_particles_values");
    checkCudaErrors(cudaMalloc((void**)&d_neighboringParticles, N * maxNeighbors * sizeof(int)), "Allocate d_neighboringParticles");
    checkCudaErrors(cudaMalloc((void**)&d_ngbCounts, N * sizeof(int)), "Allocate d_ngbCounts");

    // Copy data to GPU
    checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice), "Copy x to d_x");
    checkCudaErrors(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice), "Copy y to d_y");
    checkCudaErrors(cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice), "Copy z to d_z");
    checkCudaErrors(cudaMemcpy(d_cell_particles_offsets, cell_particles_offsets, (GridSize * GridSize * GridSize + 1) * sizeof(int), cudaMemcpyHostToDevice), "Copy cell_particles_offsets to d_cell_particles_offsets");
    checkCudaErrors(cudaMemcpy(d_cell_particles_values, cell_particles_values, N * sizeof(int), cudaMemcpyHostToDevice), "Copy cell_particles_values to d_cell_particles_values");

    // Launch the kernel
    dim3 blockSize(256); 
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    findNeighbors<<<gridSize, blockSize>>>(d_x, d_y, d_z, max_dist, GridSize, d_cell_particles_offsets, d_cell_particles_values, d_neighboringParticles, maxNeighbors, d_ngbCounts, N);
    
    // Error check after kernel launch
    checkCudaErrors(cudaGetLastError(), "Kernel launch failure");
    checkCudaErrors(cudaDeviceSynchronize(), "Kernel synchronization failure");

    // Copy data back from GPU to host
    checkCudaErrors(cudaMemcpy(ngbCounts, d_ngbCounts, N * sizeof(int), cudaMemcpyDeviceToHost), "Copy d_ngbCounts to ngbCounts");

    for (int i = 0; i < 10; i++)
    {
      cout << "count = " << ngbCounts[i] << endl;
    }

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_cell_particles_offsets);
    cudaFree(d_cell_particles_values);
    cudaFree(d_neighboringParticles);
    cudaFree(d_ngbCounts);

    // Free host memory
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] cell_particles_offsets;
    delete[] cell_particles_values;
    delete[] ngbCounts;

    cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TT).count() << " ms" << endl;

    return 0;
}

