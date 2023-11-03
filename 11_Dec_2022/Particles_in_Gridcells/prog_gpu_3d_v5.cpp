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


// Define constants for maximum particles in each grid cell
const int MAX_PARTICLES_PER_CELL = 500;  // This is a guess. Adjust based on your data.

// Define the GPU grid data structure
struct GridCell
{
    int particles[MAX_PARTICLES_PER_CELL];
    int count;
};


//===== getNeighborsParticles
__device__ int3 get_grid_cell_device(float x_p, float y_p, float z_p, float max_dist, int GridSize) {
    int cell_x = static_cast<int>(x_p * GridSize / max_dist);
    int cell_y = static_cast<int>(y_p * GridSize / max_dist);
    int cell_z = static_cast<int>(z_p * GridSize / max_dist);
    return make_int3(cell_x, cell_y, cell_z);
}



//populateGrid
__global__ void populateGrid(float *x, float *y, float *z, float max_dist, int GridSize, GridCell *grid, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        int3 cell = get_grid_cell_device(x[i], y[i], z[i], max_dist, GridSize);

        int linearIndex = cell.x * GridSize * GridSize + cell.y * GridSize + cell.z;

        int idx = atomicAdd(&(grid[linearIndex].count), 1);
        if (idx < MAX_PARTICLES_PER_CELL)
        {
            grid[linearIndex].particles[idx] = i;
        }
    }
}


//======= findNeighbors
__global__ void findNeighbors(float *x, float *y, float *z, float max_dist, int GridSize, GridCell *grid,
                              int* neighboringParticles, const int maxNeighbors, int *ngbCounts, int N) 
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



                        GridCell cell = grid[cell_idx];
                        for (int idx = 0; idx < cell.count && idx < MAX_PARTICLES_PER_CELL; idx++)
                        {
                            if (neighborsCount < maxNeighbors)
                            {
                                neighboringParticles[i * maxNeighbors + neighborsCount++] = cell.particles[idx];
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




int main()
{
    auto TT = std::chrono::high_resolution_clock::now();

    const int N = 2000000;
    const float beg = -1.0;
    const float end = 1.0;
    const int GridSize = 20;

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
    
    
    // GPU related declarations
    float *d_x, *d_y, *d_z;
    int *d_neighboringParticles, *d_ngbCounts;
    
    const int maxNeighbors = 500; // You'll need to determine a suitable value for this
    
    int *ngbCounts = new int[N](); // To store neighbors count for each particle
    
    GridCell* d_grid;
    checkCudaErrors(cudaMalloc((void**)&d_grid, GridSize * GridSize * GridSize * sizeof(GridCell)), "Allocate d_grid");
    checkCudaErrors(cudaMemset(d_grid, 0, GridSize * GridSize * GridSize * sizeof(GridCell)), "Memset d_grid");
    
    // Allocate GPU memory
    checkCudaErrors(cudaMalloc((void**)&d_x, N * sizeof(float)), "Allocate d_x");
    checkCudaErrors(cudaMalloc((void**)&d_y, N * sizeof(float)), "Allocate d_y");
    checkCudaErrors(cudaMalloc((void**)&d_z, N * sizeof(float)), "Allocate d_z");
    checkCudaErrors(cudaMalloc((void**)&d_neighboringParticles, N * maxNeighbors * sizeof(int)), "Allocate d_neighboringParticles");
    checkCudaErrors(cudaMalloc((void**)&d_ngbCounts, N * sizeof(int)), "Allocate d_ngbCounts");
    
    

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

    
    int blockSize = 256;                            // number of threads in a block
    int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid
    
    

    // Copy data to GPU
    checkCudaErrors(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice), "Copy x to d_x");
    checkCudaErrors(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice), "Copy y to d_y");
    checkCudaErrors(cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice), "Copy z to d_z");
    

    // Launch the populateGrid kernel
    populateGrid<<<gridSize, blockSize>>>(d_x, d_y, d_z, max_dist, GridSize, d_grid, N);

    cout << "Grid cells populated with particles indices." << endl;

    
    findNeighbors<<<gridSize, blockSize>>>(d_x, d_y, d_z, max_dist, GridSize, d_grid, d_neighboringParticles, maxNeighbors, d_ngbCounts, N);
    
    // Error check after kernel launch
    checkCudaErrors(cudaGetLastError(), "Kernel launch failure");
    checkCudaErrors(cudaDeviceSynchronize(), "Kernel synchronization failure");

    // Copy data back from GPU to host
    checkCudaErrors(cudaMemcpy(ngbCounts, d_ngbCounts, N * sizeof(int), cudaMemcpyDeviceToHost), "Copy d_ngbCounts to ngbCounts");

    /*
    for (int i = 0; i < 10; i++)
    {
      cout << "count = " << ngbCounts[i] << endl;
    }
    */
    
    
    
    // If you need the data back on the CPU:
    GridCell* h_grid = new GridCell[GridSize * GridSize * GridSize];
    checkCudaErrors(cudaMemcpy(h_grid, d_grid, GridSize * GridSize * GridSize * sizeof(GridCell), cudaMemcpyDeviceToHost), "Copy d_grid to h_grid");
    
    
    
    int *nCell;
    
    int i = 100;
    if (i < GridSize * GridSize * GridSize) // Check to ensure the index is within bounds
    {
      std::cout << "Cell " << i << " has " << h_grid[i].count << " particles: ";
      
      nCell = new int[h_grid[i].count];
      
      for (int j = 0; j < h_grid[i].count; ++j)
      {
          std::cout << h_grid[i].particles[j] << " ";
          nCell[j] = h_grid[i].particles[j];
      }
      std::cout << std::endl;
    } else
    {
      std::cout << "Cell index out of bounds." << std::endl;
    }

    

    // Open a binary file for writing
    std::ofstream outFile("data.bin", std::ios::binary);

    // Write the integer N
    outFile.write(reinterpret_cast<const char*>(&N), sizeof(int));

    // Write the arrays x, y, z
    outFile.write(reinterpret_cast<const char*>(x), N * sizeof(float));
    outFile.write(reinterpret_cast<const char*>(y), N * sizeof(float));
    outFile.write(reinterpret_cast<const char*>(z), N * sizeof(float));

    // Write nCell's data
    int nCellLength = h_grid[i].count; // This captures the number of elements in nCell
    outFile.write(reinterpret_cast<const char*>(&nCellLength), sizeof(int)); // First write the number of elements
    outFile.write(reinterpret_cast<const char*>(nCell), nCellLength * sizeof(int)); // Then write the actual elements

    // Close the binary file
    outFile.close();

    

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_neighboringParticles);
    cudaFree(d_ngbCounts);
    cudaFree(d_grid);

    // Free host memory
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] ngbCounts;
    delete[] h_grid;
    delete[] nCell;

    cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TT).count() << " ms" << endl;

    return 0;
}

