




// Define constants for maximum particles in each grid cell
const int MAX_PARTICLES_PER_CELL = 100;  // This is a guess. Adjust based on your data.

// Define the GPU grid data structure
struct GridCell
{
    int particles[MAX_PARTICLES_PER_CELL];
    int count;
};



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

// In your main function:
GridCell* d_grid;
checkCudaErrors(cudaMalloc((void**)&d_grid, GridSize * GridSize * GridSize * sizeof(GridCell)), "Allocate d_grid");
checkCudaErrors(cudaMemset(d_grid, 0, GridSize * GridSize * GridSize * sizeof(GridCell)), "Memset d_grid");

// Launch the kernel
populateGrid<<<gridSize, blockSize>>>(d_x, d_y, d_z, max_dist, GridSize, d_grid, N);

// If you need the data back on the CPU:
GridCell* h_grid = new GridCell[GridSize * GridSize * GridSize];
checkCudaErrors(cudaMemcpy(h_grid, d_grid, GridSize * GridSize * GridSize * sizeof(GridCell), cudaMemcpyDeviceToHost), "Copy d_grid to h_grid");

// ... use h_grid for further CPU processing ...

delete[] h_grid;
cudaFree(d_grid);

