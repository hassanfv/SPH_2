#include <iostream>
#include <algorithm>
#include <fstream>
#include <random>
#include <utility>
#include <vector>
#include <map>
#include <chrono>

using namespace std;


//===== get_grid_cell
pair<int, int> get_grid_cell(float x_p, float y_p, float max_dist, int GridSize)
{
  int cell_x = static_cast<int>(x_p * GridSize / max_dist);
  int cell_y = static_cast<int>(y_p * GridSize / max_dist);

  return make_pair(cell_x, cell_y);
}


//===== generateCellParticlesArrays
void generateCellParticlesArrays(
    const map<pair<int, int>, int*>& cell_particles, // maps to arrays instead of vectors
    const int GridSize,
    const int maxParticles, // you need to provide an estimate for the maximum number of particles across all cells
    int* cell_particles_offsets,
    int* cell_particles_values)
{
    // We initialize with one 0 for the starting point of the very first cell
    cell_particles_offsets[0] = 0;
    int valuesIndex = 0;  // Keeps track of where we are in the cell_particles_values array
    int offsetsIndex = 1; // Starts at 1 because the first position is already set

    // Process each cell in cell_particles and fill the values array
    for (int i = 0; i < GridSize; ++i) {
        for (int j = 0; j < GridSize; ++j) {
            pair<int, int> cell = {i, j};
            auto iter = cell_particles.find(cell);

            if (iter != cell_particles.end()) {
                // This cell has particles
                int* particles = iter->second;
                int particleCount = iter->second[0]; // assuming first element of array gives count of particles
                for (int p = 1; p <= particleCount; ++p) {
                    cell_particles_values[valuesIndex++] = particles[p];
                }
                cell_particles_offsets[offsetsIndex++] = valuesIndex;
            } else {
                // This cell doesn't have particles, offset remains same as previous
                cell_particles_offsets[offsetsIndex++] = valuesIndex;
            }
        }
    }
}


/*
// CUDA kernel to compute neighboring particles for each particle
__global__ void cudaFunct(float *x, float *y, float max_dist, int GridSize, 
                          int *cell_particles_offsets, int *cell_particles_values, 
                          int *all_neighboring_particles_offsets, int *all_neighboring_particles_values,
                          int max_x, int max_y, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N)
        return;  // Check boundary condition

    float x_p = x[i];
    float y_p = y[i];
  
    int cell_x = static_cast<int>(x_p * GridSize / max_dist);
    int cell_y = static_cast<int>(y_p * GridSize / max_dist);
    int particle_cell_idx = cell_y * GridSize + cell_x;  // Convert 2D coordinates to 1D index

    // First, insert particles from the central cell (particle's own cell)
    int start_idx = cell_particles_offsets[particle_cell_idx];
    int end_idx = cell_particles_offsets[particle_cell_idx + 1];

    int all_neighboring_particles_start = all_neighboring_particles_offsets[i];

    for (int j = start_idx; j < end_idx; j++)
    {
        all_neighboring_particles_values[all_neighboring_particles_start++] = cell_particles_values[j];
    }

    // Now inserting particles from neighboring cells
    // Define potential relative coordinates for neighbors
    int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    for (int d = 0; d < 8; d++)
    {
        int neighbor_x = cell_x + dx[d];
        int neighbor_y = cell_y + dy[d];

        // Check if the neighbor coordinates are within grid limits
        if (0 <= neighbor_x && neighbor_x < max_x && 0 <= neighbor_y && neighbor_y < max_y)
        {
            int neighbor_cell_idx = neighbor_y * GridSize + neighbor_x;
            start_idx = cell_particles_offsets[neighbor_cell_idx];
            end_idx = cell_particles_offsets[neighbor_cell_idx + 1];

            for (int j = start_idx; j < end_idx; j++)
            {
                all_neighboring_particles_values[all_neighboring_particles_start++] = cell_particles_values[j];
            }
        }
    }
}
*/







int main()
{

  auto TT = std::chrono::high_resolution_clock::now();

  const int N = 1000000; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const float beg = -1.0;
  const float end = 1.0;
  
  const int GridSize = 20; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  // Setting up random number generation
  const unsigned int SEED = 42;
  mt19937 gen(SEED);
  uniform_real_distribution<> dis(beg, end);

  // Declaring arrays
  float x[N], y[N];

  // Generating random points
  for (int i = 0; i < N; ++i)
  {
    x[i] = dis(gen);
    y[i] = dis(gen);
  }

  // Finding and subtracting minimum values from x and y
  float minX = *min_element(x, x + N);
  float minY = *min_element(y, y + N);
  for (int i = 0; i < N; ++i)
  {
    x[i] -= minX;
    y[i] -= minY;
  }

  // Finding max_dist
  float maxX = *max_element(x, x + N);
  float maxY = *max_element(y, y + N);
  float max_dist = max(maxX, maxY);

  cout << "max_dist: " << max_dist << endl;
  
  
  //--------------------- Can be used by all particles !!!
  // Create a map to hold particles for each cell
  map<pair<int, int>, int*> cell_particles_map; // Create a new map that matches the function's signature

  // Iterate over the particles and assign them to cells
  for (int i = 0; i < N; ++i)
  {
    pair<int, int> cell = get_grid_cell(x[i], y[i], max_dist, GridSize);
    if(cell_particles_map.find(cell) == cell_particles_map.end())
    {
        cell_particles_map[cell] = new int[N+1]; // +1 to store the count at the 0th index
        cell_particles_map[cell][0] = 0; // initialize count
    }
    int count = cell_particles_map[cell][0];
    cell_particles_map[cell][count+1] = i;
    cell_particles_map[cell][0]++;
  }
  //----------------------
  

  // Allocate memory for cell_particles_offsets and cell_particles_values
  int* cell_particles_offsets = new int[GridSize * GridSize + 1];
  int* cell_particles_values = new int[N]; // Assuming maximum N particles across all cells

  // Call the function
  generateCellParticlesArrays(cell_particles_map, GridSize, N, cell_particles_offsets, cell_particles_values);

  

  cout << "I'm here!!!!" << endl;




  ofstream outFile("data.bin", ios::binary);

  // Save N and GridSize
  outFile.write(reinterpret_cast<const char*>(&N), sizeof(N));
  outFile.write(reinterpret_cast<const char*>(&GridSize), sizeof(GridSize));

  // Save x and y arrays
  outFile.write(reinterpret_cast<const char*>(x), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(y), N * sizeof(float));
  
  // Correct raw pointer usage:
  int offsets_size = GridSize * GridSize + 1; // The size of cell_particles_offsets is known
  int values_size = cell_particles_offsets[offsets_size - 1]; // Last value in offsets gives the total count

  outFile.write(reinterpret_cast<const char*>(&offsets_size), sizeof(offsets_size));
  outFile.write(reinterpret_cast<const char*>(cell_particles_offsets), offsets_size * sizeof(int));

  outFile.write(reinterpret_cast<const char*>(&values_size), sizeof(values_size));
  outFile.write(reinterpret_cast<const char*>(cell_particles_values), values_size * sizeof(int));

  outFile.close();





  return 0;
}





