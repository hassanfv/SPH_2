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
    const std::map<std::pair<int, int>, std::vector<int>>& cell_particles,
    std::vector<int>& cell_particles_offsets,
    std::vector<int>& cell_particles_values)
{
    // Clear existing data from output vectors
    cell_particles_offsets.clear();
    cell_particles_values.clear();

    // Start with an offset of 0
    cell_particles_offsets.push_back(0);


    for (const auto& kv : cell_particles) 
    {
        // Concatenate this cell's particle indices to the values array
        cell_particles_values.insert(cell_particles_values.end(), kv.second.begin(), kv.second.end());

        // Push the current size of the values array as the next cell's offset
        cell_particles_offsets.push_back(cell_particles_values.size());
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
  map<pair<int, int>, vector<int>> cell_particles;

  // Iterate over the particles and assign them to cells
  for (int i = 0; i < N; ++i)
  {
    pair<int, int> cell = get_grid_cell(x[i], y[i], max_dist, GridSize);
    cell_particles[cell].push_back(i);
  }
  //----------------------


  std::vector<int> cell_particles_offsets, cell_particles_values;
  generateCellParticlesArrays(cell_particles, cell_particles_offsets, cell_particles_values);

  //===== copying cell_particles_offsets, cell_particles_values to ARRAYS to use in GPU!
  int N_offsets = cell_particles_offsets.size();
  int N_values = cell_particles_values.size();
  
  int *cell_particles_offsetz = new int[N_offsets];
  int *cell_particles_valuez = new int[N_values];
  
  for (int i = 0; i < N_offsets; i++)
  {
    cell_particles_offsetz[i] = cell_particles_offsets[i];
    //cout << "ofsets = " << cell_particles_offsetz[i] << endl;
  }
  
  for (int i = 0; i < N_values; i++)
  {
    cell_particles_valuez[i] = cell_particles_values[i];
  }


  

  


  vector<vector<int>> all_neighboring_particles;
  // You can create a reference here if you find it easier to work with, but it's not strictly necessary.
  vector<vector<int>> &ref_all_neighboring_particles = all_neighboring_particles;  
  all_neighboring_particles.resize(N);  // Resize the outer vector to hold data for all N particles

  cout << "I'm here!!!!" << endl;




  ofstream outFile("data.bin", ios::binary);

  // Save N and GridSize
  outFile.write(reinterpret_cast<const char*>(&N), sizeof(N));
  outFile.write(reinterpret_cast<const char*>(&GridSize), sizeof(GridSize));

  // Save x and y arrays
  outFile.write(reinterpret_cast<const char*>(x), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(y), N * sizeof(float));

  // Save cell_particles_offsets and cell_particles_values
  int offsets_size = cell_particles_offsets.size();
  int values_size = cell_particles_values.size();

  outFile.write(reinterpret_cast<const char*>(&offsets_size), sizeof(offsets_size));
  outFile.write(reinterpret_cast<const char*>(cell_particles_offsets.data()), offsets_size * sizeof(int));

  outFile.write(reinterpret_cast<const char*>(&values_size), sizeof(values_size));
  outFile.write(reinterpret_cast<const char*>(cell_particles_values.data()), values_size * sizeof(int));

  outFile.close();





  return 0;
}





