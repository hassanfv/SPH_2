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
    const map<pair<int, int>, vector<int>>& cell_particles, 
    const int GridSize,
    int* cell_particles_offsets,
    int* cell_particles_values)
{
    cell_particles_offsets[0] = 0;
    int valuesIndex = 0;
    int offsetsIndex = 1;

    for (int i = 0; i < GridSize; ++i) {
        for (int j = 0; j < GridSize; ++j) {
            pair<int, int> cell = {i, j};
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





//===== getNeighborsParticles
// This function will retrieve all the particles from neighboring cells for the cell of particle with index i_p
vector<int> getNeighborsParticles(int i_p, float *x, float *y, float max_dist, int GridSize, int* cell_particles_offsets, int* cell_particles_values)
{
  pair<int, int> particle_cell = get_grid_cell(x[i_p], y[i_p], max_dist, GridSize);

  vector<int> neighboringParticles;

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {

      int neighbor_x = particle_cell.first + dx;
      int neighbor_y = particle_cell.second + dy;

      // Make sure we are within grid boundaries
      if (neighbor_x >= 0 && neighbor_x < GridSize && neighbor_y >= 0 && neighbor_y < GridSize) {
        int cell_idx = neighbor_x * GridSize + neighbor_y;

        // Ensure we don't access out-of-bounds index
        if(cell_idx + 1 >= GridSize * GridSize) {
            continue;
        }

        int start_offset = cell_particles_offsets[cell_idx];
        int end_offset = cell_particles_offsets[cell_idx + 1];

        for (int idx = start_offset; idx < end_offset; ++idx) {
            neighboringParticles.push_back(cell_particles_values[idx]);
        }
      }
    }
  }

  return neighboringParticles;
}





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
  float *x = new float[N];
  float *y = new float[N];

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
  map<pair<int, int>, vector<int>> cell_particles_map;

  auto T_T = std::chrono::high_resolution_clock::now();

  // Iterate over the particles and assign them to cells
  for (int i = 0; i < N; ++i)
  {
    pair<int, int> cell = get_grid_cell(x[i], y[i], max_dist, GridSize);
    cell_particles_map[cell].push_back(i);
  }
  //----------------------
  auto end_T = std::chrono::high_resolution_clock::now();
  auto elapsed_T = std::chrono::duration_cast<std::chrono::nanoseconds>(end_T - T_T);
  cout << "T_T = " << elapsed_T.count() * 1e-9 << endl;
  

  // Allocate memory for cell_particles_offsets and cell_particles_values
  int* cell_particles_offsets = new int[GridSize * GridSize + 1]();
  int* cell_particles_values = new int[N](); // Assuming maximum N particles across all cells

  // Call the function
  
  generateCellParticlesArrays(cell_particles_map, GridSize, cell_particles_offsets, cell_particles_values);

  cout << "I'm here!!!!" << endl;


  int i_p = 150; // Example particle index
  vector<int> neighbors = getNeighborsParticles(i_p, x, y, max_dist, GridSize, cell_particles_offsets, cell_particles_values);
  // Print or process the neighboring particles as needed
  
  int k = 0;
  for(int idx : neighbors) {
    //cout << idx << " ";
    k++;
  }
  
  cout << "COUNTS = " << k << endl;




  // Saving to a binary file
  const string filename = "data.bin";
  ofstream outfile(filename, ios::binary);

  // Write N and GridSize
  outfile.write(reinterpret_cast<const char*>(&N), sizeof(N));
  outfile.write(reinterpret_cast<const char*>(&GridSize), sizeof(GridSize));

  // Write x and y arrays
  outfile.write(reinterpret_cast<const char*>(x), N * sizeof(float));
  outfile.write(reinterpret_cast<const char*>(y), N * sizeof(float));

  // Write neighbors
  int neighbors_size = neighbors.size();
  outfile.write(reinterpret_cast<const char*>(&neighbors_size), sizeof(neighbors_size));
  outfile.write(reinterpret_cast<const char*>(neighbors.data()), neighbors_size * sizeof(int));

  outfile.close();



  return 0;
}





