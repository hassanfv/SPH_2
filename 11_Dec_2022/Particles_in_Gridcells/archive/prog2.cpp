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


//===== find_neighbor_cells
vector<pair<int, int>> find_neighbor_cells(int cell_x, int cell_y, int max_x, int max_y)
{
    vector<pair<int, int>> neighbors;

    // Define potential relative coordinates for neighbors
    pair<int, int> relative_coords[] = 
    {
        {-1, -1}, {0, -1}, {1, -1},
        {-1,  0},          {1,  0},
        {-1,  1}, {0,  1}, {1,  1}
    };

    for (const auto& coord : relative_coords)
    {
        int neighbor_x = cell_x + coord.first;
        int neighbor_y = cell_y + coord.second;

        // Check if the neighbor coordinates are within grid limits
        if (0 <= neighbor_x && neighbor_x < max_x && 0 <= neighbor_y && neighbor_y < max_y)
        {
            neighbors.push_back({neighbor_x, neighbor_y});
            //cout << neighbor_x << ", " << neighbor_y << endl;
        }
    }

    return neighbors;
}






int main()
{

  auto TT = std::chrono::high_resolution_clock::now();

  const int N = 50000; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
  
  int i_p = 20;
  float x_p = x[i_p];
  float y_p = y[i_p];
  
  
  pair<int, int> cell = get_grid_cell(x_p, y_p, max_dist, GridSize);
  
  int cell_x = cell.first;
  int cell_y = cell.second;
  
  cout << "For position (" << x_p << ", " << y_p << "):" << endl;
  cout << "Cell is (" << cell.first << ", " << cell.second << ")" << endl;
  cout << endl;
  
  vector<pair<int, int>> neighbors = find_neighbor_cells(cell_x, cell_y, GridSize, GridSize);
    
  cout << "Neighbors for cell (" << cell_x << ", " << cell_y << "):" << endl;
  for (const auto& neighbor : neighbors) 
  {
    cout << "(" << neighbor.first << ", " << neighbor.second << ")" << endl;
  }
  
  
  
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

  
  // For demonstration, let's find neighbors of particle with index i_p
  pair<int, int> particle_cell = get_grid_cell(x[i_p], y[i_p], max_dist, GridSize);
  vector<pair<int, int>> neighbor_cells = find_neighbor_cells(particle_cell.first, particle_cell.second, GridSize, GridSize);

  // Gather all particles in neighboring cells
  vector<int> neighboring_particles;
  
  // First, insert particles from the central cell (particle's own cell)
  if (cell_particles.find(particle_cell) != cell_particles.end()) 
  {
    neighboring_particles.insert(neighboring_particles.end(), cell_particles[particle_cell].begin(), cell_particles[particle_cell].end());
  }
  
  // Now inserting particles from neighboring cells
  for (const auto& cell : neighbor_cells) 
  {
    if (cell_particles.find(cell) != cell_particles.end()) 
    {
      neighboring_particles.insert(neighboring_particles.end(), cell_particles[cell].begin(), cell_particles[cell].end());
    }
  }

  
  // Print the neighboring particles of particle with index i_p
  cout << "The neighbors of particle " << i_p << " are: ";
  int kk = 0;
  for (int index : neighboring_particles) 
  {
    //cout << index << " ";
    kk ++;
  }
  cout << "N ngb = " << kk << endl;
  cout << endl;
  
  
  auto end_TT = std::chrono::high_resolution_clock::now();
  auto elapsed_TT = std::chrono::duration_cast<std::chrono::nanoseconds>(end_TT - TT);
  cout << "T_TT = " << elapsed_TT.count() * 1e-9 << endl;
  
  
  std::ofstream outFile("data.bin", std::ios::out | std::ios::binary);
  
  if (outFile.is_open())
  {
  // Write i_p
  outFile.write(reinterpret_cast<const char*>(&i_p), sizeof(i_p));

  // Write size of x array and data
  outFile.write(reinterpret_cast<const char*>(&N), sizeof(N));
  outFile.write(reinterpret_cast<const char*>(x), sizeof(float) * N);

  // Write size of y array and data
  outFile.write(reinterpret_cast<const char*>(&N), sizeof(N));
  outFile.write(reinterpret_cast<const char*>(y), sizeof(float) * N);

  // Write size of neighboring_particles vector and data
  int n_size = neighboring_particles.size();
  outFile.write(reinterpret_cast<const char*>(&n_size), sizeof(n_size));
  outFile.write(reinterpret_cast<const char*>(neighboring_particles.data()), sizeof(int) * n_size);

  outFile.close();
  }
  else
  {
    std::cerr << "Unable to open file for writing!" << std::endl;
  }

  

  return 0;
}





