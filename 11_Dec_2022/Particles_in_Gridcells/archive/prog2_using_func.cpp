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


//===== funct
void funct(float *x, float *y, float max_dist, int GridSize, map<pair<int, int>, vector<int>> &cell_particles,
           vector<vector<int>> *pAllNeighboringParticles, int max_x, int max_y, int N)
{
  // Ensure the pointer isn't null
  if (!pAllNeighboringParticles)
      return;

  vector<vector<int>>& all_neighboring_particles = *pAllNeighboringParticles;  // Dereference the pointer to get the actual vector

  for (int i = 0; i < N; i++)
  {
    float x_p = x[i];
    float y_p = y[i];
  
    int cell_x = static_cast<int>(x_p * GridSize / max_dist);
    int cell_y = static_cast<int>(y_p * GridSize / max_dist);
    pair<int, int> particle_cell = {cell_x, cell_y};

    // First, insert particles from the central cell (particle's own cell)
    if (cell_particles.find(particle_cell) != cell_particles.end()) 
    {
        all_neighboring_particles[i].insert(all_neighboring_particles[i].end(), cell_particles[particle_cell].begin(), cell_particles[particle_cell].end());
    }

    // Now inserting particles from neighboring cells
    // Define potential relative coordinates for neighbors
    pair<int, int> relative_coords[] = 
    {
        {-1, -1}, {0, -1}, {1, -1},
        {-1,  0},          {1,  0},
        {-1,  1}, {0,  1}, {1,  1}
    };

    for (const auto &coord : relative_coords)
    {
      int neighbor_x = cell_x + coord.first;
      int neighbor_y = cell_y + coord.second;

      // Check if the neighbor coordinates are within grid limits
      if (0 <= neighbor_x && neighbor_x < max_x && 0 <= neighbor_y && neighbor_y < max_y)
      {
      
        pair<int, int> cell_tmp = {neighbor_x, neighbor_y};
        if (cell_particles.find(cell_tmp) != cell_particles.end())
          {
            all_neighboring_particles[i].insert(all_neighboring_particles[i].end(), cell_particles[cell_tmp].begin(), cell_particles[cell_tmp].end());
          }
      }
    }
  }
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
  
  
  auto T_T = std::chrono::high_resolution_clock::now();
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
  auto end_T = std::chrono::high_resolution_clock::now();
  auto elapsed_T = std::chrono::duration_cast<std::chrono::nanoseconds>(end_T - T_T);
  cout << "T_T1 = " << elapsed_T.count() * 1e-9 << endl;
  


  vector<vector<int>> all_neighboring_particles;
  // You can create a reference here if you find it easier to work with, but it's not strictly necessary.
  vector<vector<int>> &ref_all_neighboring_particles = all_neighboring_particles;  
  all_neighboring_particles.resize(N);  // Resize the outer vector to hold data for all N particles

  cout << "I'm here!!!!" << endl;



  auto T_T2 = std::chrono::high_resolution_clock::now();
  funct(x, y, max_dist, GridSize, cell_particles, &all_neighboring_particles, GridSize, GridSize, N);
  auto end_T2 = std::chrono::high_resolution_clock::now();
  auto elapsed_T2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_T2 - T_T2);
  cout << "T_T2 = " << elapsed_T2.count() * 1e-9 << endl;
  
  
  

  int i_p = 0;
  
  vector<int> n_ngb = all_neighboring_particles[i_p];
  
  /*
  for (int i = 0; i < n_ngb.size(); i++)
  {
    cout << "n_ngb[" << i << "] = " << n_ngb[i] << endl;
  }
  */
  

 // Save to binary file
  ofstream outFile("data.bin", ios::binary);

  // Save N
  outFile.write(reinterpret_cast<const char*>(&N), sizeof(N));
  
  // Save i_p
  outFile.write(reinterpret_cast<const char*>(&i_p), sizeof(i_p));
  
  // Save size of n_ngb and then each element of n_ngb
  int size_n_ngb = n_ngb.size();
  outFile.write(reinterpret_cast<const char*>(&size_n_ngb), sizeof(size_n_ngb));
  outFile.write(reinterpret_cast<const char*>(n_ngb.data()), size_n_ngb * sizeof(int));

  // Save x and y arrays
  outFile.write(reinterpret_cast<const char*>(x), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(y), N * sizeof(float));
  
  outFile.close();
  

  return 0;
}





