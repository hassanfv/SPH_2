#include <iostream>
#include <algorithm>
#include <random>
#include <utility>
#include <vector>
#include <map>

using namespace std;


pair<int, int> get_grid_cell(float x_p, float y_p, float max_dist, int GridSize)
{
  int cell_x = static_cast<int>(x_p * GridSize / max_dist);
  int cell_y = static_cast<int>(y_p * GridSize / max_dist);

  return make_pair(cell_x, cell_y);
}



vector<pair<int, int>> find_neighbor_cells(int cell_x, int cell_y, int max_x, int max_y) {
    vector<pair<int, int>> neighbors;

    // Define potential relative coordinates for neighbors
    pair<int, int> relative_coords[] = {
        {-1, -1}, {0, -1}, {1, -1},
        {-1,  0},          {1,  0},
        {-1,  1}, {0,  1}, {1,  1}
    };

    for (const auto& coord : relative_coords) {
        int neighbor_x = cell_x + coord.first;
        int neighbor_y = cell_y + coord.second;

        // Check if the neighbor coordinates are within grid limits
        if (0 <= neighbor_x && neighbor_x < max_x && 0 <= neighbor_y && neighbor_y < max_y) {
            neighbors.push_back({neighbor_x, neighbor_y});
            cout << neighbor_x << ", " << neighbor_y << endl;
        }
    }

    return neighbors;
}






int main()
{

  const int N = 100;
  const float beg = -1.0;
  const float end = 1.0;
  
  const int GridSize = 10;

  // Setting up random number generation
  random_device rd;
  mt19937 gen(rd());
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
  
  
  
  
  // Create a map to hold particles for each cell
  map<pair<int, int>, vector<int>> cell_particles;

  // Iterate over the particles and assign them to cells
  for (int i = 0; i < N; ++i)
  {
    pair<int, int> cell = get_grid_cell(x[i], y[i], max_dist, GridSize);
    cell_particles[cell].push_back(i);
  }

  // Print particles in each cell
  for (const auto& entry : cell_particles)
  {
    const auto& cell = entry.first;
    const auto& particles = entry.second;
    
    cout << "Cell (" << cell.first << ", " << cell.second << ") has particles: ";
    for (int index : particles)
    {
      cout << index << " ";
    }
    cout << endl;
  }
  
  
  

  return 0;
}





