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
int getNeighborsParticles(int i_p, float *x, float *y, float *z, float max_dist, int GridSize, int* cell_particles_offsets, int* cell_particles_values) 
{
    tuple<int, int, int> particle_cell = get_grid_cell(x[i_p], y[i_p], z[i_p], max_dist, GridSize);

    vector<int> neighboringParticles;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {

                int neighbor_x = get<0>(particle_cell) + dx;
                int neighbor_y = get<1>(particle_cell) + dy;
                int neighbor_z = get<2>(particle_cell) + dz;

                if (neighbor_x >= 0 && neighbor_x < GridSize &&
                    neighbor_y >= 0 && neighbor_y < GridSize &&
                    neighbor_z >= 0 && neighbor_z < GridSize) 
                {
                    int cell_idx = neighbor_x * GridSize * GridSize + neighbor_y * GridSize + neighbor_z;

                    if (cell_idx + 1 >= GridSize * GridSize * GridSize) {
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
    }
    
    return neighboringParticles.size();
}




int main() {
    auto TT = std::chrono::high_resolution_clock::now();

    const int N = 1000000;
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
    auto genTT = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        tuple<int, int, int> grid_cell = get_grid_cell(x[i], y[i], z[i], max_dist, GridSize);
        cell_particles_map[grid_cell].push_back(i);
    }

    cout << "Grid cells populated with particles indices." << endl;

    int *cell_particles_offsets = new int[GridSize * GridSize * GridSize + 1]();
    int *cell_particles_values = new int[N];

    generateCellParticlesArrays(cell_particles_map, GridSize, cell_particles_offsets, cell_particles_values);
    cout << "Offsets and values arrays generated." << endl;

    int i_p = 0;

    int siz = getNeighborsParticles(i_p, x, y, z, max_dist, GridSize, cell_particles_offsets, cell_particles_values);

    cout << "size = " << siz << endl;
    cout << endl;


    delete[] x;
    delete[] y;
    delete[] z;
    delete[] cell_particles_offsets;
    delete[] cell_particles_values;

    cout << "Execution time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - TT).count() << " ms" << endl;

    return 0;
}

