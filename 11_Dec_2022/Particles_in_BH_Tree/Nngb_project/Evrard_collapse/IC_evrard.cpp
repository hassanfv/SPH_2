
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;

const int Grid = 50; // size of the grid
const float Mtot = 1.0; // total mass of the sphere

// Function to create the grid
void createGrid(float* x, float* y, float* z, int size) {
    float step = 1.0 / (Grid / 2.0);
    for (int i = 0; i < size; i++) {
        int ix = i / (Grid * Grid);
        int iy = (i / Grid) % Grid;
        int iz = i % Grid;

        x[i] = (ix - (Grid / 2 - 0.5)) * step;
        y[i] = (iy - (Grid / 2 - 0.5)) * step;
        z[i] = (iz - (Grid / 2 - 0.5)) * step;
    }
}




int main() {
    int size = Grid * Grid * Grid;
    float* x = new float[size];
    float* y = new float[size];
    float* z = new float[size];

    createGrid(x, y, z, size);
    
    // Stretching initial conditions to get 1/r density distribution
    for (int i = 0; i < size; i++) {
        float r = sqrt(sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])); // Note the float sqrt!
        if (r > 0) {
            x[i] *= r;
            y[i] *= r;
            z[i] *= r;
        }
    }
    
    // Counting particles inside the unit sphere
    int number_particles = 0;
    for (int i = 0; i < size; i++) {
        if (sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) < 1.0) {
            number_particles++;
        }
    }
    
    
    float* xx = new float[number_particles];
    float* yy = new float[number_particles];
    float* zz = new float[number_particles];
    
    float* Mass = new float[number_particles];
    float* Uthermal = new float[number_particles];

    float particle_mass = Mtot / number_particles;

    int k = 0;
    for (int i = 0; i < size; i++)
    {
      if (sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) < 1.0)
        {
          xx[k] = x[i];
          yy[k] = y[i];
          zz[k] = z[i];
          
          Mass[k] = particle_mass;
          Uthermal[k] = 0.05;
          k++;
        }
    }
    
    cout << "We use " << number_particles << " particles" << endl;
    

    // Write to a binary file
    ofstream outFile("evrard_collapse_data.bin", ios::out | ios::binary);

    // Writing number_particles
    outFile.write(reinterpret_cast<const char*>(&number_particles), sizeof(number_particles));

    // Writing x, y, z arrays
    for (int i = 0; i < number_particles; ++i) {
        outFile.write(reinterpret_cast<const char*>(&xx[i]), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&yy[i]), sizeof(float));
        outFile.write(reinterpret_cast<const char*>(&zz[i]), sizeof(float));
    }

    outFile.close();


    // Clean up
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] Mass;
    delete[] Uthermal;


}




