
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;

const int Grid = 50; // size of the grid
const float Mtot = 1.0; // total mass of the sphere

// Function to create the grid
void createGrid(float* x, float* y, float* z, int size)
{
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
  for (int i = 0; i < size; i++)
  {
    float r = sqrt(sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])); // Note the float sqrt!
    if (r > 0) {
      x[i] *= r;
      y[i] *= r;
      z[i] *= r;
    }
  }
  
  // Counting particles inside the unit sphere
  int number_particles = 0;
  for (int i = 0; i < size; i++)
  {
    if (sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) < 1.0) {
      number_particles++;
    }
  }
  
  
  float* xx = new float[number_particles];
  float* yy = new float[number_particles];
  float* zz = new float[number_particles];
  
  float* vx = new float[number_particles];
  float* vy = new float[number_particles];
  float* vz = new float[number_particles];
  
  float* h = new float[number_particles];
  
  float* eps = new float[number_particles];
  
  int *Typ = new int[number_particles];
  
  float* mass = new float[number_particles];
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
        
        vx[k] = 0.0f;
        vy[k] = 0.0f;
        vz[k] = 0.0f;
        
        h[k] = 0.1;
        eps[k] = h[k];
        
        Typ[k] = 0;
        
        mass[k] = particle_mass;
        Uthermal[k] = 0.05;
        k++;
      }
  }
  
  cout << "We use " << number_particles << " particles" << endl;
    
  
  //====== Output to a binary file ===========
  std::string filename = "IC_Evrard_" + std::to_string(number_particles) + ".bin";
  
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if(!out)
  {
    std::cerr << "Cannot open the file." << std::endl;
    return 1;  // or any other error handling
  }
  
  // Save N_tot at the start of the file
  out.write((char*)&number_particles, sizeof(int));

  out.write((char*)Typ, number_particles * sizeof(int));

  out.write((char*)xx, number_particles * sizeof(float));
  out.write((char*)yy, number_particles * sizeof(float));
  out.write((char*)zz, number_particles * sizeof(float));

  out.write((char*)vx, number_particles * sizeof(float));
  out.write((char*)vy, number_particles * sizeof(float));
  out.write((char*)vz, number_particles * sizeof(float));

  out.write((char*)Uthermal, number_particles * sizeof(float));
  out.write((char*)h, number_particles * sizeof(float));
  out.write((char*)eps, number_particles * sizeof(float));
  out.write((char*)mass, number_particles * sizeof(float));

  out.close();
  
  float G = 1.0;
  float L_AGN_code_unit = 1.0;
  float M_dot_in_code_unit = 1.0;
  float vin_in_code_unit = 1.0;
  float u_for_10K_Temp = 1.0;
  float m_sph_outflow = 1.0;
  float sigma_in_code_unit = 1.0;
  float UnitDensity_in_cgs = 1.0;
  float Unit_u_in_cgs = 1.0;
  float unitTime_in_s = 1.0;
  float unitLength_in_cm = 1.0;
  
  //===== Saving the parameters and constants! ========  
  std::ofstream outfile("params.txt");
  if (outfile.is_open()) 
  {
    outfile << filename << "\n";
    outfile << number_particles << "\n";
    outfile << number_particles << "\n";
    outfile << G << "\n";
    outfile << L_AGN_code_unit << "\n"; // Note will be multiplied by dt in the code.
    outfile << M_dot_in_code_unit << "\n"; // Note will be multiplied by dt in the code.
    outfile << vin_in_code_unit << "\n";
    outfile << u_for_10K_Temp << "\n";
    outfile << m_sph_outflow << "\n";
    outfile << sigma_in_code_unit << "\n";
    outfile << UnitDensity_in_cgs << "\n";
    outfile << Unit_u_in_cgs << "\n";
    outfile << unitTime_in_s << "\n";
    outfile << unitLength_in_cm << "\n";

    outfile.close();
  } else {
    std::cerr << "Unable to open file for writing!";
  }
    
    

}




