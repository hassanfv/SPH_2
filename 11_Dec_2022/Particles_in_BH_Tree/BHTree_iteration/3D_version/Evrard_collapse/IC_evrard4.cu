%%writefile test.cu

#include <iostream>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <chrono>

using namespace std;

const int Grid = 100; // size of the grid
const float Mtot = 1.0; // total mass of the sphere


__global__ void getSmoothing(float *x, float *y, float *z, float *h, float *Nngb_previous, float coeff, int Nup, int Ndown, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N)
  {

    float h_new = 2.0f * h[i];
    int N_iter = 0;
    int k = 0;
    
    int k_pre = Nup; // Just a choice so that it does not activate in the first run!

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {

        dx = x[j] - x[i];
        dy = y[j] - y[i];
        dz = z[j] - z[i];
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          if (i == 505851)
            printf("%d\n", j);
            //printf("j, k, h_new = %d, %d, %f\n", j, k, h_new);

          k++;
        }
        
        
        //-----------
        if (k > Nup) // To stop unnecessary search after it reaches k > Nup!
        {
          break;
        }
        //-----------
      }

      //-----------
      if (((k < Ndown) && (k_pre > Nup)) || ((k > Nup) && (k_pre < Ndown))) // To prevent oscillation outside Nup and Ndown values!!
      {
        coeff = coeff / 2.0f;
      }
      //-----------

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * h[i];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * h[i];
      }

      k_pre = k;

      N_iter++;
      if (N_iter > 500)
      {
        printf("MAX N_iter in the smoothing_h kernel reached !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        break;
      }
    }

    
    if (i == 505851)
      printf("k = %d\n", k);

    Nngb_previous[i] = k;
    h[i] = 0.5 * h_new;

  }
}



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
  
  //float* eps = new float[number_particles];
  
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
        
        Typ[k] = 0;
        
        mass[k] = particle_mass;
        Uthermal[k] = 0.05;
        k++;
      }
  }
  
  cout << "We use " << number_particles << " particles" << endl;
  
  
  
  //======= determining the smoothing length h ======
  
  int N = number_particles;
  
  float *h, *d_h, *d_x, *d_y, *d_z, *Nngb, *d_Nngb;

  h = new float[N];
  cudaMalloc(&d_h, N * sizeof(float));
  
  Nngb = new float[N];
  cudaMalloc(&d_Nngb, N * sizeof(float));
  
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_z, N * sizeof(float));
  
  for (int i = 0; i < N; i++)
  {
    h[i] = 0.025; // initial estimate (use clever guess)!
    Nngb[i] = 0.0;
  }
  
  cudaMemcpy(d_h, h, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Nngb, Nngb, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_x, xx, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, yy, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, zz, N * sizeof(float), cudaMemcpyHostToDevice);
  
  int Nngb_ref = 64;
  int Nup = Nngb_ref + 5;
  int Ndown = Nngb_ref - 5;
  
  float coeff = 0.02;

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  getSmoothing<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_Nngb, coeff, Nup, Ndown, N);
  cudaDeviceSynchronize();
  
  cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);
  
  
  //===== defining grav. epsilon ======
  float *eps = new float[N];
  
  for (int i = 0; i < N; i++)
  {
    eps[i] = h[i];
  }
  

  cout << "h[505851] = " << h[505851] << endl;
  
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




