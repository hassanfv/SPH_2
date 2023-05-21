#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <mpi.h>

using namespace std;


const float G = 1.0f;
const float Gamma = 5.0f / 3.0f;
float visc_alpha = 1.0f;

int Ndown = 64 - 5;
int Nup = 64 + 5;
float Nngb_f = 64.0f;
float coeff = 0.001f;

float dt = 1e-2;

int N = 10000;

int N_tot = 11000;


int main(int argc, char** argv) 
{
    MPI_Init(&argc, &argv);

    int nCPUs;
    MPI_Comm_size(MPI_COMM_WORLD, &nCPUs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    
        if (rank == 0)
    {
        std::vector<int> Typvec(N_tot);
        std::vector<float> xvec(N_tot);
        std::vector<float> yvec(N_tot);
        std::vector<float> zvec(N_tot);
        std::vector<float> vxvec(N_tot);
        std::vector<float> vyvec(N_tot);
        std::vector<float> vzvec(N_tot);
        std::vector<float> massvec(N_tot);
        std::vector<float> hvec(N_tot);
        std::vector<float> epsvec(N_tot);
        std::vector<float> uvec(N_tot);

        // Check if the binary file exists
        std::ifstream file("IC_000k.bin", std::ios::binary);
        if (!file)
        {
            std::cout << "Could not open the binary file." << std::endl;
        }
        file.close();

        // Open the binary file for reading
        file.open("IC_000k.bin", std::ios::binary);
        if (file)
        {
            // Read the first array
            file.read(reinterpret_cast<char *>(Typvec.data()), sizeof(int) * Typvec.size());

            file.read(reinterpret_cast<char *>(xvec.data()), sizeof(float) * xvec.size());
            file.read(reinterpret_cast<char *>(yvec.data()), sizeof(float) * yvec.size());
            file.read(reinterpret_cast<char *>(zvec.data()), sizeof(float) * zvec.size());

            file.read(reinterpret_cast<char *>(vxvec.data()), sizeof(float) * vxvec.size());
            file.read(reinterpret_cast<char *>(vyvec.data()), sizeof(float) * vyvec.size());
            file.read(reinterpret_cast<char *>(vzvec.data()), sizeof(float) * vzvec.size());

            file.read(reinterpret_cast<char *>(massvec.data()), sizeof(float) * massvec.size());
            file.read(reinterpret_cast<char *>(hvec.data()), sizeof(float) * hvec.size());
            file.read(reinterpret_cast<char *>(epsvec.data()), sizeof(float) * epsvec.size());
            file.read(reinterpret_cast<char *>(uvec.data()), sizeof(float) * uvec.size());

            // Close the file
            file.close();
        }
        else
        {
            std::cout << "Failed to open the file." << std::endl;
        }
        
        x = new float[N];
  y = new float[N];
  z = new float[N];

  vx = new float[N];
  vy = new float[N];
  vz = new float[N];

  accx = new float[N];
  accy = new float[N];
  accz = new float[N];

  mass = new float[N];
  h = new float[NG];
  hprevious = new float[NG];
  rho = new float[N];
  eps = new float[N];
  P = new float[NG];
  csnd = new float[NG];

  divV = new float[NG];
  curlV = new float[NG];

  accx_sph = new float[NG];
  accy_sph = new float[NG];
  accz_sph = new float[NG];

  accx_tot = new float[N];
  accy_tot = new float[N];
  accz_tot = new float[N];

  abs_acc_g = new float[N];
  abs_acc_tot = new float[N];
  v_sig = new float[N];
  dh_dt = new float[NG];

  u = new float[NG];
  dudt = new float[NG];
  uprevious = new float[NG];
  utprevious = new float[NG];

  Nngb_previous = new float[NG];
        
        }
    
    
    
    
    return 0;
}
