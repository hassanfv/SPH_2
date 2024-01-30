%%writefile test.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

__device__ const float PI = 3.14159265358979323846f;


//==========================================
//============ getDensity ==================
//==========================================
__global__ void getDensity(int *Typ, float *x, float *y, float *z, float *xpos, float *ypos, float *zpos,
                           float *mass, float *rhopos, float *h, float *hpos, int Npos, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < Npos)
  {

    float myPi = PI;

    float dx, dy, dz, rr, hij, sig, q, hij3;
    float WIij;
    float ss = 0.0f;

    for (int j = 0; j < N; j++)
    {
      if (Typ[j] == 0)
      {
        dx = xpos[i] - x[j];
        dy = ypos[i] - y[j];
        dz = zpos[i] - z[j];

        rr = sqrt(dx * dx + dy * dy + dz * dz);
        hij = 0.5f * (hpos[i] + h[j]);

        if (rr <= 2.0f * hij)
        {

          sig = 1.0 / myPi;
          q = rr / hij;
          hij3 = hij * hij * hij;
          WIij = 0.0f;

          if (q <= 1.0)
          {
            WIij = sig / hij3 * (1.0f - (3.0f / 2.0f) * q * q + (3.0f / 4.0f) * q * q * q);
          }

          if ((q > 1.0f) && (q <= 2.0))
          {
            WIij = sig / hij3 * (1.0f / 4.0f) * (2.0f - q) * (2.0f - q) * (2.0f - q);
          }

          ss += mass[j] * WIij;
        }
      }
    }
    rhopos[i] = ss + 1e-15; // for some positions it seems WIij is zero (could be a very low density regin!!!)
  }
}


//===== readBinaryFile
void readBinaryFile(const char* filename, int& N, int& N_ionFrac, int** Typ, float** x, float** y, float** z,
                    float** vx, float** vy, float** vz, float** rho, float** h, float** u, float** mass, float** ionFrac)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Read N and N_ionFrac
    file.read(reinterpret_cast<char*>(&N), sizeof(N));
    file.read(reinterpret_cast<char*>(&N_ionFrac), sizeof(N_ionFrac));

    // Allocate memory for arrays
    *Typ = new int[N];
    *x = new float[N];
    *y = new float[N];
    *z = new float[N];
    *vx = new float[N];
    *vy = new float[N];
    *vz = new float[N];
    *rho = new float[N];
    *h = new float[N];
    *u = new float[N];
    *mass = new float[N];
    *ionFrac = new float[N_ionFrac];

    // Read data into arrays
    file.read(reinterpret_cast<char*>(*Typ), N * sizeof(int));
    file.read(reinterpret_cast<char*>(*x), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*y), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*z), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*vx), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*vy), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*vz), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*rho), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*h), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*u), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*mass), N * sizeof(float));
    file.read(reinterpret_cast<char*>(*ionFrac), N_ionFrac * sizeof(float));

    file.close();
}




//===== getIonFrac
__global__ void getIonFrac(int *Typ, float *x, float *y, float *z, float *xpos, float *ypos, float *zpos,
                           float *rho, float *m, float *h, float *hpos, float *ionFrac, float *iFc,
                           int N, int Npos, int N_M)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < Npos)
  {

    float myPi = PI;

    float s, ss;
    float dx, dy, dz, rr, hij, sig, q, WIij, ionFrac_j_k;

    for (int k = 0; k < N_M; k++) // k represents the 14 species we have in our ionFrac.
    {
      s = 0.0f;
      ss = 0.0f;

      for (int j = 0; j < N; j++)
      {
        if (Typ[j] == 0)
        {
          dx = xpos[i] - x[j];
          dy = ypos[i] - y[j];
          dz = zpos[i] - z[j];
          rr = sqrt(dx*dx + dy*dy + dz*dz);

          hij = 0.5f * (hpos[i] + h[j]);

          sig = 1.0f/myPi;
          q = rr / hij;

          WIij = 0.0f;

          if (q <= 1.0f)
            WIij = sig / hij / hij / hij * (1.0 - (3.0/2.0) * q * q + (3.0/4.0) * q * q * q);

          if ((q > 1.0) && (q <= 2.0))
            WIij = sig / hij / hij / hij * (1.0/4.0) * (2.0 - q) * (2.0 - q) * (2.0 - q);
           
          ionFrac_j_k = ionFrac[j * N_M + k];
          s += m[j] * ionFrac_j_k / rho[j] * WIij;
          ss += m[j] / rho[j] * WIij;
        }
      }
      if (ss == 0)
      {
        iFc[i * N_M + k] = 1e-20;
      }
      else
      {
        iFc[i * N_M + k] = s/ss;
      }
    }
  }
}


int main()
{

  const char* filename = "G-0.111565.bin";

  //float unit_velocity_cgs = 4.5867e+06; // cm/s #!!!!!!!!!!!!!!!!!!!!!!!!
  //float unit_u = 2.10378e+13; //!!!!!!!!!!!!!!!!!!!!!!!!
  float unit_rho = 3.30981e-23; // !!!!!!!!!!!!!!!!!!!
  float unit_length = 3.086e+21; //!!!!!!!!!!!!!!!!!!!

  float XH = 0.7;

  //float kB = 1.3807e-16;
  //float mu = 0.61;
  float mH = 1.673534e-24;
  //float gamma = 5.0/3.0;

  int N, N_ionFrac;
  int* Typ;
  float *x, *y, *z, *vx, *vy, *vz, *rho, *h, *u, *mass, *ionFrac;

  readBinaryFile(filename, N, N_ionFrac, &Typ, &x, &y, &z, &vx, &vy, &vz, &rho, &h, &u, &mass, &ionFrac);
  
  int N_M = 14; // number of species included in the ionFrac array!
  
  int *d_Typ;
  float *d_x, *d_y, *d_z, *d_mass, *d_h, *d_rho, *d_ionFrac;
  
  //---- Allocate memory on the device
  cudaMalloc(&d_Typ, N * sizeof(int));

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_z, N * sizeof(float));
  
  cudaMalloc(&d_mass, N * sizeof(float));
  cudaMalloc(&d_h, N * sizeof(float));
  cudaMalloc(&d_rho, N * sizeof(float));
  
  cudaMalloc(&d_ionFrac, N_ionFrac * sizeof(float));
  
  //---- copy to the device
  cudaMemcpy(d_Typ, Typ, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_mass, mass, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h, h, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rho, rho, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_ionFrac, ionFrac, N_ionFrac * sizeof(float), cudaMemcpyHostToDevice);
  
  
  
  //-------------> Reading Clumps.bin <----------------
  std::ifstream file("Clumps.bin", std::ios::binary);
  if (!file)
  {
    std::cerr << "Cannot open file!" << std::endl;
    return 1;
  }

  int j = 2; // For example, if you want to read the 3rd sublist (0-indexed)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  std::vector<int> nx; // This will store the elements of the j-th sublist

  for (int i = 0; i <= j; ++i)
  {
    unsigned int length;
    file.read(reinterpret_cast<char*>(&length), sizeof(length));

    if (file.eof())
    {
      std::cerr << "Reached end of file before finding the " << j << "-th sublist." << std::endl;
      return 1;
    }

    std::vector<int> sublist(length);
    for (unsigned int k = 0; k < length; ++k)
    {
      int value;
      file.read(reinterpret_cast<char*>(&value), sizeof(value));
      if (i == j)
      { // If this is the j-th sublist, store the value in nx
        nx.push_back(value);
      }
    }
  }
  //--------------- End of Reading Clumps.bin --------------------
  
  float *nH_cgs = new float[N];
  for (int i = 0; i < N; i++)
  {
  nH_cgs[i] = rho[i] * unit_rho * XH / mH;
  }
  
  int clumpSize = nx.size();
  
  //---- Finding index with max(nH) in the clump ---> To make sure that the line of sight passes through the densest region of the clump!
  float maxi = 0.0f;
  int nxMax = 0;
  for (int i = 0; i < clumpSize; i++)
  {
    if (nH_cgs[nx[i]] > maxi)
    {
      maxi = nH_cgs[nx[i]];
      nxMax = nx[i];
    }
  }
  
  cout << "max nH_cgs = " << maxi << endl;
  
  //--------- drawing the line of sight -------------
  int Npos = 2000;
  
  float *xpos = new float[Npos];
  float *ypos = new float[Npos];
  float *zpos = new float[Npos];
  
  float *rpos = new float[Npos];

  float *hpos = new float[Npos];
  
  float *rhopos = new float[Npos];

  float p0[] = {0.0, 0.0, 0.0};
  float p1[] = {x[nxMax], y[nxMax], z[nxMax]};
  
  cout << "p1 = " << x[nxMax] << ", " << y[nxMax] << ", " << z[nxMax] << endl;


  float stp = 2.0f/Npos;
  float tt = 0.0;
  for (int i = 0; i < Npos; i++)
  {
    xpos[i] = p0[0] + (p1[0] - p0[0]) * tt;
    ypos[i] = p0[1] + (p1[1] - p0[1]) * tt;
    zpos[i] = p0[2] + (p1[2] - p0[2]) * tt;
    rpos[i] = sqrt(xpos[i] * xpos[i] + ypos[i] * ypos[i] + zpos[i] * zpos[i]); //-----> This is the line of sight!
    tt += stp;
  }
  
  //---- Setting hpos for all points along the line of sight to the mean h of all the particles in the clump.
  float s = 0.0f;
  for (int i = 0; i < clumpSize; i++)
  {
    s += h[nx[i]];
  }
  
  float mean_h = s / static_cast<float>(clumpSize);
  cout << "mean_h = " << mean_h << endl;
  
  for (int i = 0; i < Npos; i++)
  {
    hpos[i] = mean_h;
    rhopos[i] = 0.0;
  }
  //-------------------------------------------------
  
  int blockSize = 256;
  int gridSize = (Npos + blockSize - 1) / blockSize; // Number of blocks in a grid
  
  float *d_xpos, *d_ypos, *d_zpos, *d_hpos, *d_rhopos;
  //---- Allocate memory on the device for the line-of-sight
  cudaMalloc(&d_xpos, Npos * sizeof(float));
  cudaMalloc(&d_ypos, Npos * sizeof(float));
  cudaMalloc(&d_zpos, Npos * sizeof(float));
  cudaMalloc(&d_hpos, Npos * sizeof(float));
  cudaMalloc(&d_rhopos, Npos * sizeof(float));
  
  //---- copy to the device
  cudaMemcpy(d_xpos, xpos, Npos * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ypos, ypos, Npos * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_zpos, zpos, Npos * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hpos, hpos, Npos * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rhopos, rhopos, Npos * sizeof(float), cudaMemcpyHostToDevice);
  
  
  float *iFc = new float[Npos * N_M];
  
  float *d_iFc;
  
  cudaMalloc(&d_iFc, Npos * N_M * sizeof(float));
  cudaMemcpy(d_iFc, iFc, Npos * N_M * sizeof(float), cudaMemcpyHostToDevice);
  
  getIonFrac<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_xpos, d_ypos, d_zpos, d_rho, d_mass, d_h, d_hpos, d_ionFrac, d_iFc, N, Npos, N_M);
  cudaError_t launchError = cudaGetLastError();
  if (launchError != cudaSuccess)
  {
      cerr << "Kernel launch failed: " << cudaGetErrorString(launchError) << endl;
  }

  cudaDeviceSynchronize();
  cudaError_t syncError = cudaGetLastError();
  if (syncError != cudaSuccess)
  {
      cerr << "CUDA error after kernel execution: " << cudaGetErrorString(syncError) << endl;
  }
  
  cudaMemcpy(iFc, d_iFc, Npos * N_M * sizeof(float), cudaMemcpyDeviceToHost);
  
  getDensity<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_xpos, d_ypos, d_zpos, d_mass, d_rhopos, d_h, d_hpos, Npos, N);
  launchError = cudaGetLastError();
  if (launchError != cudaSuccess)
  {
      cerr << "Kernel launch failed: " << cudaGetErrorString(launchError) << endl;
  }

  cudaDeviceSynchronize();
  syncError = cudaGetLastError();
  if (syncError != cudaSuccess)
  {
      cerr << "CUDA error after kernel execution: " << cudaGetErrorString(syncError) << endl;
  }

  cudaMemcpy(rhopos, d_rhopos, Npos * sizeof(float), cudaMemcpyDeviceToHost);

  float *nHpos_cgs = new float[Npos];

  for (int i = 0; i < Npos; i++)
  {
    nHpos_cgs[i] = rhopos[i] * unit_rho * XH / mH;
  }


  /*
  for (int i = 0; i < Npos; i++)
  {
    cout << i << ", " << nHpos_cgs[i] << endl;
  }
  exit(0);
  */

  float dl;
  float *Ncol =new float[N_M];
  for (int i = 0; i < N_M; i++)
  {
    Ncol[i] = 0.0;
  }

  for (int k = 0; k < N_M; k++)
  {
    float ss = 0.0;
    for (int i = 0; i < Npos-1; i++)
    {
      dl = (rpos[i+1] - rpos[i]) * unit_length;
      ss += nHpos_cgs[i] * dl * iFc[i * N_M + k];
      //cout << i << ", " << dl << ", " << nHpos_cgs[i] << ", " << iFc[i * N_M + k] << endl;
    }
    Ncol[k] = ss;
  }
  
  for (int i = 0; i < N_M; i++)
  {
    cout << log10(Ncol[i]) << endl;
  }

}


