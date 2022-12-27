
#include <iostream>
#include <cmath>
#include <fstream>
#include "myPhotoLibsGPU.h"

using namespace std;


__global__ void createCoolingGrid(float *rhoGrid, float *uGrid, float *res1,
                                  float *res2, float *res3, float *res4,
                                  float dt, float XH, int N_rho, int N_u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if(i < N_rho)
  {
    float ux, delta_u;
    int k = i * N_u;

    for(int j = 0; j < N_u; j++)
    {
      ux = DoCooling(rhoGrid[i], uGrid[j], dt, XH);
      delta_u = uGrid[j] - ux;

      res1[k] = uGrid[j];
      res2[k] = rhoGrid[i];
      res3[k] = dt;
      res4[k] = delta_u;

      k++;
    }
  }
}


const float XH = 0.76f;
const float mH = 1.6726e-24; // gram
const float dt  = 500.0f * 3600.0f * 24.0f * 365.24f; // 500 YEARS.

const int N_rho = 800;
const int N_u = 800;
const int N = N_rho * N_u;

int main(){

  float Tmin = 1e4;
  float Tmax = 1e6;

  float stp_T = (log10(Tmax) - log10(Tmin)) / N_u;

  float *Tgrid = new float[N_u];

  for(int i = 0; i < N_u; i++)
  {
    Tgrid[i] = pow(10, (log10(Tmin) + i * stp_T));
  }

  //-------- Converting T to u.
  /* This value is not very important. We just want to have a grid for u !! You could put nHcgs = 0.1, or 0.01, or ... !!! */
  float nHcgs = 1.0; //  cm^-3

  float *uGrid = new float[N_u];

  for(int i = 0; i < N_u; i++)
  {
    uGrid[i] = convert_Temp_to_u(Tgrid[i], nHcgs, XH);
  }

  float *d_uGrid;
  cudaMalloc(&d_uGrid, N_u*sizeof(float));
  cudaMemcpy(d_uGrid, uGrid, N_u*sizeof(float), cudaMemcpyHostToDevice);
  //-------- T to u conversion DONE !

  float nH_min = 1e-4;
  float nH_max = 1e3;
  float rho_min = nH_min * mH;
  float rho_max = nH_max * mH;

  float stp_rho = (log10(rho_max) - log10(rho_min)) / N_rho;

  float *rhoGrid = new float[N_rho];

  for(int i = 0; i < N_rho; i++)
  {
    rhoGrid[i] = pow(10, (log10(rho_min) + i * stp_rho));
  }

  float *d_rhoGrid;
  cudaMalloc(&d_rhoGrid, N_rho*sizeof(float));
  cudaMemcpy(d_rhoGrid, rhoGrid, N_rho*sizeof(float), cudaMemcpyHostToDevice);

  //---- Declaring the res arrays.
  float *res1 = new float[N];
  float *res2 = new float[N];
  float *res3 = new float[N];
  float *res4 = new float[N];

  for(int i = 0; i < N; i ++)
  {
    res1[i] = 0.0f;
    res2[i] = 0.0f;
    res3[i] = 0.0f;
    res4[i] = 0.0f;
  }

  float *d_res1, *d_res2, *d_res3, *d_res4; 

  cudaMalloc(&d_res1, N*sizeof(float));
  cudaMalloc(&d_res2, N*sizeof(float));
  cudaMalloc(&d_res3, N*sizeof(float));
  cudaMalloc(&d_res4, N*sizeof(float));

  cudaMemcpy(d_res1, res1, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res1, res2, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res1, res3, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_res1, res4, N*sizeof(float), cudaMemcpyHostToDevice);
  // ---- End of res arrays declaration.

  int blockSize = 256; // number of threads in a block
  int gridSize = (N_rho + blockSize - 1) / blockSize; // Number of blocks in a grid

  createCoolingGrid<<<gridSize, blockSize>>>(d_rhoGrid, d_uGrid, d_res1,
                                             d_res2, d_res3, d_res4, dt,
                                             XH, N_rho, N_u);
  cudaDeviceSynchronize();

  cudaMemcpy(res1, d_res1, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(res2, d_res2, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(res3, d_res3, N*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(res4, d_res4, N*sizeof(float), cudaMemcpyDeviceToHost);

  /*
  for(int i = 0; i < 10; i++)
  {
    cout << res1[i] << ", " << res2[i] << ", " << res3[i] << ", " << res4[i] << endl;
  }
  */

  ofstream outfile("CoolingGrid.csv");

  outfile << "u" << "," << "rho" << "," << "dt" << "," << "delta_u" << endl; //header !

  for(int i = 0; i < N; i++){
    outfile << res1[i] << "," << res2[i] << "," << res3[i] << "," << res4[i] << endl;
  }


  delete[] Tgrid;
  delete[] uGrid;
  delete[] rhoGrid;
  delete[] res1;
  delete[] res2;
  delete[] res3;
  delete[] res4;

  cudaFree(d_uGrid); cudaFree(d_rhoGrid); cudaFree(d_res1);
  cudaFree(d_res2); cudaFree(d_res3); cudaFree(d_res4);

}
