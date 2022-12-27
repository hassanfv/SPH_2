
#include <iostream>
#include <cmath>
#include <fstream>
#include "myPhotoLibsGPU.h"

using namespace std;


__global__ void createCoolingGrid(float *rhoGX, float *uGX, float *res1,
                                  float *res2, float *res3, float *res4,
                                  float dt, float XH, int N_rho_beg,
                                  int N_rho_end, int N_u)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if((i >= N_rho_beg) && (i < N_rho_end))
  {

    float ux, delta_u;
    int k = i * N_u;

    for(int j = 0; j < N_u; j++)
    {
      ux = DoCooling(rhoGX[i-N_rho_beg], uGX[j], dt, XH);
      delta_u = uGX[j] - ux;

      res1[k] = uGX[j];
      res2[k] = rhoGX[i-N_rho_beg];
      res3[k] = dt;
      res4[k] = delta_u;

      k++;
    }
  }
}


const float XH = 0.76;
const float mH = 1.6726e-24; // gram
const float dt  = 500.0f * 3600.0f * 24.0f * 365.24f; // 500 YEARS.

const int N_rho = 1000;
const int N_u = 1000;
const int N = N_rho * N_u;

int main(){

  float Tmin = 1e4;
  float Tmax = 1e6;

  float stp_T = (log10(Tmax) - log10(Tmin)) / N_u;

  float *Tgrid;

  cudaMallocManaged(&Tgrid, N_u*sizeof(float));

  for(int i = 0; i < N_u; i++)
  {
    Tgrid[i] = pow(10, (log10(Tmin) + i * stp_T));
  }

  //-------- Converting T to u.
  /* This value is not very important. We just want to have a grid for u !! You could put nHcgs = 0.1, or 0.01, or ... !!! */
  float nHcgs = 1.0; //  cm^-3

  float *uGrid;

  cudaMallocManaged(&uGrid, N_u*sizeof(float));

  for(int i = 0; i < N_u; i++)
  {
    uGrid[i] = convert_Temp_to_u(Tgrid[i], nHcgs, XH);
  }

  cudaFree(Tgrid);
  //-------- T to u conversion DONE !

  float nH_min = 1e-4;
  float nH_max = 1e3;
  float rho_min = nH_min * mH;
  float rho_max = nH_max * mH;

  float stp_rho = (log10(rho_max) - log10(rho_min)) / N_rho;

  float *rhoGrid;

  cudaMallocManaged(&rhoGrid, N_rho*sizeof(float));

  for(int i = 0; i < N_rho; i++)
  {
    rhoGrid[i] = pow(10, (log10(rho_min) + i * stp_rho));
  }

  //---- Declaring the res arrays.

  float *res1, *res2, *res3, *res4;

  cudaMallocManaged(&res1, N*sizeof(float));
  cudaMallocManaged(&res2, N*sizeof(float));
  cudaMallocManaged(&res3, N*sizeof(float));
  cudaMallocManaged(&res4, N*sizeof(float));

  for(int i = 0; i < N; i ++)
  {
    res1[i] = 0.0f;
    res2[i] = 0.0f;
    res3[i] = 0.0f;
    res4[i] = 0.0f;
  }

  // ---- End of res arrays declaration.

  ofstream outfile("CoolingGrid_UM.csv");
  outfile << "u" << "," << "rho" << "," << "dt" << "," << "delta_u" << endl; //header !

  //------ Splitting rhoGrid (I) --------

  for(int j = 0; j < 4; j++)
  {

    int N_rho_beg = j * 250;
    int N_rho_end = (j+1) * 250;
    int N_uX = 1000;

    int N_rhoX = N_rho_end - N_rho_beg;
    
    int NN = N_rhoX * N_uX;

    float *uGX, *rhoGX;
    cudaMallocManaged(&rhoGX, N_rhoX*sizeof(float));
    cudaMallocManaged(&uGX, N_uX*sizeof(float));

    for(int i = 0; i < N_uX; i++)
    {
      uGX[i] = uGrid[i];
    }

    for(int i = N_rho_beg; i < (N_rho_beg+N_rhoX); i++)
    {
      rhoGX[i-N_rho_beg] = rhoGrid[i];
    }

    int blockSize = 256; // number of threads in a block
    int gridSize = 1000;// (N_rhoX + blockSize - 1) / blockSize; // Number of blocks in a grid

    createCoolingGrid<<<gridSize, blockSize>>>(rhoGX, uGX, res1, res2,
                                              res3, res4, dt, XH,
                                              N_rho_beg, N_rho_end, N_uX);
    cudaDeviceSynchronize();

    cout << "H = " << j*NN << ", " << (j+1)*NN << endl;
    cout << N_rho_beg << ", " << N_rho_end << endl;

    for(int i = j*NN; i < (j+1)*NN; i++){
      outfile << res1[i] << "," << res2[i] << "," << res3[i] << "," << res4[i] << endl;
    }

    for(int i = j*NN; i < j*NN+5; i++){
      cout << res1[i] << "," << res2[i] << "," << res3[i] << "," << res4[i] << endl;
    }

    cudaFree(rhoGX);
    cudaFree(uGX);
  }

  cudaFree(uGrid); cudaFree(rhoGrid); cudaFree(res1);
  cudaFree(res2); cudaFree(res3); cudaFree(res4);

}
