%%writefile test.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib> // This is ONLY used for the "exit(0)" function !!

#include <curand.h>
#include <curand_kernel.h>

using namespace std;



//===== findClosestIndex
__device__ int find_nx0(float *A, int N, float x)
{
  int nxClosest = 0; // Start with the first element as the closest
  float smallestDiff = fabsf(A[0] - x); // Initialize the smallest difference using fabsf for float absolute value in CUDA

  for (int i = 1; i < N; ++i)
  {
    float currentDiff = fabsf(A[i] - x); // Compute the current difference using fabsf
    if (currentDiff < smallestDiff) { // If the current difference is smaller, update the closest element
      smallestDiff = currentDiff;
      nxClosest = i;
    }
  }

  int nx0 = 0; // just a place holder.

  if (x >= A[nxClosest])
    nx0 = nxClosest;
  else
    nx0 = nxClosest - 1;

  if (nx0 == N - 1) // To handle cases in which the index is at the last element
    nx0 = nxClosest - 1;

  if (nx0 == -1) // Handle case where computed index is below 0
    nx0 = 0;

  return nx0; // Return nx0. nx1 will be computed elsewhere!
}



//===== interpolate_5d_hypercube
__device__ float interpolate_5d_hypercube(float nH_p, float u_p, float r_p, float NH_p, float dt_yrs, float *uarr, float *nHGrid, float *uGrid,
                                          float *rGrid, float *NHGrid, float *dtGrid,
                                          int nxnH0, int nxu0, int nxr0, int nxNH0, int nxtime0,
                                          int nxnH1, int nxu1, int nxr1, int nxNH1, int nxtime1,
                                          int N_nH, int N_T, int N_r, int N_NH, float N_t)
{
  float dx = (u_p - uGrid[nxu0]) / (uGrid[nxu1] - uGrid[nxu0]);  //// !!!!!!!!!!! CHECK if we use log(u), what happens in accuracy!!!!!!!!!!!!!!!!!!!!!!!!!!
  float dy = (nH_p - nHGrid[nxnH0]) / (nHGrid[nxnH1] - nHGrid[nxnH0]); 
  float dz = (r_p - rGrid[nxr0]) / (rGrid[nxr1] - rGrid[nxr0]);
  float dw = (NH_p - NHGrid[nxNH0]) / (NHGrid[nxNH1] - NHGrid[nxNH0]);
  float dv = (dt_yrs - dtGrid[nxtime0]) / (dtGrid[nxtime1] - dtGrid[nxtime0]);

  float pData[32];

  int S1 = N_nH * N_r * N_NH * N_t;
  int S2 = N_r * N_NH * N_t;
  int S3 = N_NH * N_t;
  int S4 = N_t;

  int nxs = 0;
  for (int i = nxu0; i <= nxu1; i++)
  {
    for (int j = nxnH0; j <= nxnH1; j++)
    {
      for (int k = nxr0; k <= nxr1; k++)
      {
        for (int l = nxNH0; l <= nxNH1; l++)
        {
          for (int m = nxtime0; m <= nxtime1; m++)
          {
            pData[nxs] = uarr[i * S1 + j * S2 + k * S3 + l * S4 + m];
            nxs += 1;
          }
        }
      }
    }
  }

  //-------------> 5-D interpolation  <---------------
  // Interpolate along the x-axis
  float c00 = pData[0] * (1.0f - dx) + pData[16] * dx;
  float c01 = pData[1] * (1.0f - dx) + pData[17] * dx;
  float c02 = pData[2] * (1.0f - dx) + pData[18] * dx;
  float c03 = pData[3] * (1.0f - dx) + pData[19] * dx;
  float c04 = pData[4] * (1.0f - dx) + pData[20] * dx;
  float c05 = pData[5] * (1.0f - dx) + pData[21] * dx;
  float c06 = pData[6] * (1.0f - dx) + pData[22] * dx;
  float c07 = pData[7] * (1.0f - dx) + pData[23] * dx;
  float c08 = pData[8] * (1.0f - dx) + pData[24] * dx;
  float c09 = pData[9] * (1.0f - dx) + pData[25] * dx;
  float c10 = pData[10] * (1.0f - dx) + pData[26] * dx;
  float c11 = pData[11] * (1.0f - dx) + pData[27] * dx;
  float c12 = pData[12] * (1.0f - dx) + pData[28] * dx;
  float c13 = pData[13] * (1.0f - dx) + pData[29] * dx;
  float c14 = pData[14] * (1.0f - dx) + pData[30] * dx;
  float c15 = pData[15] * (1.0f - dx) + pData[31] * dx;

  // Interpolate along the y-axis
  float c0 = c00 * (1.0f - dy) + c04 * dy;
  float c1 = c01 * (1.0f - dy) + c05 * dy;
  float c2 = c02 * (1.0f - dy) + c06 * dy;
  float c3 = c03 * (1.0f - dy) + c07 * dy;
  float c4 = c08 * (1.0f - dy) + c12 * dy;
  float c5 = c09 * (1.0f - dy) + c13 * dy;
  float c6 = c10 * (1.0f - dy) + c14 * dy;
  float c7 = c11 * (1.0f - dy) + c15 * dy;

  // Interpolate along the z-axis
  float c = c0 * (1.0f - dz) + c4 * dz;
  float d = c1 * (1.0f - dz) + c5 * dz;
  float e = c2 * (1.0f - dz) + c6 * dz;
  float f = c3 * (1.0f - dz) + c7 * dz;

  // Interpolate along the w-axis
  float uu = c * (1.0f - dw) + e * dw;
  float vv = d * (1.0f - dw) + f * dw;

  // Finally, interpolate along the e-axis
  float uFinal = uu * (1.0f - dv) + vv * dv;

  return uFinal; // Use powf(10, LamC) for floating-point exponentiation
}



//===== doCooling
__global__ void doCooling(int *Typ, float *x, float *y, float *z, float *rho, float *u, float *NH,
                          float *nHGrid, float *rGrid, float *NHGrid, float *dtGrid, float *uarr, 
                          int N_nH, int N_T, int N_r, int N_NH, int N_t, float gamma, float kB, float mH,
                          float nHLowBound, float nHUpBound, float rLowBound, float rUpBound,
                          float NHLowBound, float NHUpBound, float timeLowBound, float timeUpBound, float dt,
                          float unit_time_in_sec, float unit_density_in_cgs, float unit_length_in_kpc, float unit_u_in_cgs, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if ((i < N) && (Typ[i] == 0))
  {
    float XH = 0.7;
    
    float u_p = u[i] * unit_u_in_cgs;

    float nH_p = rho[i] * unit_density_in_cgs * XH / mH;
    nH_p = log10(nH_p);
    
    float NH_p = NH[i];
    
    float r_p = sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);

    // Adjusting input parameters to be within the bounds
    nH_p = max(nHLowBound + 0.0001f, min(nH_p, nHUpBound - 0.0001f));
    r_p = max(rLowBound + 0.0001f, min(r_p, rUpBound - 0.0001f));
    NH_p = max(NHLowBound + 0.0001f, min(NH_p, NHUpBound - 0.0001f));

    // Finding indexes for interpolation
    int nxnH0 = find_nx0(nHGrid, N_nH, nH_p);
    int nxr0 = find_nx0(rGrid, N_r, r_p);
    int nxNH0 = find_nx0(NHGrid, N_NH, NH_p);
    
    int nxnH1 = nxnH0 + 1;
    int nxr1 = nxr0 + 1;
    int nxNH1 = nxNH0 + 1;

    // Finding index for u
    int S1 = N_nH * N_r * N_NH * N_t;
    int S2 = N_r * N_NH * N_t;
    int S3 = N_NH * N_t;
    int S4 = N_t;
    
    int nxu0 = -1;
    
    float uGrid[91]; //!!!!!!!!!!!!!! CHECK IF THIS IS CORRECT TO DO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    for (int j = 0; j < N_T; j++)
    {
      int Lnx = j * S1 + nxnH0 * S2 + nxr0 * S3 + nxNH0 * S4 + 0; // interested in values at t = 0!
      uGrid[j] = uarr[Lnx]; 
      if (nxu0 == -1 && u_p <= uarr[Lnx])
      {
        nxu0 = j - 1;
      }
    }
    
    if (nxu0 == -1)
      nxu0 = N_T - 2;

    float dt_yrs = dt * unit_time_in_sec / 3600.0 / 24.0 / 365.25;
    dt_yrs = max(timeLowBound + 0.0001f, min(dt_yrs, timeUpBound - 0.0001f)); 
    int nxtime0 = find_nx0(dtGrid, N_t, dt_yrs);

    
    int nxu1 = nxu0 + 1;
    int nxtime1 = nxtime0 + 1;
    
    if (i == 3)
    {
      //printf("nxnH0, nxr0, nxNH0, nxu0, nxtime0 = %d, %d, %d, %d, %d\n", nxnH0, nxr0, nxNH0, nxu0, nxtime0);
      //printf("nxnH1, nxr1, nxNH1, nxu1, nxtime1 = %d, %d, %d, %d, %d\n\n", nxnH1, nxr1, nxNH1, nxu1, nxtime1);
      
      //printf("nH-1, r-1, NH-1, u-1, time-1 = %f, %f, %f, %e, %f\n\n", nHGrid[nxnH0-1], rGrid[nxr0-1], NHGrid[nxNH0-1], uGrid[nxu0-1], dtGrid[nxtime0-1]);
      printf("nH0, r0, NH0, u0, time0 = %f, %f, %f, %e, %f\n\n", nHGrid[nxnH0], rGrid[nxr0], NHGrid[nxNH0], uGrid[nxu0], dtGrid[nxtime0]);
      printf("nH_p, r_p, NH_p, u_p, dt_yrs = %f, %f, %f, %e, %f\n\n", nH_p, r_p, NH_p, u_p, dt_yrs);
      printf("nH1, r1, NH1, u1, time1 = %f, %f, %f, %e, %f\n\n", nHGrid[nxnH1], rGrid[nxr1], NHGrid[nxNH1], uGrid[nxu1], dtGrid[nxtime1]);
    }

    // Performing 5D interpolation
    float u_after_dt = interpolate_5d_hypercube(nH_p, u_p, r_p, NH_p, dt_yrs, uarr, nHGrid, uGrid, rGrid, NHGrid, dtGrid, nxnH0, nxu0, nxr0, nxNH0, nxtime0,
                                                nxnH1, nxu1, nxr1, nxNH1, nxtime1, N_nH, N_T, N_r, N_NH, N_t);
    u[i] = u_after_dt / unit_u_in_cgs;
  }
}



int main()
{

  float kB = 1.3807e-16;
  float mH = 1.673534e-24;
  float gamma = 5.0f/3.0f;

  std::ifstream file("HCoolChimes.bin", std::ios::binary);
  if (!file.is_open())
  {
    std::cerr << "Failed to open HCoolChimes.bin file" << std::endl;
    return 1; // Exit if file not opened successfully
  }
    
  // Reading integers
  int N_T, N_nH, N_r, N_NH, N_t;
  file.read(reinterpret_cast<char*>(&N_T), sizeof(N_T));
  file.read(reinterpret_cast<char*>(&N_nH), sizeof(N_nH));
  file.read(reinterpret_cast<char*>(&N_r), sizeof(N_r));
  file.read(reinterpret_cast<char*>(&N_NH), sizeof(N_NH));
  file.read(reinterpret_cast<char*>(&N_t), sizeof(N_t));

  // Reading floats
  float nHLowBound, nHUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound, timeLowBound, timeUpBound;
  file.read(reinterpret_cast<char*>(&nHLowBound), sizeof(nHLowBound));
  file.read(reinterpret_cast<char*>(&nHUpBound), sizeof(nHUpBound));
  file.read(reinterpret_cast<char*>(&rLowBound), sizeof(rLowBound));
  file.read(reinterpret_cast<char*>(&rUpBound), sizeof(rUpBound));
  file.read(reinterpret_cast<char*>(&NHLowBound), sizeof(NHLowBound));
  file.read(reinterpret_cast<char*>(&NHUpBound), sizeof(NHUpBound));
  file.read(reinterpret_cast<char*>(&timeLowBound), sizeof(timeLowBound));
  file.read(reinterpret_cast<char*>(&timeUpBound), sizeof(timeUpBound));

  // Reading arrays
  float* uarr = new float[N_T * N_nH * N_r * N_NH * N_t];
  file.read(reinterpret_cast<char*>(uarr), (N_T * N_nH * N_r * N_NH * N_t) * sizeof(float));

  float* nHGrid = new float[N_nH];
  file.read(reinterpret_cast<char*>(nHGrid), N_nH * sizeof(float));

  float* rGrid = new float[N_r];
  file.read(reinterpret_cast<char*>(rGrid), N_r * sizeof(float));

  float* NHGrid = new float[N_NH];
  file.read(reinterpret_cast<char*>(NHGrid), N_NH * sizeof(float));
  
  float* dtGrid = new float[N_t];
  file.read(reinterpret_cast<char*>(dtGrid), N_t * sizeof(float));
  file.close();
  
  //--- declaring for GPU
  float *d_nHGrid, *d_rGrid, *d_NHGrid, *d_dtGrid, *d_uarr;
  
  cudaMalloc(&d_nHGrid, N_nH * sizeof(float));
  cudaMalloc(&d_rGrid, N_r * sizeof(float));
  cudaMalloc(&d_NHGrid, N_NH * sizeof(float));
  cudaMalloc(&d_dtGrid, N_t * sizeof(float));
  
  int N_tot = N_T * N_nH * N_r * N_NH * N_t;

  cudaMalloc(&d_uarr, N_tot * sizeof(float));
  
  //--- Copying from Host to Device
  cudaMemcpy(d_nHGrid, nHGrid, N_nH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rGrid, rGrid, N_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_NHGrid, NHGrid, N_NH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dtGrid, dtGrid, N_t * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_uarr, uarr, N_tot * sizeof(float), cudaMemcpyHostToDevice);
  //------------- End of reading and preparing HCool table file -----------




  //------> EXAMPLE CASE: create rho_p, u_p, and NH array on the host <---------
  float unit_time_in_sec = 365.25 * 24.0 * 3600.0;
  float unit_density_in_cgs = 1.673534e-24; // corresponds to nH = 1 cm^-3
  float unit_length_in_kpc = 1.0;
  float unit_u_in_cgs = 1e13;
  
  int Npart = 4;
  
  int *Typ = new int[Npart];
  for (int i = 0; i < Npart; i++)
    Typ[i] = 0;  // 0 represents gas particles!
  
  float *rho = new float[Npart];
  float *x = new float[Npart];
  float *y = new float[Npart];
  float *z = new float[Npart];
  float *u_p = new float[Npart];
  float *NH = new float[Npart];
  
  int k = 0;
  for (float i = 0.0; i < 4.0; i++)
  {
    rho[k] = pow(10.0, i);
    
    x[k] = 0.0;
    y[k] = 0.0;
    z[k] = 0.02 + i * 0.1;
    
    u_p[k] = kB * pow(10, 4.0+0.25*i) / (gamma - 1.0) / 0.55 / mH / unit_u_in_cgs;
    
    NH[k] = 20.0 + i * 0.5; // Note that NH must be in log10!!!
    
    k++;
  }
  
  z[3] = 0.25; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  float dt = 5.7; // in code unit!
  
  int *d_Typ;
  float *d_x, *d_y, *d_z, *d_u_p, *d_NH, *d_rho;
  
  cudaMalloc(&d_Typ, Npart * sizeof(int));
  cudaMalloc(&d_rho, Npart * sizeof(float));
  
  cudaMalloc(&d_x, Npart * sizeof(float));
  cudaMalloc(&d_y, Npart * sizeof(float));
  cudaMalloc(&d_z, Npart * sizeof(float));
  
  cudaMalloc(&d_u_p, Npart * sizeof(float));
  cudaMalloc(&d_NH, Npart * sizeof(float));
  
  cudaMemcpy(d_Typ, Typ, Npart * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rho, rho, Npart * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_x, x, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, Npart * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_u_p, u_p, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_NH, NH, Npart * sizeof(float), cudaMemcpyHostToDevice);


  doCooling<<<1, 4>>>(d_Typ, d_x, d_y, d_z, d_rho, d_u_p, d_NH,
                      d_nHGrid, d_rGrid, d_NHGrid, d_dtGrid, d_uarr, 
                      N_nH, N_T, N_r, N_NH, N_t, gamma, kB, mH,
                      nHLowBound, nHUpBound, rLowBound, rUpBound,
                      NHLowBound, NHUpBound, timeLowBound, timeUpBound, dt,
                      unit_time_in_sec, unit_density_in_cgs, unit_length_in_kpc, unit_u_in_cgs, Npart);
                      
  cudaDeviceSynchronize();
  
  
  for (int i = 0; i < Npart; i++)
  {
    cout << "u Before = " << u_p[i] << endl;
  }
  cout << endl;
  
  cudaMemcpy(u_p, d_u_p, Npart * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < Npart; i++)
  {
    cout << "u After = " << u_p[i] << endl;
  }
  


}


