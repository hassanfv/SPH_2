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





//===== interpolate_4d_hypercube
__device__ float interpolate_4d_hypercube(float nH_p, float T_p, float r_p, float NH_p, float *nH, float *T, float *r, float *NH,
                                          float *Gam, float *Lam, int nxnH0, int nxT0, int nxr0, int nxNH0, int nxnH1, int nxT1, int nxr1, int nxNH1,
                                          int N_nH, int N_T, int N_r, int N_NH)
{
  float dx = (nH_p - nH[nxnH0]) / (nH[nxnH1] - nH[nxnH0]);
  float dy = (T_p - T[nxT0]) / (T[nxT1] - T[nxT0]);
  float dz = (r_p - r[nxr0]) / (r[nxr1] - r[nxr0]);
  float dw = (NH_p - NH[nxNH0]) / (NH[nxNH1] - NH[nxNH0]);

  float pG[16];
  float pL[16];

  int stride_nH = N_T * N_r * N_NH;
  int stride_T = N_r * N_NH;
  int stride_r = N_NH;

  int nxs = 0;
  for (int i = nxnH0; i <= nxnH1; i++)
  {
    for (int j = nxT0; j <= nxT1; j++)
    {
      for (int k = nxr0; k <= nxr1; k++)
      {
        for (int l = nxNH0; l<= nxNH1; l++)
        {
          pG[nxs] = Gam[i * stride_nH + j * stride_T + k * stride_r + l]; // heating!
          pL[nxs] = Lam[i * stride_nH + j * stride_T + k * stride_r + l]; // cooling!
          nxs += 1;
        }
      }
    }
  }

  //-------------> HEATING <---------------
  // Interpolate along the x-axis
  float c00 = pG[0] * (1.0f - dx) + pG[8] * dx;
  float c01 = pG[1] * (1.0f - dx) + pG[9] * dx;
  float c10 = pG[2] * (1.0f - dx) + pG[10] * dx;
  float c11 = pG[3] * (1.0f - dx) + pG[11] * dx;
  float c20 = pG[4] * (1.0f - dx) + pG[12] * dx;
  float c21 = pG[5] * (1.0f - dx) + pG[13] * dx;
  float c30 = pG[6] * (1.0f - dx) + pG[14] * dx;
  float c31 = pG[7] * (1.0f - dx) + pG[15] * dx;

  // Correct interpolation along the y-axis
  float c0 = c00 * (1.0f - dy) + c10 * dy;
  float c1 = c01 * (1.0f - dy) + c11 * dy;
  float c2 = c20 * (1.0f - dy) + c30 * dy;
  float c3 = c21 * (1.0f - dy) + c31 * dy;

  // Interpolate along the z-axis
  float c = c0 * (1.0f - dz) + c2 * dz;
  float d = c1 * (1.0f - dz) + c3 * dz;

  // Finally, interpolate along the w-axis
  float GamH = c * (1.0f - dw) + d * dw;

  //-------------> COOLING <---------------
  // Interpolate along the x-axis
  c00 = pL[0] * (1.0f - dx) + pL[8] * dx;
  c01 = pL[1] * (1.0f - dx) + pL[9] * dx;
  c10 = pL[2] * (1.0f - dx) + pL[10] * dx;
  c11 = pL[3] * (1.0f - dx) + pL[11] * dx;
  c20 = pL[4] * (1.0f - dx) + pL[12] * dx;
  c21 = pL[5] * (1.0f - dx) + pL[13] * dx;
  c30 = pL[6] * (1.0f - dx) + pL[14] * dx;
  c31 = pL[7] * (1.0f - dx) + pL[15] * dx;

  // Correct interpolation along the y-axis
  c0 = c00 * (1.0f - dy) + c10 * dy;
  c1 = c01 * (1.0f - dy) + c11 * dy;
  c2 = c20 * (1.0f - dy) + c30 * dy;
  c3 = c21 * (1.0f - dy) + c31 * dy;

  // Interpolate along the z-axis
  c = c0 * (1.0f - dz) + c2 * dz;
  d = c1 * (1.0f - dz) + c3 * dz;

  // Finally, interpolate along the w-axis
  float LamC = c * (1.0f - dw) + d * dw;

  return powf(10, GamH) - powf(10, LamC); // Use powf for floating-point exponentiation
}



//===== getGamMinusLam
__device__ float getGamMinusLam(float nH_p, float u_p, float r_p, float NH_p, float *nH, float *T, float *r, float *NH, float *uarr, float *muarr,
                                float *Gam, float *Lam, int N_nH, int N_T, int N_r, int N_NH, float gamma, float kB, float mH,
                                float nHLowBound, float nHUpBound, float TLowBound, float TUpBound, float rLowBound, float rUpBound,
                                float NHLowBound, float NHUpBound)
{
  // Adjusting input parameters to be within the bounds
  nH_p = max(nHLowBound + 0.0001f, min(nH_p, nHUpBound - 0.0001f));
  r_p = max(rLowBound + 0.0001f, min(r_p, rUpBound - 0.0001f));
  NH_p = max(NHLowBound + 0.0001f, min(NH_p, NHUpBound - 0.0001f));

  // Finding indexes for interpolation
  int nxnH0 = find_nx0(nH, N_nH, nH_p);
  int nxr0 = find_nx0(r, N_r, r_p);
  int nxNH0 = find_nx0(NH, N_NH, NH_p);

  // Convert u_p to T_p
  u_p = log10(u_p);

  int S1 = N_T * N_r * N_NH;
  int S2 = N_r * N_NH;
  int S3 = N_NH;

  int nxnH1 = nxnH0 + 1;
  int nxr1 = nxr0 + 1;
  int nxNH1 = nxNH0 + 1;

  int ndx = 0;
  int nxu0 = -1;
  for (int j = 0; j < N_T; j++)
  {
    ndx = (nxnH1 * S1) + (j * S2) + (nxr1 * S3) + nxNH1;
    if (nxu0 == -1 && u_p <= uarr[ndx])
    {
      nxu0 = ndx;
      break; // Found the first index where u_p <= uarr[ndx], stop searching
    }
  }

  // Handling the case where u_p is greater than max(uarr)
  if (nxu0 == -1)
    nxu0 = N_nH * N_T * N_r * N_NH - 1;

  float mu_p = muarr[nxu0];
  float T_p = (gamma - 1.0f) * mu_p * mH / kB * powf(10, u_p);
  T_p = log10(T_p);

  // Ensuring T_p is within bounds
  T_p = max(TLowBound + 0.0001f, min(T_p, TUpBound - 0.0001f));

  int nxT0 = find_nx0(T, N_T, T_p);
  int nxT1 = nxT0 + 1;

  // Performing 4D interpolation
  float Gam_minus_Lam = interpolate_4d_hypercube(nH_p, T_p, r_p, NH_p, nH, T, r, NH, Gam, Lam, nxnH0, nxT0, nxr0, nxNH0,
                                                 nxnH1, nxT1, nxr1, nxNH1, N_nH, N_T, N_r, N_NH);

  return Gam_minus_Lam;
}



//===== doCooling
__global__ void doCooling(int *Typ, float *x, float *y, float *z, float *rho_p, float *u_p, float *NHtot,
                          float *nHGrid, float *TGrid, float *rGrid, float *NHGrid, float *uarr, float *muarr, float *Gam, float *Lam, 
                          int N_nH, int N_T, int N_r, int N_NH, float gamma, float kB, float mH,
                          float nHLowBound, float nHUpBound, float TLowBound, float TUpBound, float rLowBound, float rUpBound,
                          float NHLowBound, float NHUpBound, float dt,
                          float unit_time_in_sec, float unit_density_in_cgs, float unit_length_in_kpc, float unit_u_in_cgs, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if ((i < N) && (Typ[i] == 0))
  {
    float XH = 0.7;

    float dt_s = dt * unit_time_in_sec;

    float rho_cgs = rho_p[i] * unit_density_in_cgs;
    float nH_p = XH * rho_cgs / mH;
    nH_p = log10(nH_p);

    float u_ad = u_p[i] * unit_u_in_cgs; // log10 of it will be taken inside the "getGamMinusLam" function!

    float r_p = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    r_p = r_p * unit_length_in_kpc;

    float NH_p = NHtot[i]; // !!!!!! MAKE sure that NHtot are already in log10 form!!!!!!!!!!!!

    float GL = 0.0;
    

    float u_old = u_ad; 

    float u = u_old;
    float u_upper = u;

    printf("XX nH_p, u, r_p, NH_p = %f, %e, %f, %f\n", nH_p, u, r_p, NH_p);

    // log10 of u will be taken inside the "getGamMinusLam" function!
    GL = getGamMinusLam(nH_p, u, r_p, NH_p, nHGrid, TGrid, rGrid, NHGrid, uarr, muarr,
                        Gam, Lam, N_nH, N_T, N_r, N_NH, gamma, kB, mH,
                        nHLowBound, nHUpBound, TLowBound, TUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound);
    float f_u = u - (u_ad + (GL/rho_cgs*dt_s));

    float u_lower = u; // just a place holder!
    float f_u_upper = f_u;

    //-----> (f_u > 0.0) <--------
    if (f_u > 0.0)
    {
      while (f_u_upper > 0.0)
      {
        GL = getGamMinusLam(nH_p, u_upper, r_p, NH_p, nHGrid, TGrid, rGrid, NHGrid, uarr, muarr,
                            Gam, Lam, N_nH, N_T, N_r, N_NH, gamma, kB, mH,
                            nHLowBound, nHUpBound, TLowBound, TUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound);
        f_u_upper = u_upper - (u_ad + (GL/rho_cgs*dt_s));
        u_upper = u_upper / 1.1;
      }
        
      u_lower = u_upper;
      u_upper = u_lower * 1.1;
    }

    //-----> (f_u < 0.0) <--------
    if (f_u < 0.0)
    {    
      while (f_u_upper < 0.0)
      {
        GL = getGamMinusLam(nH_p, u_upper, r_p, NH_p, nHGrid, TGrid, rGrid, NHGrid, uarr, muarr,
                            Gam, Lam, N_nH, N_T, N_r, N_NH, gamma, kB, mH,
                            nHLowBound, nHUpBound, TLowBound, TUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound);
        f_u_upper = u_upper - (u_ad + (GL/rho_cgs*dt_s));
        u_upper = u_upper * 1.1;
      }
      u_upper = u_upper;
      u_lower = u_upper / 1.1;
    }

    int MAXniter = 100;
    int niter = 0;
    float du = u;

    while (du/u > 1e-4  && niter < MAXniter)
    {
      u = 0.5f * (u_lower + u_upper);
      
      GL = getGamMinusLam(nH_p, u, r_p, NH_p, nHGrid, TGrid, rGrid, NHGrid, uarr, muarr,
                          Gam, Lam, N_nH, N_T, N_r, N_NH, gamma, kB, mH,
                          nHLowBound, nHUpBound, TLowBound, TUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound);
      f_u = u - (u_ad + (GL/rho_cgs*dt_s));
      
      if (f_u > 0.0)
        u_upper = u;
      else
        u_lower = u;
      
      du = abs(u_upper - u_lower);
      
      niter++;
      
      if (niter == 100)
        printf("MAXniter in doCooling function REACHED !!!!!!!!!!\n");
    }
    
    u_p[i] = u / unit_u_in_cgs;
    
  }
}






int main()
{

  float kB = 1.3807e-16;
  float mH = 1.673534e-24;
  float gamma = 5.0f/3.0f;

  ifstream binFile("HCoolMu.bin", ios::binary);
  if (!binFile.is_open()) 
  {
    cerr << "Failed to open HCoolMu.bin file" << endl;
    return 1;
  }

  // Scalar values
  int N_nH, N_T, N_r, N_NH;
  float nHLowBound, nHUpBound, TLowBound, TUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound;

  // Read scalar values
  binFile.read(reinterpret_cast<char*>(&N_nH), sizeof(N_nH));
  binFile.read(reinterpret_cast<char*>(&N_T), sizeof(N_T));
  binFile.read(reinterpret_cast<char*>(&N_r), sizeof(N_r));
  binFile.read(reinterpret_cast<char*>(&N_NH), sizeof(N_NH));
  binFile.read(reinterpret_cast<char*>(&nHLowBound), sizeof(nHLowBound));
  binFile.read(reinterpret_cast<char*>(&nHUpBound), sizeof(nHUpBound));
  binFile.read(reinterpret_cast<char*>(&TLowBound), sizeof(TLowBound));
  binFile.read(reinterpret_cast<char*>(&TUpBound), sizeof(TUpBound));
  binFile.read(reinterpret_cast<char*>(&rLowBound), sizeof(rLowBound));
  binFile.read(reinterpret_cast<char*>(&rUpBound), sizeof(rUpBound));
  binFile.read(reinterpret_cast<char*>(&NHLowBound), sizeof(NHLowBound));
  binFile.read(reinterpret_cast<char*>(&NHUpBound), sizeof(NHUpBound));

  // Allocate memory for arrays
  float *nHGrid = new float[N_nH], *TGrid = new float[N_T], *rGrid = new float[N_r], *NHGrid = new float[N_NH];
  float *Heat = new float[N_nH * N_T * N_r * N_NH], *Cool = new float[N_nH * N_T * N_r * N_NH];
  float *muarr = new float[N_nH * N_T * N_r * N_NH], *uarr = new float[N_nH * N_T * N_r * N_NH];

  // Read grids
  binFile.read(reinterpret_cast<char*>(nHGrid), N_nH * sizeof(float));
  binFile.read(reinterpret_cast<char*>(TGrid), N_T * sizeof(float));
  binFile.read(reinterpret_cast<char*>(rGrid), N_r * sizeof(float));
  binFile.read(reinterpret_cast<char*>(NHGrid), N_NH * sizeof(float));

  // Read data arrays
  binFile.read(reinterpret_cast<char*>(Heat), N_nH * N_T * N_r * N_NH * sizeof(float));
  binFile.read(reinterpret_cast<char*>(Cool), N_nH * N_T * N_r * N_NH * sizeof(float));
  binFile.read(reinterpret_cast<char*>(muarr), N_nH * N_T * N_r * N_NH * sizeof(float));
  binFile.read(reinterpret_cast<char*>(uarr), N_nH * N_T * N_r * N_NH * sizeof(float));

  binFile.close();
  
  
  //--- declaring for GPU
  float *d_nHGrid, *d_TGrid, *d_rGrid, *d_NHGrid;
  float *d_Heat, *d_Cool, *d_muarr, *d_uarr;
  
  cudaMalloc(&d_nHGrid, N_nH * sizeof(float));
  cudaMalloc(&d_TGrid, N_T * sizeof(float));
  cudaMalloc(&d_rGrid, N_r * sizeof(float));
  cudaMalloc(&d_NHGrid, N_NH * sizeof(float));
  
  int N_tot = N_nH * N_T * N_r * N_NH;
  
  cudaMalloc(&d_Heat, N_tot * sizeof(float));
  cudaMalloc(&d_Cool, N_tot * sizeof(float));
  cudaMalloc(&d_muarr, N_tot * sizeof(float));
  cudaMalloc(&d_uarr, N_tot * sizeof(float));
  
  //--- Copying from Host to Device
  cudaMemcpy(d_nHGrid, nHGrid, N_nH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_TGrid, TGrid, N_T * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rGrid, rGrid, N_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_NHGrid, NHGrid, N_NH * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_Heat, Heat, N_tot * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Cool, Cool, N_tot * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_muarr, muarr, N_tot * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uarr, uarr, N_tot * sizeof(float), cudaMemcpyHostToDevice);
  //------------- End of reading and preparing HCool table file -----------
  
  

  //--- create rho_p, u_p, and NHtot array on the host
  
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
  float *NHtot = new float[Npart];
  
  int k = 0;
  for (float i = 0.0; i < 4.0; i++)
  {
    rho[k] = pow(10.0, i);
    
    x[k] = 0.0;
    y[k] = 0.0;
    z[k] = 0.02 + i * 0.1;
    
    u_p[k] = kB * pow(10, 4.0+0.25*i) / (gamma - 1.0) / 0.55 / mH / unit_u_in_cgs;
    
    NHtot[k] = 20.0 + i * 0.5; // Note that NHtot must be in log10!!!
    
    k++;
  }
  
  float dt = 5.0; // in code unit!
  
  int *d_Typ;
  float *d_x, *d_y, *d_z, *d_u_p, *d_NHtot, *d_rho;
  
  cudaMalloc(&d_Typ, Npart * sizeof(int));
  cudaMalloc(&d_rho, Npart * sizeof(float));
  
  cudaMalloc(&d_x, Npart * sizeof(float));
  cudaMalloc(&d_y, Npart * sizeof(float));
  cudaMalloc(&d_z, Npart * sizeof(float));
  
  cudaMalloc(&d_u_p, Npart * sizeof(float));
  cudaMalloc(&d_NHtot, Npart * sizeof(float));
  
  cudaMemcpy(d_Typ, Typ, Npart * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rho, rho, Npart * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_x, x, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, Npart * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_u_p, u_p, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_NHtot, NHtot, Npart * sizeof(float), cudaMemcpyHostToDevice);
  
  float muT = 0.6;
  
  float XH = 0.7;
  
  for (int i = 0; i < Npart; i++)
  {
    cout << "u Before = " << u_p[i] << endl;
    cout << "T Before = " << u_p[i]*(gamma - 1.0) * muT * mH / kB * unit_u_in_cgs << endl;
    cout << "nH = " << rho[i] * XH / mH * unit_density_in_cgs << endl; 
    float rr = sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);
    cout << "r_p = " << rr << endl;
    cout << "NHtot = " << NHtot[i] << endl;
    cout << "-----------------" << endl << endl;
    
  }
  cout << endl;
  
  doCooling<<<1, 4>>>(d_Typ, d_x, d_y, d_z, d_rho, d_u_p, d_NHtot, d_nHGrid, d_TGrid, d_rGrid, d_NHGrid, d_uarr, d_muarr, d_Heat, d_Cool,
                      N_nH, N_T, N_r, N_NH, gamma, kB, mH, nHLowBound, nHUpBound, TLowBound, TUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound,
                      dt, unit_time_in_sec, unit_density_in_cgs, unit_length_in_kpc, unit_u_in_cgs, Npart);
  
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    cerr << "CUDA error: " << cudaGetErrorString(error) << endl;
  }
  
  cudaDeviceSynchronize();
  
  
  cudaMemcpy(u_p, d_u_p, Npart * sizeof(float), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < Npart; i++)
  {
    cout << "u After = " << u_p[i] << endl;
    cout << "T After = " << u_p[i]*(gamma - 1.0) * muT * mH / kB * unit_u_in_cgs << endl;
  }
  

}




