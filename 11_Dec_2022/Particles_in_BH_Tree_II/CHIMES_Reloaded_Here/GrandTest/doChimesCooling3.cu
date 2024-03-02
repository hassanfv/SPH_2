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
__device__ float interpolate_5d_hypercube(float nH_p, float T_p, float r_p, float NH_p, float dt_yrs, float *Tarr, float *nHGrid, float *TGrid,
                                          float *rGrid, float *NHGrid, float *dtGrid,
                                          int nxnH0, int nxT0, int nxr0, int nxNH0, int nxtime0,
                                          int nxnH1, int nxT1, int nxr1, int nxNH1, int nxtime1,
                                          int N_nH, int N_T, int N_r, int N_NH, float N_t)
{
  float dx = (T_p - TGrid[nxT0]) / (TGrid[nxT1] - TGrid[nxT0]);  //!! CHECK if we use log(u), what happens in accuracy!!!!!!
  float dy = (nH_p - nHGrid[nxnH0]) / (nHGrid[nxnH1] - nHGrid[nxnH0]);
  float dz = (r_p - rGrid[nxr0]) / (rGrid[nxr1] - rGrid[nxr0]);
  float dw = (NH_p - NHGrid[nxNH0]) / (NHGrid[nxNH1] - NHGrid[nxNH0]);
  float dv = (dt_yrs - dtGrid[nxtime0]) / (dtGrid[nxtime1] - dtGrid[nxtime0]);

  printf("dx(u), dy(nH), dz(r), dw(NH), dv(dt) = %f, %f, %f, %f, %f\n\n", dx, dy, dz, dw, dv);

  float pData[32];

  int S1 = N_nH * N_r * N_NH * N_t;
  int S2 = N_r * N_NH * N_t;
  int S3 = N_NH * N_t;
  int S4 = N_t;

  int nxs = 0;
  for (int i = nxT0; i <= nxT1; i++)
  {
    for (int j = nxnH0; j <= nxnH1; j++)
    {
      for (int k = nxr0; k <= nxr1; k++)
      {
        for (int l = nxNH0; l <= nxNH1; l++)
        {
          for (int m = nxtime0; m <= nxtime1; m++)
          {
            pData[nxs] = Tarr[i * S1 + j * S2 + k * S3 + l * S4 + m];
            nxs += 1;
            printf("PG = %e\n", Tarr[i * S1 + j * S2 + k * S3 + l * S4 + m]);
          }
        }
      }
    }
  }

  printf("\n\n\n");

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
  /*
  float c0 = c00 * (1.0f - dy) + c04 * dy;
  float c1 = c01 * (1.0f - dy) + c05 * dy;
  float c2 = c02 * (1.0f - dy) + c06 * dy;
  float c3 = c03 * (1.0f - dy) + c07 * dy;
  float c4 = c08 * (1.0f - dy) + c12 * dy;
  float c5 = c09 * (1.0f - dy) + c13 * dy;
  float c6 = c10 * (1.0f - dy) + c14 * dy;
  float c7 = c11 * (1.0f - dy) + c15 * dy;
  */
  
  // Interpolate along the y-axis
  float c0 = c00 * (1.0f - dy) + c08 * dy;
  float c1 = c01 * (1.0f - dy) + c09 * dy;
  float c2 = c02 * (1.0f - dy) + c10 * dy;
  float c3 = c03 * (1.0f - dy) + c11 * dy;
  float c4 = c04 * (1.0f - dy) + c12 * dy;
  float c5 = c05 * (1.0f - dy) + c13 * dy;
  float c6 = c06 * (1.0f - dy) + c14 * dy;
  float c7 = c07 * (1.0f - dy) + c15 * dy;

  // Interpolate along the z-axis
  float c = c0 * (1.0f - dz) + c4 * dz;
  float d = c1 * (1.0f - dz) + c5 * dz;
  float e = c2 * (1.0f - dz) + c6 * dz;
  float f = c3 * (1.0f - dz) + c7 * dz;

  // Interpolate along the w-axis
  float uu = c * (1.0f - dw) + e * dw;
  float vv = d * (1.0f - dw) + f * dw;

  // Finally, interpolate along the e-axis
  float TFinal = uu * (1.0f - dv) + vv * dv;

  return TFinal; // Use powf(10, LamC) for floating-point exponentiation
}



//===== doCooling
__global__ void doCooling(int *Typ, float *x, float *y, float *z, float *rho, float *T, float *NH,
                          float *nHGrid, float *rGrid, float *NHGrid, float *dtGrid, float *TGrid, float *Tarr,
                          int N_nH, int N_T, int N_r, int N_NH, int N_t, float gamma, float kB, float mH,
                          float nHLowBound, float nHUpBound, float rLowBound, float rUpBound, float TLowBound, float TUpBound,
                          float NHLowBound, float NHUpBound, float timeLowBound, float timeUpBound, float dt,
                          float unit_time_in_sec, float unit_density_in_cgs, float unit_length_in_kpc, float unit_u_in_cgs, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    float XH = 0.7;

    //float u_p = u[i] * unit_u_in_cgs;
    float T_p = T[i];

    float nH_p = rho[i] * unit_density_in_cgs * XH / mH;
    nH_p = log10(nH_p);

    float NH_p = NH[i];

    float r_p = sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]);

    // Adjusting input parameters to be within the bounds
    nH_p = max(nHLowBound + 0.0001f, min(nH_p, nHUpBound - 0.0001f));
    r_p = max(rLowBound + 0.0001f, min(r_p, rUpBound - 0.0001f));
    NH_p = max(NHLowBound + 0.0001f, min(NH_p, NHUpBound - 0.0001f));
    T_p = max(TLowBound + 0.0001f, min(T_p, TUpBound - 0.0001f));

    // Finding indexes for interpolation
    int nxnH0 = find_nx0(nHGrid, N_nH, nH_p);
    int nxr0 = find_nx0(rGrid, N_r, r_p);
    int nxNH0 = find_nx0(NHGrid, N_NH, NH_p);

    int nxnH1 = nxnH0 + 1;
    int nxr1 = nxr0 + 1;
    int nxNH1 = nxNH0 + 1;

    int nxT0 = find_nx0(TGrid, N_T, T_p);

    float dt_yrs = dt * unit_time_in_sec / 3600.0 / 24.0 / 365.25;
    dt_yrs = max(timeLowBound + 0.0001f, min(dt_yrs, timeUpBound - 0.0001f));
    int nxtime0 = find_nx0(dtGrid, N_t, dt_yrs);


    int nxT1 = nxT0 + 1;
    int nxtime1 = nxtime0 + 1;


    if (i == 0)
    {
      //printf("nxnH0, nxr0, nxNH0, nxT0, nxtime0 = %d, %d, %d, %d, %d\n", nxnH0, nxr0, nxNH0, nxT0, nxtime0);
      //printf("nxnH1, nxr1, nxNH1, nxT1, nxtime1 = %d, %d, %d, %d, %d\n\n", nxnH1, nxr1, nxNH1, nxT1, nxtime1);

      //printf("nH-1, r-1, NH-1, T-1, time-1 = %f, %f, %f, %e, %f\n\n", nHGrid[nxnH0-1], rGrid[nxr0-1], NHGrid[nxNH0-1], TGrid[nxT0-1], dtGrid[nxtime0-1]);
      printf("nH0, r0, NH0, T0, time0 = %f, %f, %f, %e, %f\n\n", nHGrid[nxnH0], rGrid[nxr0], NHGrid[nxNH0], TGrid[nxT0], dtGrid[nxtime0]);
      printf("nH_p, r_p, NH_p, T_p, dt_yrs = %f, %f, %f, %e, %f\n\n", nH_p, r_p, NH_p, T_p, dt_yrs);
      printf("nH1, r1, NH1, T1, time1 = %f, %f, %f, %e, %f\n\n", nHGrid[nxnH1], rGrid[nxr1], NHGrid[nxNH1], TGrid[nxT1], dtGrid[nxtime1]);
    }


    // Performing 5D interpolation
    float T_after_dt = interpolate_5d_hypercube(nH_p, T_p, r_p, NH_p, dt_yrs, Tarr, nHGrid, TGrid, rGrid, NHGrid, dtGrid, nxnH0, nxT0, nxr0, nxNH0, nxtime0,
                                                nxnH1, nxT1, nxr1, nxNH1, nxtime1, N_nH, N_T, N_r, N_NH, N_t);
    T[i] = T_after_dt;// / unit_u_in_cgs;
  }
}



int main()
{

  float kB = 1.3807e-16;
  float mH = 1.673534e-24;
  float gamma = 5.0f/3.0f;

  std::ifstream file("HCoolChimesT.bin", std::ios::binary);
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
  float nHLowBound, nHUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound, timeLowBound, timeUpBound, TLowBound, TUpBound;
  file.read(reinterpret_cast<char*>(&nHLowBound), sizeof(nHLowBound));
  file.read(reinterpret_cast<char*>(&nHUpBound), sizeof(nHUpBound));
  file.read(reinterpret_cast<char*>(&rLowBound), sizeof(rLowBound));
  file.read(reinterpret_cast<char*>(&rUpBound), sizeof(rUpBound));
  file.read(reinterpret_cast<char*>(&NHLowBound), sizeof(NHLowBound));
  file.read(reinterpret_cast<char*>(&NHUpBound), sizeof(NHUpBound));
  file.read(reinterpret_cast<char*>(&timeLowBound), sizeof(timeLowBound));
  file.read(reinterpret_cast<char*>(&timeUpBound), sizeof(timeUpBound));
  
  file.read(reinterpret_cast<char*>(&TLowBound), sizeof(TLowBound));
  file.read(reinterpret_cast<char*>(&TUpBound), sizeof(TUpBound));

  // Reading arrays
  float* Tarr = new float[N_T * N_nH * N_r * N_NH * N_t];
  file.read(reinterpret_cast<char*>(Tarr), (N_T * N_nH * N_r * N_NH * N_t) * sizeof(float));

  float* nHGrid = new float[N_nH];
  file.read(reinterpret_cast<char*>(nHGrid), N_nH * sizeof(float));

  float* rGrid = new float[N_r];
  file.read(reinterpret_cast<char*>(rGrid), N_r * sizeof(float));

  float* NHGrid = new float[N_NH];
  file.read(reinterpret_cast<char*>(NHGrid), N_NH * sizeof(float));

  float* dtGrid = new float[N_t];
  file.read(reinterpret_cast<char*>(dtGrid), N_t * sizeof(float));
  
  float* TGrid = new float[N_T];
  file.read(reinterpret_cast<char*>(TGrid), N_t * sizeof(float));
  file.close();

  //--- declaring for GPU
  float *d_nHGrid, *d_rGrid, *d_NHGrid, *d_dtGrid, *d_TGrid, *d_Tarr;

  cudaMalloc(&d_nHGrid, N_nH * sizeof(float));
  cudaMalloc(&d_rGrid, N_r * sizeof(float));
  cudaMalloc(&d_NHGrid, N_NH * sizeof(float));
  cudaMalloc(&d_dtGrid, N_t * sizeof(float));
  cudaMalloc(&d_TGrid, N_T * sizeof(float));

  int N_tot = N_T * N_nH * N_r * N_NH * N_t;

  cudaMalloc(&d_Tarr, N_tot * sizeof(float));

  //--- Copying from Host to Device
  cudaMemcpy(d_nHGrid, nHGrid, N_nH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rGrid, rGrid, N_r * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_NHGrid, NHGrid, N_NH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dtGrid, dtGrid, N_t * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_TGrid, TGrid, N_T * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_Tarr, Tarr, N_tot * sizeof(float), cudaMemcpyHostToDevice);
  //------------- End of reading and preparing HCool table file -----------




  //------> EXAMPLE CASE: create rho_p, u_p, and NH array on the host <---------
  float unit_time_in_sec = 365.25 * 24.0 * 3600.0;
  float unit_density_in_cgs = 1.673534e-24; // corresponds to nH = 1 cm^-3
  float unit_length_in_kpc = 1.0;
  float unit_u_in_cgs = 1e11;

  int Npart = 1;

  int *Typ = new int[Npart];
  for (int i = 0; i < Npart; i++)
    Typ[i] = 0;  // 0 represents gas particles!

  float *rho = new float[Npart];
  float *x = new float[Npart];
  float *y = new float[Npart];
  float *z = new float[Npart];
  //float *u_p = new float[Npart];
  float *T_p = new float[Npart];
  float *NH = new float[Npart];

  float *rr_p = new float[Npart];

  int k = 0;
  for (int i = 0; i < Npart; i++)
  {
    rho[k] = 10.0 / 0.7; //pow(10.0, i);

    x[k] = 0.0;
    y[k] = 0.0;
    z[k] = 0.08;// + i * 0.003;

    rr_p[k] = sqrt(x[k]*x[k] + y[k]*y[k] + z[k]*z[k]);

    //u_p[k] = 1000.0;//kB * pow(10, 0.0) / (gamma - 1.0) / 0.55 / mH / unit_u_in_cgs;
    T_p[k] = log10(10000.0);//kB * pow(10, 0.0) / (gamma - 1.0) / 0.55 / mH / unit_u_in_cgs;

    NH[k] = 20.0;// + i * 0.5; // Note that NH must be in log10!!!

    k++;
  }

  //z[3] = 0.25; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  float dt = 3.0; // in code unit!

  int *d_Typ;
  float *d_x, *d_y, *d_z, *d_T_p, *d_NH, *d_rho;

  cudaMalloc(&d_Typ, Npart * sizeof(int));
  cudaMalloc(&d_rho, Npart * sizeof(float));

  cudaMalloc(&d_x, Npart * sizeof(float));
  cudaMalloc(&d_y, Npart * sizeof(float));
  cudaMalloc(&d_z, Npart * sizeof(float));

  //cudaMalloc(&d_u_p, Npart * sizeof(float));
  cudaMalloc(&d_T_p, Npart * sizeof(float));
  cudaMalloc(&d_NH, Npart * sizeof(float));

  cudaMemcpy(d_Typ, Typ, Npart * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rho, rho, Npart * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_x, x, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, Npart * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_T_p, T_p, Npart * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_NH, NH, Npart * sizeof(float), cudaMemcpyHostToDevice);


  doCooling<<<1, Npart>>>(d_Typ, d_x, d_y, d_z, d_rho, d_T_p, d_NH,
                          d_nHGrid, d_rGrid, d_NHGrid, d_dtGrid, d_TGrid, d_Tarr,
                          N_nH, N_T, N_r, N_NH, N_t, gamma, kB, mH,
                          nHLowBound, nHUpBound, rLowBound, rUpBound, TLowBound, TUpBound,
                          NHLowBound, NHUpBound, timeLowBound, timeUpBound, dt,
                          unit_time_in_sec, unit_density_in_cgs, unit_length_in_kpc, unit_u_in_cgs, Npart);

  cudaDeviceSynchronize();


  float *TB = new float[Npart];

  for (int i = 0; i < Npart; i++)
  {
    TB[i] = T_p[i];
    //cout << "T Before = " << T_p[i] << endl;
  }
  cout << endl;

  cudaMemcpy(T_p, d_T_p, Npart * sizeof(float), cudaMemcpyDeviceToHost);

  cout << "T After = " << T_p[0] << endl << endl;

  for (int i = 0; i < Npart; i++)
  {
    cout << "r_p, TB, TA = " << rr_p[i] << ", " << TB[i] << ", " << T_p[i] << endl;
  }



}


