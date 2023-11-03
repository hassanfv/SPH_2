
//%%writefile test.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <tuple>
#include "hfvCLibs_v2.h"
#include <cstdlib> // This is ONLY used for the "exit(0)" function !!

// Added the isothermal gravitational field acceleration. (24 May 2023).
// Added the reading of the params.txt file and updated the IC reading file section and function. (22 May 2023).

using namespace std;

int main()
{

  float dt = 8e-7; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 10;
  const int Nup = Nngb + 10;
  const float coeff = 0.005f; // used for smoothing length.
  
  const float kpc_in_cm = 3.086e21;
  
  //*******************************************************************
  //******************* Reading Cooling File **************************
  //*******************************************************************
  ifstream infile("coolHeatGridZ.bin", ios::binary);

  if (!infile) {
    cerr << "Failed to open coolHeatGrid.bin file." << endl;
    return 1;
  }

  // Read the sizes
  int N_kpc, N_T, N_nH, N_Z, N_M, N_Time;
  infile.read(reinterpret_cast<char*>(&N_kpc), sizeof(int));
  infile.read(reinterpret_cast<char*>(&N_nH), sizeof(int));
  infile.read(reinterpret_cast<char*>(&N_Z), sizeof(int));
  infile.read(reinterpret_cast<char*>(&N_T), sizeof(int));
  infile.read(reinterpret_cast<char*>(&N_M), sizeof(int));
  infile.read(reinterpret_cast<char*>(&N_Time), sizeof(int));

  // Allocate and read the densities, temperatures, metallicities, and timeArr arrays
  vector<float> kpcArr(N_kpc);     // float
  vector<float> densities(N_nH);     // float
  vector<float> metallicities(N_Z);  // float
  vector<float> temperatures(N_T);   // float
  vector<float> timeArr(N_Time);     // float

  infile.read(reinterpret_cast<char*>(kpcArr.data()), N_kpc * sizeof(float));
  infile.read(reinterpret_cast<char*>(densities.data()), N_nH * sizeof(float));
  infile.read(reinterpret_cast<char*>(metallicities.data()), N_Z * sizeof(float));
  infile.read(reinterpret_cast<char*>(temperatures.data()), N_T * sizeof(float));
  infile.read(reinterpret_cast<char*>(timeArr.data()), N_Time * sizeof(float));

  // Allocate and read the flattened res and muArr array
  int N_HCool = N_kpc * N_T * N_nH * N_Z * N_Time;
  vector<float> res_flattened(N_HCool);  // float
  vector<float> muArr(N_HCool);  // float
  
  int N_metalz = N_kpc * N_T * N_nH * N_Z * N_M * N_Time;
  vector<float> metalzArr(N_metalz);  // float

  infile.read(reinterpret_cast<char*>(res_flattened.data()), N_HCool * sizeof(float));
  infile.read(reinterpret_cast<char*>(muArr.data()), N_HCool * sizeof(float)); // Note that muA and res_flattedned have the same structure!!
  
  infile.read(reinterpret_cast<char*>(metalzArr.data()), N_metalz * sizeof(float));

  infile.close();
  
  
  float *Temp, *d_Temp, *nH, *d_nH, *Z, *d_Z, *Time, *d_Time, *HCool, *d_HCool, *muA, *d_muA, *kpc, *d_kpc, *metalz, *d_metalz;
  float *U, *d_U; // They will be used only inside the hcooling function as I could not define them inside the hcooling function (GPU climitations!!)
  
  kpc = new float[N_kpc];
  Temp = new float[N_T];
  nH = new float[N_nH];
  Z = new float[N_Z];
  Time = new float[N_Time];
  HCool = new float[N_HCool];
  muA = new float[N_HCool];
  U = new float[N_T];
  metalz = new float[N_metalz];
  
  cudaMalloc(&d_kpc, N_kpc * sizeof(float));
  cudaMalloc(&d_Temp, N_T * sizeof(float));
  cudaMalloc(&d_nH, N_nH * sizeof(float));
  cudaMalloc(&d_Z, N_Z * sizeof(float));
  cudaMalloc(&d_Time, N_Time * sizeof(float));
  cudaMalloc(&d_HCool, N_HCool * sizeof(float));
  cudaMalloc(&d_muA, N_HCool * sizeof(float));
  cudaMalloc(&d_U, N_T * sizeof(float));
  cudaMalloc(&d_metalz, N_metalz * sizeof(float));
  
  for (int i = 0; i < N_kpc; i++)
  {
    kpc[i] = kpcArr[i];
  }
  
  for (int i = 0; i < N_T; i++)
  {
    Temp[i] = temperatures[i];
    U[i] = 0.0f;
  }
  
  for (int i = 0; i < N_nH; i++)
  {
    nH[i] = densities[i];
  }
  
  for (int i = 0; i < N_Z; i++)
  {
    Z[i] = metallicities[i];
  }
  
  for (int i = 0; i < N_Time; i++)
  {
    Time[i] = timeArr[i];
  }
  
  for (int i = 0; i < N_HCool; i++)
  {
    HCool[i] = res_flattened[i];
  }
  
  for (int i = 0; i < N_HCool; i++)
  {
    muA[i] = muArr[i];
  }

  for (int i = 0; i < N_metalz; i++)
  {
    metalz[i] = metalzArr[i];
  }

  // Copy from Host to Device // Will ONLY be use in one GPU (i.e. GPU 0) !
  cudaMemcpy(d_kpc, kpc, N_kpc * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Temp, Temp, N_T * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nH, nH, N_nH * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Z, Z, N_Z * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Time, Time, N_Time * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_HCool, HCool, N_HCool * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_muA, muA, N_HCool * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_U, U, N_T * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_metalz, metalz, N_metalz * sizeof(float), cudaMemcpyHostToDevice);

  //********************************************************************
  //**************** Reading the params.txt file ***********************
  //********************************************************************
  std::string filename;
  int N;
  float G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
        unitLength_in_cm;

  readParams(filename, N, G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
             unitLength_in_cm);

  std::cout << "filename: " << filename << "\n";
  std::cout << "N: " << N << "\n";
  std::cout << "G: " << G << "\n";
  std::cout << "L_AGN_code_unit: " << L_AGN_code_unit << "\n";
  std::cout << "M_dot_in_code_unit: " << M_dot_in << "\n";
  std::cout << "vin_in_code_unit: " << v_in << "\n";
  std::cout << "u_for_10K_Temp: " << u_for_10K_Temp << "\n";
  std::cout << "m_sph_high_res: " << m_sph_high_res << "\n";
  std::cout << "sigma: " << sigma << "\n";
  
  std::cout << "UnitDensity_in_cgs: " << UnitDensity_in_cgs << "\n";
  std::cout << "Unit_u_in_cgs: " << Unit_u_in_cgs << "\n";
  std::cout << "unitTime_in_s: " << unitTime_in_s << "\n";
  
  std::cout << "unitLength_in_cm: " << unitLength_in_cm << "\n";
  
  //*********************************************************************
  //******************** Reading the IC file ****************************
  //*********************************************************************
  auto data = readVectorsFromFile(N, filename);

  std::vector<int> &Typvec = std::get<0>(data);
  std::vector<float> &xvec = std::get<1>(data);
  std::vector<float> &yvec = std::get<2>(data);
  std::vector<float> &zvec = std::get<3>(data);
  std::vector<float> &vxvec = std::get<4>(data);
  std::vector<float> &vyvec = std::get<5>(data);
  std::vector<float> &vzvec = std::get<6>(data);
  std::vector<float> &massvec = std::get<7>(data);
  std::vector<float> &hvec = std::get<8>(data);
  std::vector<float> &epsvec = std::get<9>(data);
  std::vector<float> &uvec = std::get<10>(data);

  // declaring the arrays.
  int *Typ;
  float *x, *y, *z, *vx, *vy, *vz;
  float *mass, *h, *rho;
  float *accx, *accy, *accz, *eps;
  float *P, *csnd, *divV, *curlV;
  float *accx_sph, *accy_sph, *accz_sph;
  float *accx_tot, *accy_tot, *accz_tot;
  float *abs_acc_g, *abs_acc_tot, *v_sig, *dh_dt;
  float *u, *dudt, *utprevious;
  float *Nngb_previous; // Note that both are floats and not int! check smoothing func. to see why!
  float *dt_particles;
  float *dudt_pre;

  float gammah = 5.0f / 3.0f;
  float GAMMA_MINUS1 = gammah - 1.0f;
  
  int N_ionFrac = N * N_M; // We have N_M species for each particle (N = total number of particles)
  
  float *ionFrac; // Will ONLY be use in one GPU (i.e. GPU 0) !

  Typ = new int[N];

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
  h = new float[N];
  rho = new float[N];
  eps = new float[N];
  P = new float[N];
  csnd = new float[N];

  divV = new float[N];
  curlV = new float[N];

  accx_sph = new float[N];
  accy_sph = new float[N];
  accz_sph = new float[N];

  accx_tot = new float[N];
  accy_tot = new float[N];
  accz_tot = new float[N];

  abs_acc_g = new float[N];
  abs_acc_tot = new float[N];
  v_sig = new float[N];
  dh_dt = new float[N];
  dt_particles = new float[N];

  u = new float[N];
  dudt = new float[N];
  utprevious = new float[N];
  
  dudt_pre = new float[N];

  Nngb_previous = new float[N];
  
  ionFrac = new float[N_ionFrac];
  
  // Initialize x, y, z, etc on the Host.
  for (int i = 0; i < N; i++)
  {
    Typ[i] = Typvec[i];

    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];

    vx[i] = vxvec[i];
    vy[i] = vyvec[i];
    vz[i] = vzvec[i];

    mass[i] = massvec[i];
    eps[i] = epsvec[i];

    accx[i] = 0.0f;
    accy[i] = 0.0f;
    accz[i] = 0.0f;

    accx_tot[i] = 0.0f;
    accy_tot[i] = 0.0f;
    accz_tot[i] = 0.0f;

    abs_acc_g[i] = 0.0f;
    abs_acc_tot[i] = 0.0f;
    v_sig[i] = 0.0f;

    h[i] = hvec[i]; // place holder.
    rho[i] = 0.0f;  // place holder.
    P[i] = 0.0f;    // placeholder.
    csnd[i] = 0.0f; // placeholder.

    divV[i] = 0.0f;  // placeholder.
    curlV[i] = 0.0f; // placeholder.

    accx_sph[i] = 0.0f;
    accy_sph[i] = 0.0f;
    accz_sph[i] = 0.0f;

    dh_dt[i] = 0.0f;

    u[i] = uvec[i];
    dudt[i] = 0.0f;
    utprevious[i] = 0.0f;
    
    dudt_pre[i] = 0.0f;

    dt_particles[i] = 0.0f;

    if (Typ[i] == 0)
    {
      Nngb_previous[i] = Nngb_f;
    }
    else
    {
      Nngb_previous[i] = 0.0f;
    }
  }
  
  for (int i = 0; i < N_ionFrac; i++)
  {
    ionFrac[i] = 0.0;
  }
  
  
  int nGPUs;
  cudaGetDeviceCount(&nGPUs);

  cout << "nGPUs = " << nGPUs << endl;
  
  // Pointers for device memory
  int *d_Typ[nGPUs];
  float *d_x[nGPUs], *d_y[nGPUs], *d_z[nGPUs], *d_vx[nGPUs], *d_vy[nGPUs], *d_vz[nGPUs];
  float *d_mass[nGPUs], *d_h[nGPUs], *d_rho[nGPUs];
  float *d_accx[nGPUs], *d_accy[nGPUs], *d_accz[nGPUs], *d_eps[nGPUs];
  float *d_P[nGPUs], *d_csnd[nGPUs], *d_divV[nGPUs], *d_curlV[nGPUs];
  float *d_accx_sph[nGPUs], *d_accy_sph[nGPUs], *d_accz_sph[nGPUs];
  float *d_accx_tot[nGPUs], *d_accy_tot[nGPUs], *d_accz_tot[nGPUs];
  float *d_abs_acc_g[nGPUs], *d_abs_acc_tot[nGPUs];
  float *d_v_sig[nGPUs], *d_dh_dt[nGPUs], *d_u[nGPUs], *d_dudt[nGPUs];
  float *d_utprevious[nGPUs];
  float *d_Nngb_previous[nGPUs]; // Note that both are floats and not int! check smoothing func. to see why!
  float *d_dt_particles[nGPUs];
  float *d_dudt_pre[nGPUs];
  float *d_ionFrac; // Will ONLY be use in one GPU (i.e. GPU 0) !

  cudaMalloc(&d_ionFrac, N_ionFrac * sizeof(float));
  cudaMemcpy(d_ionFrac, ionFrac, N_ionFrac * sizeof(float), cudaMemcpyHostToDevice);

for (int i = 0; i < nGPUs; i++)
  {

    cudaMalloc(&d_Typ[i], N * sizeof(int));

    cudaMalloc(&d_x[i], N * sizeof(float));
    cudaMalloc(&d_y[i], N * sizeof(float));
    cudaMalloc(&d_z[i], N * sizeof(float));

    cudaMalloc(&d_vx[i], N * sizeof(float));
    cudaMalloc(&d_vy[i], N * sizeof(float));
    cudaMalloc(&d_vz[i], N * sizeof(float));

    cudaMalloc(&d_accx[i], N * sizeof(float));
    cudaMalloc(&d_accy[i], N * sizeof(float));
    cudaMalloc(&d_accz[i], N * sizeof(float));

    cudaMalloc(&d_mass[i], N * sizeof(float));
    cudaMalloc(&d_h[i], N * sizeof(float));
    cudaMalloc(&d_rho[i], N * sizeof(float));
    cudaMalloc(&d_eps[i], N * sizeof(float));
    cudaMalloc(&d_P[i], N * sizeof(float));
    cudaMalloc(&d_csnd[i], N * sizeof(float));

    cudaMalloc(&d_divV[i], N * sizeof(float));
    cudaMalloc(&d_curlV[i], N * sizeof(float));

    cudaMalloc(&d_accx_sph[i], N * sizeof(float));
    cudaMalloc(&d_accy_sph[i], N * sizeof(float));
    cudaMalloc(&d_accz_sph[i], N * sizeof(float));

    cudaMalloc(&d_accx_tot[i], N * sizeof(float));
    cudaMalloc(&d_accy_tot[i], N * sizeof(float));
    cudaMalloc(&d_accz_tot[i], N * sizeof(float));

    cudaMalloc(&d_abs_acc_g[i], N * sizeof(float));
    cudaMalloc(&d_abs_acc_tot[i], N * sizeof(float));
    cudaMalloc(&d_v_sig[i], N * sizeof(float));
    cudaMalloc(&d_dh_dt[i], N * sizeof(float));
    cudaMalloc(&d_dt_particles[i], N * sizeof(float));

    cudaMalloc(&d_u[i], N * sizeof(float));
    cudaMalloc(&d_dudt[i], N * sizeof(float));
    cudaMalloc(&d_utprevious[i], N * sizeof(float));
    
    cudaMalloc(&d_dudt_pre[i], N * sizeof(float));

    cudaMalloc(&d_Nngb_previous[i], N * sizeof(float));
      

  // Copy from Host to Device.
  cudaMemcpy(d_Typ[i], Typ, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_x[i], x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y[i], y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z[i], z, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_vx[i], vx, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy[i], vy, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vz[i], vz, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx[i], accx, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy[i], accy, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz[i], accz, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_mass[i], mass, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h[i], h, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_rho[i], rho, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_eps[i], eps, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P[i], P, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csnd[i], csnd, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_divV[i], divV, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_curlV[i], curlV, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx_sph[i], accx_sph, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy_sph[i], accy_sph, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz_sph[i], accz_sph, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx_tot[i], accx_tot, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy_tot[i], accy_tot, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz_tot[i], accz_tot, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_abs_acc_g[i], abs_acc_g, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_abs_acc_tot[i], abs_acc_tot, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_sig[i], v_sig, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dh_dt[i], dh_dt, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dt_particles[i], dt_particles, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_u[i], u, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dudt[i], dudt, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_utprevious[i], utprevious, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_dudt_pre[i], dudt_pre, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_Nngb_previous[i], Nngb_previous, N * sizeof(float), cudaMemcpyHostToDevice);
  
  }
  
  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  const float visc_alpha = 1.0f;

  float t;

  t = 0.0f;

  float tEnd = 5.0f;
  float Nt = ceil(tEnd / dt) + 1;

  float Zmetal = 0.1; // ==> [Z/H] = -1.
  
  // =========== Determining the beg and end index of the array for each GPU ===========

  const int N_per_GPU = N / nGPUs;
  const int remainder = N % nGPUs;

  int *beg, *end, *MLen; // MLen means Memory length which is end - beg!

  beg = new int[nGPUs];
  end = new int[nGPUs];
  MLen = new int[nGPUs];

  for (int rank = 0; rank < nGPUs; rank++)
  {
    if (rank < remainder)
    {
      beg[rank] = rank * (N_per_GPU + 1);
      end[rank] = beg[rank] + N_per_GPU + 1;
      MLen[rank] = end[rank] - beg[rank];
    }
    else
    {
      beg[rank] = rank * N_per_GPU + remainder;
      end[rank] = beg[rank] + N_per_GPU;
      MLen[rank] = end[rank] - beg[rank];
    }
  }
  //=========================================================================================


  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!! START of smoothing_h !!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    cout << "AA" << endl;

    // Run the kernel only on a portion of the arrays
    smoothing_h_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i], d_h[i],
                                              Ndown, Nup, coeff, Nngb_f, d_Nngb_previous[i], d_divV[i], dt, N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Now d_h[i] for each GPU, i, contains the results for its own portion of the data!

  // Copy results to main GPU.
  int NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_h[0] + NN, 0, d_h[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // The main GPU now has the full results in d_h[0]

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already has the data, that's why i starts with 1 !
  {
    cudaMemcpyPeer(d_h[i], i, d_h[0], 0, N * sizeof(float));
  }

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF smoothing_h !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!! START of getDensity !!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    getDensity_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i], d_mass[i],
                                             d_rho[i], d_h[i], N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Now d_accx[i], d_accy[i], and d_accz[i] for each GPU i contain the results for their portion

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_rho[0] + NN, 0, d_rho[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // The main GPU now has the full results in d_rho[0]

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_rho[i], i, d_rho[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!! END of getDensity !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!! START of acc_g !!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    acc_g_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i],
                                        d_eps[i], d_accx[i], d_accy[i], d_accz[i], d_mass[i],
                                        G, N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Now d_accx[i], d_accy[i], and d_accz[i] for each GPU i contain the results for their portion

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_accx[0] + NN, 0, d_accx[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_accy[0] + NN, 0, d_accy[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_accz[0] + NN, 0, d_accz[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // The main GPU now has the full results in d_accx[0], d_accy[0], d_accz[0]

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_accx[i], i, d_accx[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_accy[i], i, d_accy[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_accz[i], i, d_accz[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF acc_g !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! START OF getPressure !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    getPressure_Adiabatic_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_P[i], d_rho[i], d_u[i],
                                                        gammah);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_P[0] + NN, 0, d_P[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_P[i], i, d_P[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF getPressure !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! START OF getCsound !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    getCsound_Adiabatic_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_csnd[i], d_u[i], gammah);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_csnd[0] + NN, 0, d_csnd[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_csnd[i], i, d_csnd[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF getCsound !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! START OF div_curlV !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    div_curlVel_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_divV[i], d_curlV[i],
                                              d_x[i], d_y[i], d_z[i],
                                              d_vx[i], d_vy[i], d_vz[i],
                                              d_rho[i], d_mass[i], d_h[i], N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_divV[0] + NN, 0, d_divV[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_curlV[0] + NN, 0, d_curlV[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_divV[i], i, d_divV[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_curlV[i], i, d_curlV[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF div_curlV !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! START OF acc_sph !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    acc_sph_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i],
                                          d_vx[i], d_vy[i], d_vz[i], d_h[i], d_csnd[i], d_rho[i],
                                          d_divV[i], d_curlV[i], d_mass[i], d_P[i],
                                          d_accx_sph[i], d_accy_sph[i], d_accz_sph[i], visc_alpha, N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_accx_sph[0] + NN, 0, d_accx_sph[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_accy_sph[0] + NN, 0, d_accy_sph[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_accz_sph[0] + NN, 0, d_accz_sph[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_accx_sph[i], i, d_accx_sph[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_accy_sph[i], i, d_accy_sph[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_accz_sph[i], i, d_accz_sph[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF acc_sph !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! START OF acc_tot !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    acc_g_sph_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i],
                                            d_accx_tot[i], d_accy_tot[i], d_accz_tot[i],
                                            d_accx[i], d_accy[i], d_accz[i],
                                            d_accx_sph[i], d_accy_sph[i], d_accz_sph[i]);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_accx_tot[0] + NN, 0, d_accx_tot[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_accy_tot[0] + NN, 0, d_accy_tot[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_accz_tot[0] + NN, 0, d_accz_tot[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_accx_tot[i], i, d_accx_tot[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_accy_tot[i], i, d_accy_tot[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_accz_tot[i], i, d_accz_tot[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF acc_tot !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! START OF du_dt !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    get_dU_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i],
                                         d_vx[i], d_vy[i], d_vz[i], d_h[i], d_csnd[i], d_rho[i],
                                         d_divV[i], d_curlV[i], d_mass[i], d_P[i], d_dudt[i],
                                         visc_alpha, N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_dudt[0] + NN, 0, d_dudt[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_dudt[i], i, d_dudt[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF du_dt !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! START OF u evolution !!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    // Run the kernel only on a portion of the arrays
    u_updater_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_u[i], d_dudt[i],
                                            d_utprevious[i], dt);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_u[0] + NN, 0, d_u[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(d_utprevious[0] + NN, 0, d_utprevious[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
  {
    cudaMemcpyPeer(d_u[i], i, d_u[0], 0, N * sizeof(float));
    cudaMemcpyPeer(d_utprevious[i], i, d_utprevious[0], 0, N * sizeof(float));
  }
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF u evolution !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  cudaSetDevice(0); // Set the first GPU as active ===> Take note of "d_leftover_mass" that is only recognizable in GPU 0 !

  const float C_CFL = 0.25;

  float h_min, h_max, h_mean;

  float leftover_mass = 0.0f;
  float *d_leftover_mass;
  cudaMalloc((void **)&d_leftover_mass, sizeof(float));
  cudaMemcpy(d_leftover_mass, &leftover_mass, sizeof(float), cudaMemcpyHostToDevice);

  // **************************************************************
  // *********************** MAIN LOOP ****************************
  // **************************************************************

  int counter = 0; // This is used to save fewer output files, e.g. 1 snap-shot per 20 time-step!

  while (t < tEnd)
  {

    auto begin = std::chrono::high_resolution_clock::now();

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF velocity evolution !!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      v_evolve_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_vx[i], d_vy[i], d_vz[i],
                                             d_accx_tot[i], d_accy_tot[i], d_accz_tot[i], dt);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_vx[0] + NN, 0, d_vx[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_vy[0] + NN, 0, d_vy[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_vz[0] + NN, 0, d_vz[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_vx[i], i, d_vx[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_vy[i], i, d_vy[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_vz[i], i, d_vz[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF velocity evolution !!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF position evolution !!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      r_evolve_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i],
                                             d_vx[i], d_vy[i], d_vz[i], dt);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_x[0] + NN, 0, d_x[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_y[0] + NN, 0, d_y[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_z[0] + NN, 0, d_z[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_x[i], i, d_x[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_y[i], i, d_y[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_z[i], i, d_z[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF position evolution !!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!! START of smoothing_h !!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      smoothing_h_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i], d_h[i],
                                                Ndown, Nup, coeff, Nngb_f, d_Nngb_previous[i], d_divV[i], dt, N);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Now d_h[i] for each GPU, i, contains the results for its own portion portion of the data!

    // Copy results to main GPU.
    int NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_h[0] + NN, 0, d_h[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // The main GPU now has the full results in d_accx[0], d_accy[0], d_accz[0]

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_h[i], i, d_h[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF smoothing_h !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!! START OF (Set eps of Gas equal to h) !!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      set_eps_of_gas_to_h_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_eps[i], d_h[i]);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Now d_h[i] for each GPU, i, contains the results for its own portion portion of the data!

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_eps[0] + NN, 0, d_eps[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // The main GPU now has the full results in d_accx[0], d_accy[0], d_accz[0]

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_eps[i], i, d_eps[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!! END OF (Set eps of Gas equal to h) !!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!! START of getDensity !!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      getDensity_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i], d_mass[i],
                                               d_rho[i], d_h[i], N);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Now d_accx[i], d_accy[i], and d_accz[i] for each GPU i contain the results for their portion

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_rho[0] + NN, 0, d_rho[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // The main GPU now has the full results in d_rho[0]

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_rho[i], i, d_rho[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!! END of getDensity !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!! START of acc_g !!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    auto T_acc_g = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      acc_g_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i],
                                          d_eps[i], d_accx[i], d_accy[i], d_accz[i], d_mass[i],
                                          G, N);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Now d_accx[i], d_accy[i], and d_accz[i] for each GPU i contain the results for their portion

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_accx[0] + NN, 0, d_accx[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accy[0] + NN, 0, d_accy[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accz[0] + NN, 0, d_accz[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // The main GPU now has the full results in d_accx[0], d_accy[0], d_accz[0]

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_accx[i], i, d_accx[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accy[i], i, d_accy[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accz[i], i, d_accz[0], 0, N * sizeof(float));
    }
    auto end_acc_g = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_g = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_g - T_acc_g);
    cout << "T_acc_g = " << elapsed_acc_g.count() * 1e-9 << endl;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF acc_g !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF getPressure !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      getPressure_Adiabatic_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_P[i], d_rho[i], d_u[i],
                                                          gammah);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_P[0] + NN, 0, d_P[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_P[i], i, d_P[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF getPressure !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF getCsound !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      getCsound_Adiabatic_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_csnd[i], d_u[i], gammah);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_csnd[0] + NN, 0, d_csnd[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_csnd[i], i, d_csnd[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF getCsound !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF div_curlV !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    auto T_divCurl = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      div_curlVel_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_divV[i], d_curlV[i],
                                                d_x[i], d_y[i], d_z[i],
                                                d_vx[i], d_vy[i], d_vz[i],
                                                d_rho[i], d_mass[i], d_h[i], N);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_divV[0] + NN, 0, d_divV[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_curlV[0] + NN, 0, d_curlV[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_divV[i], i, d_divV[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_curlV[i], i, d_curlV[0], 0, N * sizeof(float));
    }
    auto end_divCurl = std::chrono::high_resolution_clock::now();
    auto elapsed_divCurl = std::chrono::duration_cast<std::chrono::nanoseconds>(end_divCurl - T_divCurl);
    cout << "T_divCurl = " << elapsed_divCurl.count() * 1e-9 << endl;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF div_curlV !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF acc_sph !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    auto T_acc_sph = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      acc_sph_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i],
                                            d_vx[i], d_vy[i], d_vz[i], d_h[i], d_csnd[i], d_rho[i],
                                            d_divV[i], d_curlV[i], d_mass[i], d_P[i],
                                            d_accx_sph[i], d_accy_sph[i], d_accz_sph[i], visc_alpha, N);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_accx_sph[0] + NN, 0, d_accx_sph[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accy_sph[0] + NN, 0, d_accy_sph[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accz_sph[0] + NN, 0, d_accz_sph[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_accx_sph[i], i, d_accx_sph[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accy_sph[i], i, d_accy_sph[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accz_sph[i], i, d_accz_sph[0], 0, N * sizeof(float));
    }
    auto end_acc_sph = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_sph = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_sph - T_acc_sph);
    cout << "T_acc_sph = " << elapsed_acc_sph.count() * 1e-9 << endl;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF acc_sph !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF acc_tot !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      acc_g_sph_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i],
                                              d_accx_tot[i], d_accy_tot[i], d_accz_tot[i],
                                              d_accx[i], d_accy[i], d_accz[i],
                                              d_accx_sph[i], d_accy_sph[i], d_accz_sph[i]);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_accx_tot[0] + NN, 0, d_accx_tot[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accy_tot[0] + NN, 0, d_accy_tot[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accz_tot[0] + NN, 0, d_accz_tot[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_accx_tot[i], i, d_accx_tot[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accy_tot[i], i, d_accy_tot[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accz_tot[i], i, d_accz_tot[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF acc_tot !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //! START OF Isothermal Gravity (Richings et al - 2018) !!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      galaxy_isothermal_potential_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i],
                                                                d_x[i], d_y[i], d_z[i], d_accx_tot[i],
                                                                d_accy_tot[i], d_accz_tot[i], sigma, G);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_accx_tot[0] + NN, 0, d_accx_tot[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accy_tot[0] + NN, 0, d_accy_tot[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_accz_tot[0] + NN, 0, d_accz_tot[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_accx_tot[i], i, d_accx_tot[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accy_tot[i], i, d_accy_tot[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_accz_tot[i], i, d_accz_tot[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //! END OF Isothermal Gravity (Richings et al - 2018) !!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF velocity evolution !!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      v_evolve_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_vx[i], d_vy[i], d_vz[i],
                                             d_accx_tot[i], d_accy_tot[i], d_accz_tot[i], dt);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_vx[0] + NN, 0, d_vx[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_vy[0] + NN, 0, d_vy[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_vz[0] + NN, 0, d_vz[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_vx[i], i, d_vx[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_vy[i], i, d_vy[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_vz[i], i, d_vz[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF velocity evolution !!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF du_dt !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    auto T_dU = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      get_dU_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i],
                                           d_vx[i], d_vy[i], d_vz[i], d_h[i], d_csnd[i], d_rho[i],
                                           d_divV[i], d_curlV[i], d_mass[i], d_P[i], d_dudt[i],
                                           visc_alpha, N);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_dudt[0] + NN, 0, d_dudt[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_dudt[i], i, d_dudt[0], 0, N * sizeof(float));
    }
    auto end_dU = std::chrono::high_resolution_clock::now();
    auto elapsed_dU = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dU - T_dU);
    cout << "T_dU = " << elapsed_dU.count() * 1e-9 << endl;
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF du_dt !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! START OF u evolution !!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);

      // Run the kernel only on a portion of the arrays
      u_updater_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_u[i], d_dudt[i],
                                              d_utprevious[i], dt);
    }

    // Synchronize to make sure computation is done before proceeding
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Copy results to main GPU.
    NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(d_u[0] + NN, 0, d_u[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(d_utprevious[0] + NN, 0, d_utprevious[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_u[i], i, d_u[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_utprevious[i], i, d_utprevious[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!! END OF u evolution !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!! START OF Heating & Cooling !!!!!!!!!!!!!!!!! Operates ONLY on GPU 0 !!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    cudaSetDevice(0); // !!!!! Heating and Cooling are performed ONLY by the first GPU, i.e., GPU 0 !!!
    
    auto T_cool = std::chrono::high_resolution_clock::now();
    hcoolingx<<<gridSize, blockSize>>>(d_Typ[0], d_u[0], d_U, d_rho[0], d_metalz, Zmetal, dt, // Zmetal is the gas metallicity assumed.
                                      d_nH, d_Z, d_HCool, d_ionFrac, d_Time, d_x[0], d_y[0], d_z[0],
                                      d_muA, d_Temp, d_kpc, UnitDensity_in_cgs, unitTime_in_s, 
                                      Unit_u_in_cgs, unitLength_in_cm, kpc_in_cm, GAMMA_MINUS1,
                                      N_kpc, N_nH, N_Z, N_T, N_M, N_Time, N);
    cudaDeviceSynchronize();
    auto end_cool = std::chrono::high_resolution_clock::now();
    auto elapsed_cool = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cool - T_cool);
    cout << "T_cool = " << elapsed_cool.count() * 1e-9 << endl;
    
    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_u[i], i, d_u[0], 0, N * sizeof(float)); // Only u is updated during heating-cooling process!
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!! END OF Heating & Cooling !!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    //-------------------------------------------------

    cudaMemcpy(rho, d_rho[0], N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
    {
      cout << "AAA = " << rho[i] << endl;
    }

    //------------ SAVING SNAP-SHOTS ------------
    if (!(counter % 50))
    //if (counter > -1)
    {
      cudaMemcpy(Typ, d_Typ[0], N * sizeof(int), cudaMemcpyDeviceToHost);

      cudaMemcpy(x, d_x[0], N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y[0], N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z[0], N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(vx, d_vx[0], N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy[0], N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vz, d_vz[0], N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(rho, d_rho[0], N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h, d_h[0], N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(u, d_u[0], N * sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaMemcpy(mass, d_mass[0], N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(ionFrac, d_ionFrac, N_ionFrac * sizeof(float), cudaMemcpyDeviceToHost);

      // Specify the output file name
      std::string filename = "./Outputs/G-" + to_string(t * 10) + ".bin";
      // Save the arrays to binary format
      saveArraysToBinary(filename, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, Typ, N, N_ionFrac);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << "Elapsed time = " << elapsed.count() * 1e-9 << endl;
    cout << endl;
    
    // We are still on GPU 0 ====> see above for "cudaSetDevice(0);" line !!!

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //*************** START of Outflow particle injection ****************
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // Generate a seed using the high resolution clock
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    unsigned long long seed = static_cast<unsigned long long>(nanos);
    //------------

    cudaMemcpy(h, d_h[0], N * sizeof(float), cudaMemcpyDeviceToHost);
    h_min = min_finder(Typ, h, N);
    h_max = max_finder(Typ, h, N);
    h_mean = 0.5f * (h_min + h_max);

    outflow_injector<<<gridSize, blockSize>>>(d_Typ[0], d_x[0], d_y[0], d_z[0],
                                              d_vx[0], d_vy[0], d_vz[0],
                                              d_h[0], d_eps[0], d_mass[0],
                                              Nngb_f, d_Nngb_previous[0],
                                              d_u[0], M_dot_in, v_in,
                                              m_sph_high_res, u_for_10K_Temp,
                                              h_mean, d_leftover_mass, dt, N,
                                              seed);
    cudaDeviceSynchronize();

    // Copy results back to each device for next iteration
    for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(d_Typ[i], i, d_Typ[0], 0, N * sizeof(float));
      
      cudaMemcpyPeer(d_x[i], i, d_x[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_y[i], i, d_y[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_z[i], i, d_z[0], 0, N * sizeof(float));
      
      cudaMemcpyPeer(d_vx[i], i, d_vx[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_vy[i], i, d_vy[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_vz[i], i, d_vz[0], 0, N * sizeof(float));
      
      cudaMemcpyPeer(d_h[i], i, d_h[0], 0, N * sizeof(float));
      
      cudaMemcpyPeer(d_eps[i], i, d_eps[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_mass[i], i, d_mass[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_Nngb_previous[i], i, d_Nngb_previous[0], 0, N * sizeof(float));
      cudaMemcpyPeer(d_u[i], i, d_u[0], 0, N * sizeof(float));
    }
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //**************** END of Outflow particle injection *****************
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    if (!(counter % 1))
    {
      cout << "Adopted dt = " << dt << endl;
      cout << "current t = " << t << endl;
      cout << "*****************************" << endl;
      cout << endl;
    }

    counter++;
  }

  delete[] Typ;
  delete[] x;
  delete[] y;
  delete[] z;
  delete[] vx;
  delete[] vy;
  delete[] vz;
  delete[] mass;
  delete[] h;
  delete[] rho;
  delete[] accx;
  delete[] accy;
  delete[] accz;
  delete[] eps;
  delete[] P;
  delete[] csnd;
  delete[] divV;
  delete[] curlV;
  delete[] accx_sph;
  delete[] accy_sph;
  delete[] accz_sph;
  delete[] accx_tot;
  delete[] accy_tot;
  delete[] accz_tot;
  delete[] abs_acc_g;
  delete[] abs_acc_tot;
  delete[] v_sig;
  delete[] dh_dt;
  delete[] u;
  delete[] dudt;
  delete[] utprevious;

  cudaFree(d_Typ);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_vx);
  cudaFree(d_vy);
  cudaFree(d_vz);
  cudaFree(d_mass);
  cudaFree(d_h);
  cudaFree(d_rho);
  cudaFree(d_accx);
  cudaFree(d_accy);
  cudaFree(d_accz);
  cudaFree(d_P);
  cudaFree(d_csnd);
  cudaFree(d_divV);
  cudaFree(d_curlV);
  cudaFree(d_accx_sph);
  cudaFree(d_accy_sph);
  cudaFree(d_accz_sph);
  cudaFree(d_accx_tot);
  cudaFree(d_accy_tot);
  cudaFree(d_accz_tot);
  cudaFree(d_abs_acc_g);
  cudaFree(d_abs_acc_tot);
  cudaFree(d_v_sig);
  cudaFree(d_dh_dt);
  cudaFree(d_u);
  cudaFree(d_dudt);
  cudaFree(d_utprevious);
}
