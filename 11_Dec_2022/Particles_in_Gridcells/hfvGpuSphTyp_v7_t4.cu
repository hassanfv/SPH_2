%%writefile test.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <tuple>
#include <algorithm>
#include "hfvCLibs_v7_t5.h"
#include <cstdlib> // This is ONLY used for the "exit(0)" function !!

// Added the isothermal gravitational field acceleration. (24 May 2023).
// Added the reading of the params.txt file and updated the IC reading file section and function. (22 May 2023).

using namespace std;


//===== findNonZeroPositiveMin
float findNonZeroPositiveMin(const float* arr, size_t size) {
    float minVal = std::numeric_limits<float>::max(); // Start with the maximum float value

    for (size_t i = 0; i < size; ++i) {
        if (arr[i] > 0.0f && arr[i] < minVal) { // Check if the value is positive and non-zero
            minVal = arr[i];
        }
    }

    // If minVal is still std::numeric_limits<float>::max(), there were no non-zero positive values
    return (minVal == std::numeric_limits<float>::max()) ? 0.0f : minVal;
}




//====== Helper function to check CUDA errors
void checkCudaErrors(cudaError_t cudaStatus, const char* msg) {
    if (cudaStatus != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " " << cudaGetErrorString(cudaStatus) << endl;
        exit(-1);
    }
}

int main()
{

  float dt = 2e-7; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  //const int Nngb_f = 64.0f; // used in smoothing func.
  //const int Nngb = 64;
  //const int Ndown = Nngb - 5;
  //const int Nup = Nngb + 5;
  //const float coeff = 0.005f; // used for smoothing length.
  
  //const float kpc_in_cm = 3.086e21;
  

  //********************************************************************
  //**************** Reading the params.txt file ***********************
  //********************************************************************
  std::string filename;
  int N, ndx_BH;
  float G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
        unitLength_in_cm;

  readParams(filename, N, ndx_BH, G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
             unitLength_in_cm);

  std::cout << "filename: " << filename << "\n";
  std::cout << "N: " << N << "\n";
  std::cout << "ndx_BH: " << ndx_BH << "\n";
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
  auto data = readVectorsFromFile(filename);

  std::vector<int> &Typvec = std::get<0>(data);
  std::vector<float> &xvec = std::get<1>(data);
  std::vector<float> &yvec = std::get<2>(data);
  std::vector<float> &zvec = std::get<3>(data);
  std::vector<float> &vxvec = std::get<4>(data);
  std::vector<float> &vyvec = std::get<5>(data);
  std::vector<float> &vzvec = std::get<6>(data);
  std::vector<float> &uvec = std::get<7>(data);
  std::vector<float> &hvec = std::get<8>(data);
  std::vector<float> &epsvec = std::get<9>(data);
  std::vector<float> &massvec = std::get<10>(data);


  // declaring the arrays.
  int *Typ, *d_Typ;
  float *x, *d_x, *y, *d_y, *z, *d_z, *vx, *d_vx, *vy, *d_vy, *vz, *d_vz;
  float *mass, *d_mass, *h, *d_h, *rho, *d_rho;
  float *accx, *accy, *accz, *d_accx, *d_accy, *d_accz, *eps, *d_eps;
  float *P, *d_P, *csnd, *d_csnd, *divV, *d_divV, *curlV, *d_curlV;
  float *accx_sph, *accy_sph, *accz_sph, *d_accx_sph, *d_accy_sph, *d_accz_sph;
  float *accx_tot, *accy_tot, *accz_tot, *d_accx_tot, *d_accy_tot, *d_accz_tot;
  float *abs_acc_g, *abs_acc_tot, *v_sig, *dh_dt, *d_abs_acc_g, *d_abs_acc_tot;
  float *d_v_sig, *d_dh_dt, *u, *dudt, *d_u, *d_dudt, *utprevious;
  float *d_utprevious;
  float *Nngb_previous, *d_Nngb_previous; // Note that both are floats and not int! check smoothing func. to see why!
  float *dt_particles, *d_dt_particles;
  
  float *dudt_pre, *d_dudt_pre;

  float gammah = 5.0f / 3.0f;
  float GAMMA_MINUS1 = gammah - 1.0f;
  
  //int N_ionFrac = N * N_M; // We have N_M species for each particle (N = total number of particles)
  
  //float *ionFrac, *d_ionFrac;

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
  
  //ionFrac = new float[N_ionFrac];

  cudaMalloc(&d_Typ, N * sizeof(int));

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_z, N * sizeof(float));

  cudaMalloc(&d_vx, N * sizeof(float));
  cudaMalloc(&d_vy, N * sizeof(float));
  cudaMalloc(&d_vz, N * sizeof(float));

  cudaMalloc(&d_accx, N * sizeof(float));
  cudaMalloc(&d_accy, N * sizeof(float));
  cudaMalloc(&d_accz, N * sizeof(float));

  cudaMalloc(&d_mass, N * sizeof(float));
  cudaMalloc(&d_h, N * sizeof(float));
  cudaMalloc(&d_rho, N * sizeof(float));
  cudaMalloc(&d_eps, N * sizeof(float));
  cudaMalloc(&d_P, N * sizeof(float));
  cudaMalloc(&d_csnd, N * sizeof(float));

  cudaMalloc(&d_divV, N * sizeof(float));
  cudaMalloc(&d_curlV, N * sizeof(float));

  cudaMalloc(&d_accx_sph, N * sizeof(float));
  cudaMalloc(&d_accy_sph, N * sizeof(float));
  cudaMalloc(&d_accz_sph, N * sizeof(float));

  cudaMalloc(&d_accx_tot, N * sizeof(float));
  cudaMalloc(&d_accy_tot, N * sizeof(float));
  cudaMalloc(&d_accz_tot, N * sizeof(float));

  cudaMalloc(&d_abs_acc_g, N * sizeof(float));
  cudaMalloc(&d_abs_acc_tot, N * sizeof(float));
  cudaMalloc(&d_v_sig, N * sizeof(float));
  cudaMalloc(&d_dh_dt, N * sizeof(float));
  cudaMalloc(&d_dt_particles, N * sizeof(float));

  cudaMalloc(&d_u, N * sizeof(float));
  cudaMalloc(&d_dudt, N * sizeof(float));
  cudaMalloc(&d_utprevious, N * sizeof(float));
  
  cudaMalloc(&d_dudt_pre, N * sizeof(float));

  cudaMalloc(&d_Nngb_previous, N * sizeof(float));
  
  //cudaMalloc(&d_ionFrac, N_ionFrac * sizeof(float));

  // Initialize x, y, z, etc on the Host.
  for (int i = 0; i < N; i++)
  {
    Typ[i] = Typvec[i];

    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];

    vx[i] = xvec[i]; //vxvec[i];
    vy[i] = yvec[i]; //vyvec[i];
    vz[i] = zvec[i]; //vzvec[i];

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

    float Nngb_f = 60.0f;

    if (Typ[i] == 0)
    {
      Nngb_previous[i] = Nngb_f;
    }
    else
    {
      Nngb_previous[i] = 0.0f;
    }
  }
  
  

  // Copy from Host to Device.
  cudaMemcpy(d_Typ, Typ, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_vx, vx, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy, vy, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vz, vz, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx, accx, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy, accy, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz, accz, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_mass, mass, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h, h, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_rho, rho, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_eps, eps, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, P, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csnd, csnd, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_divV, divV, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_curlV, curlV, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx_sph, accx_sph, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy_sph, accy_sph, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz_sph, accz_sph, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx_tot, accx_tot, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy_tot, accy_tot, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz_tot, accz_tot, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_abs_acc_g, abs_acc_g, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_abs_acc_tot, abs_acc_tot, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_sig, v_sig, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dh_dt, dh_dt, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dt_particles, dt_particles, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_u, u, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dudt, dudt, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_utprevious, utprevious, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_dudt_pre, dudt_pre, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_Nngb_previous, Nngb_previous, N * sizeof(float), cudaMemcpyHostToDevice);
  
  
  
  
  
  
  
  
  int GridSize; // Below we define GridSize more realistically
  
  float min_h;
  float binSize;
  
  float *xx = new float[N];
  float *yy = new float[N];
  float *zz = new float[N];
  
  // GPU related declarations
  float *d_xx, *d_yy, *d_zz;
  int *d_neighboringParticles;
  
  const int maxNeighbors = 500; // You'll need to determine a suitable value for this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
  // Allocate GPU memory
  checkCudaErrors(cudaMalloc((void**)&d_xx, N * sizeof(float)), "Allocate d_xx");
  checkCudaErrors(cudaMalloc((void**)&d_yy, N * sizeof(float)), "Allocate d_yy");
  checkCudaErrors(cudaMalloc((void**)&d_zz, N * sizeof(float)), "Allocate d_zz");
  checkCudaErrors(cudaMalloc((void**)&d_neighboringParticles, N * maxNeighbors * sizeof(int)), "Allocate d_neighboringParticles");


  

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  const float visc_alpha = 1.0f;

  float tEnd = 5.0f;
  float Nt = ceil(tEnd / dt) + 1;
 
  
  for (int i = 0; i < N; ++i) // This must be done in every loop!!
  {
      xx[i] = x[i];
      yy[i] = y[i];
      zz[i] = z[i];
  }
  
// Adjusting x, y, and z
  float minX = *min_element(xx, xx + N);
  float minY = *min_element(yy, yy + N);
  float minZ = *min_element(zz, zz + N);

  for (int i = 0; i < N; ++i)
  {
    xx[i] -= minX;
    yy[i] -= minY;
    zz[i] -= minZ;
  }
  
  float maxX = *max_element(xx, xx + N);
  float maxY = *max_element(yy, yy + N);
  float maxZ = *max_element(zz, zz + N);

  float max_dist = max({maxX, maxY, maxZ});

  cout << "max_dist: " << max_dist << endl;
  
  
  
  

  min_h = findNonZeroPositiveMin(h, N);
  binSize = 1.5f * min_h;
  
  cout << "min_h = " << min_h << endl;
  
  GridSize = static_cast<int>(max_dist/binSize) + 1;
  cout << "Current GridSize = " << GridSize << endl;
  
  GridCell* d_grid;
  checkCudaErrors(cudaMalloc((void**)&d_grid, GridSize * GridSize * GridSize * sizeof(GridCell)), "Allocate d_grid");
  checkCudaErrors(cudaMemset(d_grid, 0, GridSize * GridSize * GridSize * sizeof(GridCell)), "Memset d_grid");
  
  // Copy data to GPU
  checkCudaErrors(cudaMemcpy(d_xx, xx, N * sizeof(float), cudaMemcpyHostToDevice), "Copy xx to d_xx");
  checkCudaErrors(cudaMemcpy(d_yy, yy, N * sizeof(float), cudaMemcpyHostToDevice), "Copy yy to d_yy");
  checkCudaErrors(cudaMemcpy(d_zz, zz, N * sizeof(float), cudaMemcpyHostToDevice), "Copy zz to d_zz");
  
  // Launch the populateGrid kernel
  populateGrid<<<gridSize, blockSize>>>(d_xx, d_yy, d_zz, max_dist, GridSize, d_grid, N);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
  {
      printf("Error: %s\n", cudaGetErrorString(err));
  }

  cout << "Grid cells populated with particles indices." << endl;
  
  cout << "HERE !!!" << endl;

  //-----------------------------------------------
  //----------------- getDensity ------------------
  //-----------------------------------------------
  getDensity<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_mass,
                                      d_rho, d_h, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //---------------- getPressure ------------------
  //-----------------------------------------------
  getPressure_Adiabatic<<<gridSize, blockSize>>>(d_Typ, d_P, d_rho, d_u, gammah, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //----------------- getCsound -------------------
  //-----------------------------------------------
  getCsound_Adiabatic<<<gridSize, blockSize>>>(d_Typ, d_csnd, d_u, gammah, N);
  cudaDeviceSynchronize();
  
  //-----------------------------------------------
  //----------------- div_curlV -------------------
  //-----------------------------------------------
  auto T_divV = std::chrono::high_resolution_clock::now();
  div_curlVel<<<gridSize, blockSize>>>(d_Typ, d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                       d_rho, d_mass, d_h, N);
  cudaDeviceSynchronize();
  auto end_divV = std::chrono::high_resolution_clock::now();
  auto elapsed_divV = std::chrono::duration_cast<std::chrono::nanoseconds>(end_divV - T_divV);
  cout << "T_divV = " << elapsed_divV.count() * 1e-9 << endl;
  
  cout << endl;
  cudaMemcpy(divV, d_divV, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "divV = " << divV[i] << endl;
  }
  cout << endl;
  cudaMemcpy(curlV, d_curlV, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "curlV = " << curlV[i] << endl;
  }
  cout << endl;

  //-----------------------------------------------
  //------------------ acc_sph --------------------
  //-----------------------------------------------
  auto T_acc_sph = std::chrono::high_resolution_clock::now();
  acc_sph<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                   d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                   d_accz_sph, visc_alpha, N);
  cudaDeviceSynchronize();
  auto end_acc_sph = std::chrono::high_resolution_clock::now();
  auto elapsed_acc_sph = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_sph - T_acc_sph);
  cout << "T_acc_sph = " << elapsed_acc_sph.count() * 1e-9 << endl;
  
  
  
  cudaMemcpy(accx_sph, d_accx_sph, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "accx_sph After = " << accx_sph[i] << endl;
  }
  
  
  //-----------------------------------------------
  //----------------- div_curlV_Grid -------------------
  //-----------------------------------------------
  auto T_divV2_grid = std::chrono::high_resolution_clock::now();
  div_curlVel_grid<<<gridSize, blockSize>>>(d_Typ, d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_rho, d_mass, 
                                            d_xx, d_yy, d_zz, max_dist, GridSize, d_grid,
                                            d_neighboringParticles, maxNeighbors,
                                            d_h, N);
  cudaDeviceSynchronize();
  auto end_divV2_grid = std::chrono::high_resolution_clock::now();
  auto elapsed_divV2_grid = std::chrono::duration_cast<std::chrono::nanoseconds>(end_divV2_grid - T_divV2_grid);
  cout << "T_divV2_grid = " << elapsed_divV2_grid.count() * 1e-9 << endl;
  
  cout << endl;
  cudaMemcpy(divV, d_divV, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "divV Grid = " << divV[i] << endl;
  }
  cout << endl;
  cudaMemcpy(curlV, d_curlV, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "curlV Grid = " << curlV[i] << endl;
  }
  cout << endl;
  
  //-----------------------------------------------
  //------------------ acc_sph_grid --------------------
  //-----------------------------------------------
  auto T_acc_sph_grid = std::chrono::high_resolution_clock::now();
  acc_sph_grid<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                        d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph, d_accz_sph,
                                        d_xx, d_yy, d_zz, max_dist, GridSize, d_grid,
                                        d_neighboringParticles, maxNeighbors,
                                        visc_alpha, N);
  cudaDeviceSynchronize();
  auto end_acc_sph_grid = std::chrono::high_resolution_clock::now();
  auto elapsed_acc_sph_grid = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_sph_grid - T_acc_sph_grid);
  cout << "T_acc_sph_grid = " << elapsed_acc_sph_grid.count() * 1e-9 << endl;


  cudaMemcpy(accx_sph, d_accx_sph, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "accx_sph Grid = " << accx_sph[i] << endl;
  }
  


  //-----------------------------------------------
  //------------------- du_dt ---------------------
  //-----------------------------------------------
  auto T_du_dt = std::chrono::high_resolution_clock::now();
  get_dU<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                  d_divV, d_curlV, d_mass, d_P, d_dudt,
                                  visc_alpha, N);
  cudaDeviceSynchronize();
  auto end_du_dt = std::chrono::high_resolution_clock::now();
  auto elapsed_du_dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end_du_dt - T_du_dt);
  cout << "T_du_dt = " << elapsed_du_dt.count() * 1e-9 << endl;
  
  cout << endl;
  cudaMemcpy(dudt, d_dudt, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "dudt = " << dudt[i] << endl;
  }
  
  
  cudaMemset(d_dudt, 0.0f, N * sizeof(float));
  cout << endl;
  cudaMemcpy(dudt, d_dudt, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "dudt After Resseting to 0.0 = " << dudt[i] << endl;
  }
  cout << endl;
  
  //-----------------------------------------------
  //------------------- du_dt_grid ---------------------
  //-----------------------------------------------
  auto T_du_dt2 = std::chrono::high_resolution_clock::now();
  get_dU_grid<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                       d_divV, d_curlV, d_mass, d_P, d_dudt,
                                       d_xx, d_yy, d_zz, max_dist, GridSize, d_grid,
                                       d_neighboringParticles, maxNeighbors,
                                       visc_alpha, N);
  cudaDeviceSynchronize();
  auto end_du_dt2 = std::chrono::high_resolution_clock::now();
  auto elapsed_du_dt2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_du_dt2 - T_du_dt2);
  cout << "T_du_dt Grid = " << elapsed_du_dt2.count() * 1e-9 << endl;

  cout << endl;
  cudaMemcpy(dudt, d_dudt, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "dudt Grid = " << dudt[i] << endl;
  }


}
