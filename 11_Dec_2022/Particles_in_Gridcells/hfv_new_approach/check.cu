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
#include "hfvCLibs_check.h"
#include <cstdlib> // This is ONLY used for the "exit(0)" function !!

// Added the isothermal gravitational field acceleration. (24 May 2023).
// Added the reading of the params.txt file and updated the IC reading file section and function. (22 May 2023).

using namespace std;

int main()
{

  float dt = 4e-7; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.005f; // used for smoothing length.
  
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
  
  //cudaMemcpy(d_ionFrac, ionFrac, N_ionFrac * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  //const float visc_alpha = 1.0f;

  //float t;

  //t = 0.0f;

  //float tEnd = 5.0f;
  //float Nt = ceil(tEnd / dt) + 1;

  //float Zmetal = 0.1; // ==> [Z/H] = -1.
  
  
  int MAX_ngb = 100;
  
  int MAX_N = N * MAX_ngb;
  
  int *ngb, *d_ngb;
  
  ngb = new int[MAX_N];
  
  cudaMalloc(&d_ngb, MAX_N * sizeof(int));
  
  for (int i = 0; i < MAX_N; i++)
  {
    ngb[i] = -1;
  }

  cudaMemcpy(d_ngb, ngb, MAX_N * sizeof(int), cudaMemcpyHostToDevice);


  //-----------------------------------------------
  //-------------- Smoothing Length ---------------
  //-----------------------------------------------
  auto T_h = std::chrono::high_resolution_clock::now();
  smoothing_h<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_h,
                                       N, Ndown, Nup, coeff,
                                       Nngb_f, d_Nngb_previous, d_divV, dt);
  cudaDeviceSynchronize();
  auto end_h = std::chrono::high_resolution_clock::now();
  auto elapsed_h = std::chrono::duration_cast<std::chrono::nanoseconds>(end_h - T_h);
  cout << "T_h = " << elapsed_h.count() * 1e-9 << endl;


  auto T_ngb = std::chrono::high_resolution_clock::now();
  ngbDB<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_h, d_ngb, MAX_ngb, N);
  cudaDeviceSynchronize();
  auto end_ngb = std::chrono::high_resolution_clock::now();
  auto elapsed_ngb = std::chrono::duration_cast<std::chrono::nanoseconds>(end_ngb - T_ngb);
  cout << "T_ngb = " << elapsed_ngb.count() * 1e-9 << endl;


  cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "h B = " << h[i] << endl;
  }
  cout << endl;

  auto T_h2 = std::chrono::high_resolution_clock::now();
  smoothing_h_ngb<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_h,
                                           N, Ndown, Nup, coeff,
                                           Nngb_f, d_Nngb_previous, d_divV, d_ngb, MAX_ngb, dt);
  cudaDeviceSynchronize();
  auto end_h2 = std::chrono::high_resolution_clock::now();
  auto elapsed_h2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_h2 - T_h2);
  cout << "T_h2 = " << elapsed_h2.count() * 1e-9 << endl;


  
  cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
  {
    cout << "h A = " << h[i] << endl;
  }
  cout << endl;

 
  cudaMemcpy(ngb, d_ngb, MAX_N * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 10; i++)
  {
    cout << "ngb = " << ngb[i] << endl;
  }
  
  cout << endl;
  
  for (int i = MAX_N - 10; i < MAX_N+2; i++)
  {
    cout << "ngb = " << ngb[i] << endl;
  }

  cout << endl;
  
  /*
  for (int i = 0; i < MAX_N; i++)
  {
    if (ngb[i] == -1)
    {
    cout << "i = " << i << endl;
    cout << "ngb = " << ngb[i] << endl;
    exit(0);
    }
  }
  */
  
  cudaMemcpy(Typ, d_Typ, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(ngb, d_ngb, MAX_N * sizeof(int), cudaMemcpyDeviceToHost);

  std::ofstream outFile("data.bin", std::ios::binary);
  
  outFile.write(reinterpret_cast<char*>(&N), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&MAX_N), sizeof(int));
  outFile.write(reinterpret_cast<char*>(&MAX_ngb), sizeof(int));
  outFile.write(reinterpret_cast<char*>(Typ), N * sizeof(int));
  outFile.write(reinterpret_cast<char*>(x), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(y), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(z), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(h), N * sizeof(float));
  outFile.write(reinterpret_cast<char*>(ngb), MAX_N * sizeof(int));

  outFile.close();



}



