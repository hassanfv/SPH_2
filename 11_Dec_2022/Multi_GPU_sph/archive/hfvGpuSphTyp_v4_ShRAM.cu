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
#include "hfvCppLibs_v4.h"

// Shared Memory (i.e. cudaMallocManaged) is used. (25 May 2023).
// Added the isothermal gravitational field acceleration. (24 May 2023).
// Added the reading of the params.txt file and updated the IC reading file section and function. (22 May 2023).

using namespace std;

int main()
{

  float dt = 1e-7; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.001f; // used for smoothing length.

  // Reading the params.txt file
  std::string filename;
  int N;
  float G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma;

  readParams(filename, N, G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma);

  std::cout << "filename: " << filename << "\n";
  std::cout << "N: " << N << "\n";
  std::cout << "G: " << G << "\n";
  std::cout << "L_AGN_code_unit: " << L_AGN_code_unit << "\n";
  std::cout << "M_dot_in_code_unit: " << M_dot_in << "\n";
  std::cout << "vin_in_code_unit: " << v_in << "\n";
  std::cout << "u_for_10K_Temp: " << u_for_10K_Temp << "\n";
  std::cout << "m_sph_high_res: " << m_sph_high_res << "\n";
  std::cout << "sigma: " << sigma << "\n";

  // Reading the IC file
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
  int *d_Typ;
  float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz;
  float *d_mass, *d_h, *d_rho;
  float *d_accx, *d_accy, *d_accz, *d_eps;
  float *d_P, *d_csnd, *d_divV, *d_curlV;
  float *d_accx_sph, *d_accy_sph, *d_accz_sph;
  float *d_accx_tot, *d_accy_tot, *d_accz_tot;
  float *d_abs_acc_g, *d_abs_acc_tot, *d_v_sig, *d_dh_dt;
  float *d_u, *d_dudt, *d_utprevious;
  float *d_Nngb_previous; // Note that both are floats and not int! check smoothing func. to see why!
  float *d_dt_particles;

  float gammah = 5.0f / 3.0f;

  cudaMallocManaged(&d_Typ, N * sizeof(int));

  cudaMallocManaged(&d_x, N * sizeof(float));
  cudaMallocManaged(&d_y, N * sizeof(float));
  cudaMallocManaged(&d_z, N * sizeof(float));

  cudaMallocManaged(&d_vx, N * sizeof(float));
  cudaMallocManaged(&d_vy, N * sizeof(float));
  cudaMallocManaged(&d_vz, N * sizeof(float));

  cudaMallocManaged(&d_accx, N * sizeof(float));
  cudaMallocManaged(&d_accy, N * sizeof(float));
  cudaMallocManaged(&d_accz, N * sizeof(float));

  cudaMallocManaged(&d_mass, N * sizeof(float));
  cudaMallocManaged(&d_h, N * sizeof(float));
  cudaMallocManaged(&d_rho, N * sizeof(float));
  cudaMallocManaged(&d_eps, N * sizeof(float));
  cudaMallocManaged(&d_P, N * sizeof(float));
  cudaMallocManaged(&d_csnd, N * sizeof(float));

  cudaMallocManaged(&d_divV, N * sizeof(float));
  cudaMallocManaged(&d_curlV, N * sizeof(float));

  cudaMallocManaged(&d_accx_sph, N * sizeof(float));
  cudaMallocManaged(&d_accy_sph, N * sizeof(float));
  cudaMallocManaged(&d_accz_sph, N * sizeof(float));

  cudaMallocManaged(&d_accx_tot, N * sizeof(float));
  cudaMallocManaged(&d_accy_tot, N * sizeof(float));
  cudaMallocManaged(&d_accz_tot, N * sizeof(float));

  cudaMallocManaged(&d_abs_acc_g, N * sizeof(float));
  cudaMallocManaged(&d_abs_acc_tot, N * sizeof(float));
  cudaMallocManaged(&d_v_sig, N * sizeof(float));
  cudaMallocManaged(&d_dh_dt, N * sizeof(float));
  cudaMallocManaged(&d_dt_particles, N * sizeof(float));

  cudaMallocManaged(&d_u, N * sizeof(float));
  cudaMallocManaged(&d_dudt, N * sizeof(float));
  cudaMallocManaged(&d_utprevious, N * sizeof(float));

  cudaMallocManaged(&d_Nngb_previous, N * sizeof(float));

  // Initialize x, y, and z on the Host.
  for (int i = 0; i < N; i++)
  {
    d_Typ[i] = Typvec[i];

    d_x[i] = xvec[i];
    d_y[i] = yvec[i];
    d_z[i] = zvec[i];

    d_vx[i] = vxvec[i];
    d_vy[i] = vyvec[i];
    d_vz[i] = vzvec[i];

    d_mass[i] = massvec[i];
    d_eps[i] = epsvec[i];

    d_accx[i] = 0.0f;
    d_accy[i] = 0.0f;
    d_accz[i] = 0.0f;

    d_accx_tot[i] = 0.0f;
    d_accy_tot[i] = 0.0f;
    d_accz_tot[i] = 0.0f;

    d_abs_acc_g[i] = 0.0f;
    d_abs_acc_tot[i] = 0.0f;
    d_v_sig[i] = 0.0f;

    d_h[i] = hvec[i]; // place holder.
    d_rho[i] = 0.0f;  // place holder.
    d_P[i] = 0.0f;    // placeholder.
    d_csnd[i] = 0.0f; // placeholder.

    d_divV[i] = 0.0f;  // placeholder.
    d_curlV[i] = 0.0f; // placeholder.

    d_accx_sph[i] = 0.0f;
    d_accy_sph[i] = 0.0f;
    d_accz_sph[i] = 0.0f;

    d_dh_dt[i] = 0.0f;

    d_u[i] = uvec[i];
    d_dudt[i] = 0.0f;
    d_utprevious[i] = 0.0f;

    d_dt_particles[i] = 0.0f;

    if (d_Typ[i] == 0)
    {
      d_Nngb_previous[i] = Nngb_f;
    }
    else
    {
      d_Nngb_previous[i] = 0.0f;
    }
  }

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  const float visc_alpha = 1.0f;

  float t;

  t = 0.0f;

  float tEnd = 5.0f;
  float Nt = ceil(tEnd / dt) + 1;

  //-----------------------------------------------
  //-------------- Smoothing Length ---------------
  //-----------------------------------------------
  smoothing_h<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_h,
                                       N, Ndown, Nup, coeff,
                                       Nngb_f, d_Nngb_previous, d_divV, dt);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //----------------- getDensity ------------------
  //-----------------------------------------------
  getDensity<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_mass,
                                      d_rho, d_h, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ getAcc_g -------------------
  //-----------------------------------------------
  acc_g<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                 d_mass, G, N);
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
  div_curlVel<<<gridSize, blockSize>>>(d_Typ, d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                       d_rho, d_mass, d_h, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ acc_sph --------------------
  //-----------------------------------------------
  acc_sph<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                   d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                   d_accz_sph, visc_alpha, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ acc_tot --------------------
  //-----------------------------------------------
  acc_g_sph<<<gridSize, blockSize>>>(d_Typ, d_accx_tot, d_accy_tot, d_accz_tot,
                                     d_accx, d_accy, d_accz,
                                     d_accx_sph, d_accy_sph, d_accz_sph,
                                     N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------- du_dt ---------------------
  //-----------------------------------------------

  get_dU<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                  d_divV, d_curlV, d_mass, d_P, d_dudt,
                                  visc_alpha, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //---------------- u evolution ------------------
  //-----------------------------------------------

  u_updater<<<gridSize, blockSize>>>(d_Typ, d_u, d_dudt, d_utprevious, dt, N);
  cudaDeviceSynchronize();

  const float C_CFL = 0.25;

  float h_min, h_max, h_mean;

  float *d_leftover_mass;
  cudaMallocManaged((void **)&d_leftover_mass, sizeof(float));
  // Set the value of the allocated memory to 0.0f.
  cudaMemset(d_leftover_mass, 0, sizeof(float));
  cudaDeviceSynchronize();

  // **************************************************************
  // *********************** MAIN LOOP ****************************
  // **************************************************************

  int counter = 0; // This is used to save fewer output files, e.g. 1 snap-shot per 2 time-step!

  while (t < tEnd)
  {

    auto begin = std::chrono::high_resolution_clock::now();

    //****************** velocity evolution *******************
    v_evolve<<<gridSize, blockSize>>>(d_Typ, d_vx, d_vy, d_vz, d_accx_tot, d_accy_tot,
                                      d_accz_tot, dt, N);
    cudaDeviceSynchronize();

    //****************** position evolution (BH fixed at [0, 0, 0]) *******************

    r_evolve<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, N);
    cudaDeviceSynchronize();

    //****************** Smoothing Length *********************

    smoothing_h<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_h,
                                         N, Ndown, Nup, coeff,
                                         Nngb_f, d_Nngb_previous, d_divV, dt);
    cudaDeviceSynchronize();

    //****************** Set eps of Gas equal to h ******************

    set_eps_of_gas_to_h<<<gridSize, blockSize>>>(d_Typ, d_eps, d_h, N);
    cudaDeviceSynchronize();

    //****************** getDensity ***********************
    getDensity<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_mass,
                                        d_rho, d_h, N);
    cudaDeviceSynchronize();

    //****************** getAcc_gX *************************
    auto T_acc_g = std::chrono::high_resolution_clock::now();
    acc_g<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                   d_mass, G, N);
    cudaDeviceSynchronize();
    auto end_acc_g = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_g = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_g - T_acc_g);
    cout << "T_acc_g = " << elapsed_acc_g.count() * 1e-9 << endl;

    //****************** getPressure **********************
    getPressure_Adiabatic<<<gridSize, blockSize>>>(d_Typ, d_P, d_rho, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** getCsound ************************
    getCsound_Adiabatic<<<gridSize, blockSize>>>(d_Typ, d_csnd, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** div_curlVX ************************
    auto T_divCurl = std::chrono::high_resolution_clock::now();
    div_curlVel<<<gridSize, blockSize>>>(d_Typ, d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                         d_rho, d_mass, d_h, N);
    cudaDeviceSynchronize();
    auto end_divCurl = std::chrono::high_resolution_clock::now();
    auto elapsed_divCurl = std::chrono::duration_cast<std::chrono::nanoseconds>(end_divCurl - T_divCurl);
    cout << "T_divCurl = " << elapsed_divCurl.count() * 1e-9 << endl;

    //****************** acc_sphX **************************
    auto T_acc_sph = std::chrono::high_resolution_clock::now();
    acc_sph<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                     d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                     d_accz_sph, visc_alpha, N);
    cudaDeviceSynchronize();
    auto end_acc_sph = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_sph = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_sph - T_acc_sph);
    cout << "T_acc_sph = " << elapsed_acc_sph.count() * 1e-9 << endl;

    //****************** acc_tot **************************
    auto T_acc_tot = std::chrono::high_resolution_clock::now();
    acc_g_sph<<<gridSize, blockSize>>>(d_Typ, d_accx_tot, d_accy_tot, d_accz_tot,
                                       d_accx, d_accy, d_accz,
                                       d_accx_sph, d_accy_sph, d_accz_sph,
                                       N);
    cudaDeviceSynchronize();
    auto end_acc_tot = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_tot = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_tot - T_acc_tot);
    cout << "T_acc_tot = " << elapsed_acc_tot.count() * 1e-9 << endl;

    //******* Isothermal Gravity (Richings et al - 2018) ********
    galaxy_isothermal_potential<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_accx_tot,
                                                         d_accy_tot, d_accz_tot, sigma, G, N);
    cudaDeviceSynchronize();

    //****************** velocity evolution *******************
    v_evolve<<<gridSize, blockSize>>>(d_Typ, d_vx, d_vy, d_vz, d_accx_tot, d_accy_tot,
                                      d_accz_tot, dt, N);
    cudaDeviceSynchronize();

    //******************** get_dUX (du_dt) *********************
    auto T_dU = std::chrono::high_resolution_clock::now();
    get_dU<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                    d_divV, d_curlV, d_mass, d_P, d_dudt,
                                    visc_alpha, N);
    cudaDeviceSynchronize();
    auto end_dU = std::chrono::high_resolution_clock::now();
    auto elapsed_dU = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dU - T_dU);
    cout << "T_dU = " << elapsed_dU.count() * 1e-9 << endl;

    //******************** u evolution *********************
    u_updater<<<gridSize, blockSize>>>(d_Typ, d_u, d_dudt, d_utprevious, dt, N);
    cudaDeviceSynchronize();

    //-------------------------------------------------

    for (int i = 0; i < 5; i++)
    {
      cout << "AAA = " << d_rho[i] << endl;
    }

    //------------ SAVING SNAP-SHOTS ------------
    if (!(counter % 200))
    {

      // Specify the output file name
      std::string filename = "./Outputs/G-" + to_string(t * 1) + ".bin";
      // Save the arrays to binary format
      saveArraysToBinary(filename, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_rho, d_h, d_u, d_mass, d_Typ, N);
    }

    auto T_NG = std::chrono::high_resolution_clock::now();
    auto end_NG = std::chrono::high_resolution_clock::now();
    auto elapsed_NG = std::chrono::duration_cast<std::chrono::nanoseconds>(end_NG - T_NG);
    cout << "T_NG = " << elapsed_NG.count() * 1e-9 << endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << "Elapsed time = " << elapsed.count() * 1e-9 << endl;
    cout << endl;

    //******************************************************
    //************* Updating Time-step dt ******************
    //******************************************************

    dt_array_indiv_dt<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z,
                                               d_vx, d_vy, d_vz,
                                               d_accx, d_accy, d_accz,
                                               d_accx_tot, d_accy_tot, d_accz_tot,
                                               d_h, d_csnd, d_dt_particles,
                                               d_abs_acc_g, d_abs_acc_tot,
                                               d_divV, d_dh_dt, C_CFL,
                                               visc_alpha, d_eps, N);
    cudaDeviceSynchronize();

    t += dt;

    // dt = min_finder(Typ, dt_particles, N);

    //***********************************************************
    //*************** Outflow particle injection ****************
    //***********************************************************

    // Generate a seed using the high resolution clock
    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    unsigned long long seed = static_cast<unsigned long long>(nanos);
    //------------

    h_min = min_finder(d_Typ, d_h, N);
    h_max = max_finder(d_Typ, d_h, N);
    h_mean = 0.5f * (h_min + h_max);

    outflow_injector<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z,
                                              d_vx, d_vy, d_vz,
                                              d_h, d_eps, d_mass,
                                              Nngb_f, d_Nngb_previous,
                                              d_u, M_dot_in, v_in,
                                              m_sph_high_res, u_for_10K_Temp,
                                              h_mean, d_leftover_mass, dt, N,
                                              seed);
    cudaDeviceSynchronize();

    if (!(counter % 1))
    {
      cout << "Adopted dt = " << dt << endl;
      cout << "current t = " << t << endl;
      cout << "*****************************" << endl;
      cout << endl;
    }

    counter++;
  }

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
