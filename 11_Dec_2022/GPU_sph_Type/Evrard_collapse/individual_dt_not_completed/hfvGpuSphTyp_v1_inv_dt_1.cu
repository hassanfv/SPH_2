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
#include "hfvCppLibs_indiv_dt.h"

using namespace std;

int main()
{

  float dt = 1e-4; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  float dt_min_j;

  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.001f; // used for smoothing length.
  const float G = 1.0f;
  const float C_CFL = 0.25;

  int N = 1472;
  std::string filename = "Evrard_GPU_IC_1k.bin";
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

  float gammah = 5.0f / 3.0f;

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

  Nngb_previous = new float[N];

  float *t_last, *t_next, *d_t_last, *d_t_next;
  int *activeId, *d_activeId;
  float *accx_prev, *accy_prev, *accz_prev, *d_accx_prev, *d_accy_prev, *d_accz_prev;

  t_last = new float[N];
  t_next = new float[N];
  activeId = new int[N];

  accx_prev = new float[N];
  accy_prev = new float[N];
  accz_prev = new float[N];

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

  cudaMalloc(&d_Nngb_previous, N * sizeof(float));

  cudaMalloc(&d_t_last, N * sizeof(float));
  cudaMalloc(&d_t_next, N * sizeof(float));

  cudaMalloc(&d_activeId, N * sizeof(int));

  cudaMalloc(&d_accx_prev, N * sizeof(float));
  cudaMalloc(&d_accy_prev, N * sizeof(float));
  cudaMalloc(&d_accz_prev, N * sizeof(float));

  // Initialize x, y, and z on the Host.
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

    dt_particles[i] = 0.0f;

    Nngb_previous[i] = Nngb_f;

    t_last[i] = 0.0f;
    t_next[i] = 0.0f;

    activeId[i] = 1; // all particles are first assumed active but then will be correctly modified!

    accx_prev[i] = 0.0f;
    accy_prev[i] = 0.0f;
    accz_prev[i] = 0.0f;
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

  cudaMemcpy(d_Nngb_previous, Nngb_previous, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_t_last, t_last, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_next, t_next, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_activeId, activeId, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx_prev, accx_prev, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy_prev, accy_prev, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz_prev, accz_prev, N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  const float visc_alpha = 1.0f;

  float t;

  t = 0.0f;

  float tEnd = 5.0f;
  float Nt = ceil(tEnd / dt) + 1;

  //******************************************************
  //************* Getting Time-step dt ******************
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

  create_dt_blocks<<<gridSize, blockSize>>>(d_Typ, d_dt_particles, N);
  cudaDeviceSynchronize();

  cudaMemcpy(dt_particles, d_dt_particles, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(Typ, d_Typ, N * sizeof(int), cudaMemcpyDeviceToHost);

  // t_last[nxActive] = t;
  // t_next[nxActive] = t_last[nxActive] + dt_i[nxActive];
  // Only update for active particles! But in the first run we set all particles as active and then
  // below it will update the active particle list correctly. Setting all to active in this step is
  // not only safe but necessary !!
  // update_t_last_next is only executed once in the begining of the run. In later time-steps, t_last
  // and t_next will be updated inside the acc. computation function!!
  update_t_last_next<<<gridSize, blockSize>>>(d_Typ, d_t_last, d_t_next, d_activeId, d_dt_particles, t, N);
  cudaDeviceSynchronize();

  // NOTE: t_last at the start of the run is 0.0 for all particles, and t_next
  // is equal to the dt_particle of each particle after block creation.
  cudaMemcpy(t_next, d_t_next, N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(activeId, d_activeId, N * sizeof(int), cudaMemcpyDeviceToHost);
  dt_min_j = dt_min_finder(Typ, activeId, t_next, t, N); // dt_min_finder is a HOST function here!

  // active particles are those with t_next - t = dt_min_j (i.e. min(t_next - t))
  // For the first run, we set all particles to active.
  who_is_active<<<gridSize, blockSize>>>(d_Typ, d_activeId, d_t_next, dt_min_j, t, N);
  cudaDeviceSynchronize();

  dt = dt_min_j;

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
  acc_g_block<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                       d_mass, d_activeId, G, N);
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
  acc_g_sphB<<<gridSize, blockSize>>>(d_Typ, d_accx_tot, d_accy_tot, d_accz_tot,
                                      d_accx, d_accy, d_accz,
                                      d_accx_sph, d_accy_sph, d_accz_sph,
                                      d_accx_prev, d_accy_prev, d_accz_prev,
                                      activeId, N);
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

  //-----------------------------------------------
  //------------- velocity evolution -------------- t_last should be used here for delta_t_j
  //-----------------------------------------------
  v_evolveB<<<gridSize, blockSize>>>(d_Typ, d_vx, d_vy, d_vz,
                                     d_accx_tot, d_accy_tot, d_accz_tot,
                                     d_accx_prev, d_accy_prev, d_accz_prev,
                                     d_activeId, d_t_last, t, dt, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------- position evolution -------------- t_last should be used here for delta_t_j
  //-----------------------------------------------
  r_evolveB<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                     d_accx_tot, d_accy_tot, d_accz_tot,
                                     d_accx_prev, d_accy_prev, d_accz_prev,
                                     d_activeId, d_t_last, t, dt, N);
  cudaDeviceSynchronize();

  t += dt;

  //
  //
  //
  // **************************************************************
  // *********************** MAIN LOOP ****************************
  // **************************************************************

  int counter = 0; // This is used to save fewer output files, e.g. 1 snap-shot per 2 time-step!

  while (t < tEnd)
  {
    auto begin = std::chrono::high_resolution_clock::now();

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

    create_dt_blocks<<<gridSize, blockSize>>>(d_Typ, d_dt_particles, N);
    cudaDeviceSynchronize();

    cudaMemcpy(dt_particles, d_dt_particles, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Typ, d_Typ, N * sizeof(int), cudaMemcpyDeviceToHost);

    // t_last[nxActive] = t;
    // t_next[nxActive] = t_last[nxActive] + dt_i[nxActive];
    // Only update for active particles!
    update_t_last_next<<<gridSize, blockSize>>>(d_Typ, d_t_last, d_t_next, d_activeId, d_dt_particles, t, N);
    cudaDeviceSynchronize();

    // NOTE: t_last at the start of the run is 0.0 for all particles, and t_next
    // is equal to the dt_particle of each particle after block creation.
    cudaMemcpy(t_next, d_t_next, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(activeId, d_activeId, N * sizeof(int), cudaMemcpyDeviceToHost);
    dt_min_j = dt_min_finder(Typ, activeId, t_next, t, N); // dt_min_finder is a HOST function here!

    // active particles are those with t_next - t = dt_min_j (i.e. min(t_next - t))
    // For the first run, we set all particles to active.
    who_is_active<<<gridSize, blockSize>>>(d_Typ, d_activeId, d_t_next, dt_min_j, t, N);
    cudaDeviceSynchronize();

    dt = dt_min_j;

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
    acc_g_block<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                         d_mass, d_activeId, G, N);
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
    acc_g_sphB<<<gridSize, blockSize>>>(d_Typ, d_accx_tot, d_accy_tot, d_accz_tot,
                                        d_accx, d_accy, d_accz,
                                        d_accx_sph, d_accy_sph, d_accz_sph,
                                        d_accx_prev, d_accy_prev, d_accz_prev,
                                        d_activeId, N);
    cudaDeviceSynchronize();
    auto end_acc_tot = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_tot = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_tot - T_acc_tot);
    cout << "T_acc_tot = " << elapsed_acc_tot.count() * 1e-9 << endl;

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

    //****************** velocity evolution *******************
    v_evolveB<<<gridSize, blockSize>>>(d_Typ, d_vx, d_vy, d_vz,
                                       d_accx_tot, d_accy_tot, d_accz_tot,
                                       d_accx_prev, d_accy_prev, d_accz_prev,
                                       d_activeId, d_t_last, t, dt, N);
    cudaDeviceSynchronize();

    //****************** position evolution *******************
    r_evolveB<<<gridSize, blockSize>>>(d_Typ, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                       d_accx_tot, d_accy_tot, d_accz_tot,
                                       d_accx_prev, d_accy_prev, d_accz_prev,
                                       d_activeId, d_t_last, t, dt, N);
    cudaDeviceSynchronize();

    //-------------------------------------------------

    t += dt;

    // t_last[nxActive] = t;
    // t_next[nxActive] = t_last[nxActive] + dt_i[nxActive];
    // Only update for active particles!!
    // update_t_last_next<<<gridSize, blockSize>>>(d_Typ, d_t_last, d_t_next, d_activeId, d_dt_particles, t, N);
    // cudaDeviceSynchronize();

    cudaMemcpy(rho, d_rho, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
    {
      cout << "AAA = " << rho[i] << endl;
    }

    //------------ SAVING SNAP-SHOTS ------------
    if (!(counter % 20))
    {
      cudaMemcpy(Typ, d_Typ, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(x, d_x, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(vx, d_vx, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vz, d_vz, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(rho, d_rho, N * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(u, d_u, N * sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(mass, d_mass, N * sizeof(float), cudaMemcpyDeviceToHost);

      // Specify the output file name
      std::string filename = "./Outputs/G-" + to_string(t * 1) + ".bin";
      // Save the arrays to binary format
      saveArraysToBinary(filename, x, y, z, vx, vy, vz, rho, h, u, mass, Typ, N);
    }

    auto T_NG = std::chrono::high_resolution_clock::now();
    // cudaMemcpy(d_NGroupz, NGroupz, NG * sizeof(int), cudaMemcpyHostToDevice);
    auto end_NG = std::chrono::high_resolution_clock::now();
    auto elapsed_NG = std::chrono::duration_cast<std::chrono::nanoseconds>(end_NG - T_NG);
    cout << "T_NG = " << elapsed_NG.count() * 1e-9 << endl;

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << "Elapsed time = " << elapsed.count() * 1e-9 << endl;
    cout << endl;

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
