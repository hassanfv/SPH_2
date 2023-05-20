//% % writefile test.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include "myCppSPHLibs_v7_t2.h"
using namespace std;

// smoothing func. updated + my_pi is replaced by M_PI. (28 Jan - 2023)
// In this version, we implemented the Restart possibility! (24 Jan - 2023)
// In this version, we use CLOUDY cooling & heating!
// In this version, we also include cooling.
// In this version, the output file also contains the velocity components.

string RESTART = "no"; // options = yes or no (note they are in lower case letters)!!!!!!!!!!!!!!!!!

float max_finder(float *arr, int N)
{

  float max_val = 0.0;
  for (int i = 0; i < N; i++)
  {
    if (arr[i] >= max_val)
    {
      max_val = arr[i];
    }
  }
  return max_val;
}

float min_finder(float *arr, int N)
{

  float min_val = arr[0];
  for (int i = 0; i < N; i++)
  {
    if (arr[i] <= min_val)
    {
      min_val = arr[i];
    }
  }
  return min_val;
}

int main()
{

  float dt = 1e-3; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.001f; // used for smoothing length.

  // Reading params.GPU file.
  /*
  ifstream infile;
  infile.open("params.GPU");

  int N;
  float c_0, gammah, Rcld_in_pc, Rcld_in_cm, Mcld_in_g, muu, Mach;
  float grav_const_in_cgs, G;

  if (infile.is_open())
  {
    infile >> N;
    infile >> c_0;
    infile >> gammah;
    infile >> Rcld_in_pc;
    infile >> Rcld_in_cm;
    infile >> Mcld_in_g;
    infile >> muu;
    infile >> Mach;
    infile >> grav_const_in_cgs;
    infile >> G;
    infile.close();
  }
  else
  {
    cout << "params.GPU File Not Found !!!" << endl;
  }

  long double UnitRadius_in_cm = Rcld_in_cm;
  long double UnitRadius_in_cm_2 = UnitRadius_in_cm * UnitRadius_in_cm;

  long double UnitMass_in_g = Mcld_in_g;
  //-------------------------
  long double UnitDensity_in_cgsT = UnitMass_in_g / pow(UnitRadius_in_cm, 3);
  //-------------------------
  long double Unit_u_in_cgsT = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm;
  long double Unit_P_in_cgsT = UnitDensity_in_cgsT * Unit_u_in_cgsT;
  long double unitVelocityT = sqrt(grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm);
  long double unitTime_in_sT = sqrt(pow(UnitRadius_in_cm, 3) / grav_const_in_cgs / UnitMass_in_g);

  float UnitDensity_in_cgs = (float)UnitDensity_in_cgsT;
  float Unit_u_in_cgs = (float)Unit_u_in_cgsT;
  float Unit_P_in_cgs = (float)Unit_P_in_cgsT;
  float unitVelocity = (float)unitVelocityT;
  float unitTime_in_s = (float)unitTime_in_sT;

  cout << "UnitDensity_in_cgs = " << UnitDensity_in_cgs << endl;
  cout << "Unit_u_in_cgs = " << Unit_u_in_cgs << endl;
  cout << "Unit_P_in_cgs = " << Unit_P_in_cgs << endl;
  cout << "unitVelocity = " << unitVelocity << endl;
  cout << "unitTime_in_s = " << unitTime_in_s << endl;
  cout << endl;
  */

  //***************************************
  //********** Reading IC file ************
  //***************************************

  int N = 33552;
  std::vector<float> xvec(N);
  std::vector<float> yvec(N);
  std::vector<float> zvec(N);
  std::vector<float> vxvec(N);
  std::vector<float> vyvec(N);
  std::vector<float> vzvec(N);
  std::vector<float> massvec(N);
  std::vector<float> hpreviousvec(N);
  std::vector<float> epsvec(N);
  std::vector<float> uvec(N);

  // Check if the binary file exists
  std::ifstream file("Evrard_GPU_IC_33k.bin", std::ios::binary);
  if (!file) {
    std::cout << "Could not open the binary file." << std::endl;
  }
  file.close();
  
  // Open the binary file for reading
  file.open("Evrard_GPU_IC_33k.bin", std::ios::binary);
  if (file) {
    // Read the first array
    file.read(reinterpret_cast<char*>(xvec.data()), sizeof(float) * xvec.size());
    file.read(reinterpret_cast<char*>(yvec.data()), sizeof(float) * yvec.size());
    file.read(reinterpret_cast<char*>(zvec.data()), sizeof(float) * zvec.size());
    
    file.read(reinterpret_cast<char*>(vxvec.data()), sizeof(float) * vxvec.size());
    file.read(reinterpret_cast<char*>(vyvec.data()), sizeof(float) * vyvec.size());
    file.read(reinterpret_cast<char*>(vzvec.data()), sizeof(float) * vzvec.size());
    
    file.read(reinterpret_cast<char*>(massvec.data()), sizeof(float) * massvec.size());
    file.read(reinterpret_cast<char*>(hpreviousvec.data()), sizeof(float) * hpreviousvec.size());
    file.read(reinterpret_cast<char*>(epsvec.data()), sizeof(float) * epsvec.size());
    file.read(reinterpret_cast<char*>(uvec.data()), sizeof(float) * uvec.size());

    // Close the file
    file.close();

  } else {
    std::cout << "Failed to open the file." << std::endl;
  }

  float ttime;

  //***************************************
  //******* READING COOLING GRID **********
  //***************************************
  // Reading Cooling Grid file.

/*
  string fnamex = "sorted_CloudyCoolingGrid.csv";
  const int N_u = 301;
  const int N_nH = 351;
  const int NGrid = N_u * N_nH;
  // float ref_dt_cgs = 100.0f * 365.24f * 24.0f * 3600.0f; // i.e 100 years.

  vector<vector<string>> contentx;
  vector<string> rowx;
  string linex, wordx;

  fstream filex(fnamex, ios::in);
  if (filex.is_open())
  {
    while (getline(filex, linex))
    {
      rowx.clear();

      stringstream str(linex);

      while (getline(str, wordx, ','))
        rowx.push_back(wordx);
      contentx.push_back(rowx);
    }
  }
  else
    cout << "Could not open the cooling grid file\n";
    */

  /*
  float *uZ = new float[NGrid];
  float *nHZ = new float[NGrid];
  float *heatZ = new float[NGrid];
  float *coolZ = new float[NGrid];

  for (int i = 0; i < NGrid; i++)
  {

    uZ[i] = stof(contentx[i][0]);
    nHZ[i] = stof(contentx[i][1]);
    heatZ[i] = stof(contentx[i][3]);
    coolZ[i] = stof(contentx[i][4]);
  }
  

  float *uGrid = new float[N_u];
  float *nHGrid = new float[N_nH];

  for (int i = 0; i < N_u; i++)
  {
    uGrid[i] = uZ[i * N_nH];
  }

  for (int i = 0; i < N_nH; i++)
  {
    nHGrid[i] = nHZ[i];
  }

  float MIN_u = uGrid[0];
  float MAX_u = uGrid[N_u - 1];
  */

  //float MIN_nH = nHGrid[0];
  //float MAX_nH = nHGrid[N_nH - 1];
  //****** END OF READING THE CLOUDY COOLING GRID *********

  //float *d_uGrid, *d_nHGrid, *d_uZ, *d_nHZ, *d_heatZ, *d_coolZ;

  // declaring the arrays.
  float *x, *d_x, *y, *d_y, *z, *d_z, *vx, *d_vx, *vy, *d_vy, *vz, *d_vz;
  float *mass, *d_mass, *h, *d_h, *hprevious, *d_hprevious, *rho, *d_rho;
  float *accx, *accy, *accz, *d_accx, *d_accy, *d_accz, *eps, *d_eps;
  float *P, *d_P, *csnd, *d_csnd, *divV, *d_divV, *curlV, *d_curlV;
  float *accx_sph, *accy_sph, *accz_sph, *d_accx_sph, *d_accy_sph, *d_accz_sph;
  float *accx_tot, *accy_tot, *accz_tot, *d_accx_tot, *d_accy_tot, *d_accz_tot;
  float *abs_acc_g, *abs_acc_tot, *v_sig, *dh_dt, *d_abs_acc_g, *d_abs_acc_tot;
  float *d_v_sig, *d_dh_dt, *u, *dudt, *d_u, *d_dudt, *uprevious, *utprevious;
  float *d_uprevious, *d_utprevious;
  float *Nngb_previous, *d_Nngb_previous; // Note that both are floats and not int! check smoothing func. to see why!
  float *dt_particles, *d_dt_particles;
  int *NGroupz, *d_NGroupz; //, *NGroup2, *d_NGroup2, *NGroup3, *d_NGroup3;

  float G = 1.0f;
  float gammah = 5.0f/3.0f;

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
  hprevious = new float[N];
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

  u = new float[N];
  dudt = new float[N];
  uprevious = new float[N];
  utprevious = new float[N];

  Nngb_previous = new float[N];

  dt_particles = new float[N];

  NGroupz = new int[N];
  // NGroup2 = new int[N];
  // NGroup3 = new int[N];

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
  cudaMalloc(&d_hprevious, N * sizeof(float));
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

  cudaMalloc(&d_u, N * sizeof(float));
  cudaMalloc(&d_dudt, N * sizeof(float));
  cudaMalloc(&d_uprevious, N * sizeof(float));
  cudaMalloc(&d_utprevious, N * sizeof(float));

  //cudaMalloc(&d_uGrid, N_u * sizeof(float));
  //cudaMalloc(&d_nHGrid, N_nH * sizeof(float));
  //cudaMalloc(&d_uZ, NGrid * sizeof(float));
  //cudaMalloc(&d_nHZ, NGrid * sizeof(float));
  //cudaMalloc(&d_heatZ, NGrid * sizeof(float));
  //cudaMalloc(&d_coolZ, NGrid * sizeof(float));

  cudaMalloc(&d_Nngb_previous, N * sizeof(float));
  cudaMalloc(&d_dt_particles, N * sizeof(float));

  cudaMalloc(&d_NGroupz, N * sizeof(int));
  // cudaMalloc(&d_NGroup2, N * sizeof(int));
  // cudaMalloc(&d_NGroup3, N * sizeof(int));

  /* IC file
  0  1  2  3   4   5   6  7          8    9
  x, y, z, vx, vy, vz, m, hprevious, eps, u
  */

  /* RESTART file
  0  1  2  3   4   5   6  7          7    9   10
  x, y, z, vx, vy, vz, m, hprevious, eps, u,  dudt (u wil be uprevious and dudt will be utprevious)!
  */

  // Initialize x, y, and z on the Host.
  for (int i = 0; i < N; i++)
  {
    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];

    vx[i] = vxvec[i];
    vy[i] = vyvec[i];
    vz[i] = vzvec[i];

    mass[i] = massvec[i];
    hprevious[i] = hpreviousvec[i];
    eps[i] = epsvec[i];
    h[i] = 0.0f;    // place holder.
    rho[i] = 0.0f;  // place holder.
    P[i] = 0.0f;    // placeholder.
    csnd[i] = 0.0f; // placeholder.

    divV[i] = 0.0f;  // placeholder.
    curlV[i] = 0.0f; // placeholder.

    accx[i] = 0.0f;
    accy[i] = 0.0f;
    accz[i] = 0.0f;

    accx_sph[i] = 0.0f;
    accy_sph[i] = 0.0f;
    accz_sph[i] = 0.0f;

    accx_tot[i] = 0.0f;
    accy_tot[i] = 0.0f;
    accz_tot[i] = 0.0f;

    abs_acc_g[i] = 0.0f;
    abs_acc_tot[i] = 0.0f;
    v_sig[i] = 0.0f;
    dh_dt[i] = 0.0f;

    if (RESTART == "no")
    {
      u[i] = uvec[i];
      dudt[i] = 0.0f;
      uprevious[i] = 0.0f;
      utprevious[i] = 0.0f;
    }
    else
    {
      u[i] = uvec[i];
      //dudt[i] = 0.0f;
      //uprevious[i] = stof(content[i][9]);
      //utprevious[i] = stof(content[i][10]);
    }

    Nngb_previous[i] = Nngb_f;
    dt_particles[i] = dt;

    NGroupz[i] = i;
    // NGroup2[i] = 0;
    // NGroup3[i] = 0;
  }

  // Copy from Host to Device.
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
  cudaMemcpy(d_hprevious, hprevious, N * sizeof(float), cudaMemcpyHostToDevice);
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

  cudaMemcpy(d_u, u, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dudt, dudt, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uprevious, uprevious, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_utprevious, utprevious, N * sizeof(float), cudaMemcpyHostToDevice);

  //cudaMemcpy(d_uGrid, uGrid, N_u * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_nHGrid, nHGrid, N_nH * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_uZ, uZ, NGrid * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_nHZ, nHZ, NGrid * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_heatZ, heatZ, NGrid * sizeof(float), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_coolZ, coolZ, NGrid * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_Nngb_previous, Nngb_previous, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dt_particles, dt_particles, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_NGroupz, NGroupz, N * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_NGroup2, NGroup2, N * sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_NGroup3, NGroup3, N * sizeof(int), cudaMemcpyHostToDevice);

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  const float visc_alpha = 1.0f;
  const float mH = 1.6726e-24; // gram
  const float kB = 1.3807e-16; // cm2 g s-2 K-1
  const float XH = 0.76;

  // We set MAX_dt_code_unit to avoid negative u !
  // float MAX_dt_code_unit = ref_dt_cgs / unitTime_in_s;

  float t;

  if (RESTART == "no")
  {
    t = 0.0f;
  }
  else
  {
    t = ttime;
  }

  // float dt = MAX_dt_code_unit;
  float tEnd = 5.0f;
  float Nt = ceil(tEnd / dt) + 1;

  //-----------------------------------------------
  //-------------- Smoothing Length ---------------
  //-----------------------------------------------
  if (RESTART == "no")
  {
    smoothing_h<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_hprevious,
                                         N, Ndown, Nup, coeff,
                                         Nngb_f, d_Nngb_previous, d_divV, dt);
    cudaDeviceSynchronize();
  }

  //-----------------------------------------------
  //----------------- getDensity ------------------
  //-----------------------------------------------
  getDensity<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_mass,
                                      d_rho, d_h, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ getAcc_g -------------------
  //-----------------------------------------------
  acc_g<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                 d_mass, G, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //---------------- getPressure ------------------
  //-----------------------------------------------
  getPressure_Adiabatic<<<gridSize, blockSize>>>(d_P, d_rho, d_u, gammah, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //----------------- getCsound -------------------
  //-----------------------------------------------
  getCsound_Adiabatic<<<gridSize, blockSize>>>(d_csnd, d_u, gammah, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //----------------- div_curlV -------------------
  //-----------------------------------------------
  div_curlVel<<<gridSize, blockSize>>>(d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                       d_rho, d_mass, d_h, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ acc_sph --------------------
  //-----------------------------------------------
  acc_sph<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                   d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                   d_accz_sph, visc_alpha, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ acc_tot --------------------
  //-----------------------------------------------
  acc_g_sph<<<gridSize, blockSize>>>(d_accx_tot, d_accy_tot, d_accz_tot,
                                     d_accx, d_accy, d_accz,
                                     d_accx_sph, d_accy_sph, d_accz_sph,
                                     N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------- du_dt ---------------------
  //-----------------------------------------------
  if (RESTART == "no")
  {
    get_dU<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                    d_divV, d_curlV, d_mass, d_P, d_dudt,
                                    visc_alpha, N);
    cudaDeviceSynchronize();
  }

  //-----------------------------------------------
  //---------------- u evolution ------------------
  //-----------------------------------------------
  if (RESTART == "no")
  {
    u_updater1<<<gridSize, blockSize>>>(d_u, d_dudt, dt, N);
    cudaDeviceSynchronize();
  }

  //-----------------------------------------------
  //------------- applyCloudyCooling --------------
  //-----------------------------------------------
  /*
  float current_dt_cgs = dt * unitTime_in_s;

  if (RESTART == "no")
  {
    applyCloudyCooling<<<gridSize, blockSize>>>(d_uZ, d_nHZ, d_heatZ, d_coolZ,
                                                d_uGrid, d_nHGrid, XH,
                                                MIN_u, MAX_u, MIN_nH,
                                                MAX_nH, Unit_u_in_cgs,
                                                UnitDensity_in_cgs, d_u, d_rho,
                                                gammah, mH, kB, d_dudt,
                                                current_dt_cgs, N, N_u, N_nH,
                                                NGrid);
    cudaDeviceSynchronize();
  }
  */

  //-----------------------------------------------
  //-------- updating uprevious, utprevious -------
  //-----------------------------------------------
  if (RESTART == "no")
  {
    u_ut_previous_updater<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                                   d_utprevious, N);
    cudaDeviceSynchronize();
  }

  // float v_signal, min_h, dt_cfl, max_abs_acc_g, max_abs_acc_tot;
  // float dt_f, dt_kin, min_h_dh_dt, dt_dens;
  // float dtz[4];

  const float C_CFL = 0.25;

  float dt_max;
  float dt_B0, dt_B1, dt_B2, dt_B3;
  int jG1, jG2, jG3, NG1, NG2, NG3, NN, jj;
  NG1 = N;
  jj = 1;

  float h_min, h_max;

  //float *uBeforeCooling = new float[N];

  // **************************************************************
  // *********************** MAIN LOOP ****************************
  // **************************************************************

  int counter = 0;

  while (t < tEnd)
  {

    auto begin = std::chrono::high_resolution_clock::now();

    //****************** velocity evolution *******************
    v_evolve<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_accx_tot, d_accy_tot,
                                      d_accz_tot, dt, N);
    cudaDeviceSynchronize();

    //****************** position evolution *******************
    r_evolve<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, N);
    cudaDeviceSynchronize();

    //****************** Smoothing Length *********************
    auto T_00 = std::chrono::high_resolution_clock::now();

    if (jj == 4)
    {
      jj = 1;
    }

    cout << "current jj = " << jj << endl;

    if (jj == 1)
    {
      NN = NG1;
    }
    if (jj == 2) // i.e. jj = 1 and 2
    {
      NN = NG1 + NG2;
    }
    if (jj == 3) // i.e. jj = 1, 2, and 3
    {
      NN = N; // Note that N = NG1 + NG2 + NG3
    }
    smoothing_hX2<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_hprevious,
                                          N, Ndown, Nup, coeff,
                                          Nngb_f, d_Nngb_previous, d_divV,
                                          NN, d_NGroupz, dt);
    cudaDeviceSynchronize();

    auto end_00 = std::chrono::high_resolution_clock::now();
    auto elapsed_00 = std::chrono::duration_cast<std::chrono::nanoseconds>(end_00 - T_00);
    cout << "T_00 = " << elapsed_00.count() * 1e-9 << endl;

    //****************** updating hprevious ***************
    hprevious_updater<<<gridSize, blockSize>>>(d_hprevious,
                                               d_h, N);
    cudaDeviceSynchronize();

    //****************** getDensity ***********************
    getDensity<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_mass,
                                        d_rho, d_h, N);
    cudaDeviceSynchronize();

    //****************** getAcc_gX ************************* MODIFIED FOR Individual dt!
    auto T_acc_g = std::chrono::high_resolution_clock::now();
    acc_gX<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                    d_mass, G, N, NN, d_NGroupz);
    cudaDeviceSynchronize();
    auto end_acc_g = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_g = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_g - T_acc_g);
    cout << "T_acc_g = " << elapsed_acc_g.count() * 1e-9 << endl;

    //****************** getPressure **********************
    getPressure_Adiabatic<<<gridSize, blockSize>>>(d_P, d_rho, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** getCsound ************************
    getCsound_Adiabatic<<<gridSize, blockSize>>>(d_csnd, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** div_curlVX ************************ MODIFIED FOR Individual dt!
    auto T_divCurl = std::chrono::high_resolution_clock::now();
    div_curlVelX<<<gridSize, blockSize>>>(d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                          d_rho, d_mass, d_h, N, NN, d_NGroupz);
    cudaDeviceSynchronize();
    auto end_divCurl = std::chrono::high_resolution_clock::now();
    auto elapsed_divCurl = std::chrono::duration_cast<std::chrono::nanoseconds>(end_divCurl - T_divCurl);
    cout << "T_divCurl = " << elapsed_divCurl.count() * 1e-9 << endl;

    //****************** acc_sphX ************************** MODIFIED FOR Individual dt!
    auto T_acc_sph = std::chrono::high_resolution_clock::now();
    acc_sphX<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                      d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                      d_accz_sph, visc_alpha, N, NN, d_NGroupz);
    cudaDeviceSynchronize();
    auto end_acc_sph = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_sph = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_sph - T_acc_sph);
    cout << "T_acc_sph = " << elapsed_acc_sph.count() * 1e-9 << endl;

    //****************** acc_tot **************************
    auto T_acc_tot = std::chrono::high_resolution_clock::now();
    acc_g_sph<<<gridSize, blockSize>>>(d_accx_tot, d_accy_tot, d_accz_tot,
                                       d_accx, d_accy, d_accz,
                                       d_accx_sph, d_accy_sph, d_accz_sph, N);
    cudaDeviceSynchronize();
    auto end_acc_tot = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_tot = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_tot - T_acc_tot);
    cout << "T_acc_tot = " << elapsed_acc_tot.count() * 1e-9 << endl;

    //****************** velocity evolution *******************
    v_evolve<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_accx_tot, d_accy_tot,
                                      d_accz_tot, dt, N);
    cudaDeviceSynchronize();

    //******************** get_dUX (du_dt) ********************* MODIFIED FOR Individual dt!
    auto T_dU = std::chrono::high_resolution_clock::now();
    get_dUX<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                     d_divV, d_curlV, d_mass, d_P, d_dudt,
                                     visc_alpha, N, NN, d_NGroupz);
    cudaDeviceSynchronize();
    auto end_dU = std::chrono::high_resolution_clock::now();
    auto elapsed_dU = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dU - T_dU);
    cout << "T_dU = " << elapsed_dU.count() * 1e-9 << endl;

    //******************** u evolution *********************
    u_updater_main<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                            d_utprevious, dt, N);
    cudaDeviceSynchronize();

    //******************** applyCloudyCooling *********************
    /*
    auto T_cool = std::chrono::high_resolution_clock::now();

    current_dt_cgs = dt * unitTime_in_s;

    applyCloudyCooling<<<gridSize, blockSize>>>(d_uZ, d_nHZ, d_heatZ, d_coolZ,
                                                d_uGrid, d_nHGrid, XH,
                                                MIN_u, MAX_u, MIN_nH,
                                                MAX_nH, Unit_u_in_cgs,
                                                UnitDensity_in_cgs, d_u, d_rho,
                                                gammah, mH, kB, d_dudt,
                                                current_dt_cgs, N, N_u, N_nH,
                                                NGrid);
    cudaDeviceSynchronize();
    auto end_cool = std::chrono::high_resolution_clock::now();
    auto elapsed_cool = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cool - T_cool);
    cout << "T_cool = " << elapsed_cool.count() * 1e-9 << endl;
    */    

    h_min = min_finder(h, N);
    h_max = max_finder(h, N);
    cout << "min_h = " << h_min << endl;
    cout << "max_h = " << h_max << endl;


    if(!(counter % 10)){
      cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(vx, d_vx, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vz, d_vz, N*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(rho, d_rho, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h, d_h, N*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(u, d_u, N*sizeof(float), cudaMemcpyDeviceToHost);

      ofstream outfile("./Outputs/G-"+ to_string(t*1) + ".csv");

      outfile << "x" << "," << "y" << "," << "z" << "," << "vx" << "," << "vy" << ","
              << "vz" << "," << "h" << "," << "rho" << "," << "u" << endl; // this will be the header !

      for(int i = 0; i < N; i++){
        outfile << x[i] << "," << y[i] << "," << z[i] << ","
                << vx[i] << "," << vy[i] << "," << vz[i] << ","
                << h[i] << "," << rho[i] << "," << u[i] << endl;
      }
    }

    //******* updating uprevious, utprevious ********
    auto T_u_ut = std::chrono::high_resolution_clock::now();
    u_ut_previous_updater<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                                   d_utprevious, N);
    cudaDeviceSynchronize();
    auto end_u_ut = std::chrono::high_resolution_clock::now();
    auto elapsed_u_ut = std::chrono::duration_cast<std::chrono::nanoseconds>(end_u_ut - T_u_ut);
    cout << "T_update_u_ut = " << elapsed_u_ut.count() * 1e-9 << endl;

    //******************** dt_array_indiv_dt *********************
    auto T_dt_arr = std::chrono::high_resolution_clock::now();
    dt_array_indiv_dt<<<gridSize, blockSize>>>(d_x, d_y, d_z,
                                               d_vx, d_vy, d_vz,
                                               d_accx, d_accy, d_accz,
                                               d_accx_tot, d_accy_tot, d_accz_tot,
                                               d_h, d_csnd, d_dt_particles,
                                               d_abs_acc_g, d_abs_acc_tot,
                                               d_divV, d_dh_dt, C_CFL,
                                               visc_alpha, d_eps, N);
    cudaDeviceSynchronize();
    auto end_dt_arr = std::chrono::high_resolution_clock::now();
    auto elapsed_dt_arr = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dt_arr - T_dt_arr);
    cout << "T_dt_arr = " << elapsed_dt_arr.count() * 1e-9 << endl;

    cudaMemcpy(dt_particles, d_dt_particles, N * sizeof(float), cudaMemcpyDeviceToHost);

    dt = min_finder(dt_particles, N);

    dt_max = max_finder(dt_particles, N);

    dt_B0 = dt;           // Note that this is the minimum time-step!
    dt_B1 = dt_B0 * 2.0f; // 2.0 = 2.0^1
    dt_B2 = dt_B0 * 4.0;  // 4.0 = 2.0^2
    dt_B3 = dt_B0 * 8.0;  // 8.0 = 2.0^3

    jG1 = 0;
    jG2 = 0;
    jG3 = 0;

    NG1 = 0;
    NG2 = 0;
    NG3 = 0;

    auto T_FOR = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < N; j++)
    {
      if ((dt_particles[j] >= dt_B0) && (dt_particles[j] < dt_B1))
      {
        NGroupz[jG1] = j;
        jG1++;
      }
    }

    for (int j = 0; j < N; j++)
    {
      if ((dt_particles[j] >= dt_B1) && (dt_particles[j] <= dt_B2))
      {
        NGroupz[jG1 + jG2] = j;
        jG2++;
      }
    }

    for (int j = 0; j < N; j++)
    {
      if (dt_particles[j] >= dt_B2)
      {
        NGroupz[jG1 + jG2 + jG3] = j;
        jG3++;
      }
    }

    jj++; // different jj means extrapolation of h for different groups !

    auto end_FOR = std::chrono::high_resolution_clock::now();
    auto elapsed_FOR = std::chrono::duration_cast<std::chrono::nanoseconds>(end_FOR - T_FOR);
    cout << "T_FOR = " << elapsed_FOR.count() * 1e-9 << endl;

    auto T_NG = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_NGroupz, NGroupz, N * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_NGroup2, NGroup2, N * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_NGroup3, NGroup3, N * sizeof(int), cudaMemcpyHostToDevice);
    auto end_NG = std::chrono::high_resolution_clock::now();
    auto elapsed_NG = std::chrono::duration_cast<std::chrono::nanoseconds>(end_NG - T_NG);
    cout << "T_NG = " << elapsed_NG.count() * 1e-9 << endl;

    NG1 = jG1;
    NG2 = jG2;
    NG3 = jG3;

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    cout << "Elapsed time = " << elapsed.count() * 1e-9 << endl;
    cout << endl;

    cout << "min dt = " << dt << endl;
    cout << "max dt = " << dt_max << endl;
    cout << "NG1 = " << NG1 << endl;
    cout << "NG2 = " << NG2 << endl;
    cout << "NG3 = " << NG3 << endl;

    t += dt;

    if (!(counter % 1))
    {
      cout << "Adopted dt = " << dt << endl;
      cout << "current t = " << t << endl;
      cout << "*****************************" << endl;
      cout << endl;
    }

    counter++;
  }

  delete[] x;
  delete[] y;
  delete[] z;
  delete[] vx;
  delete[] vy;
  delete[] vz;
  delete[] mass;
  delete[] h;
  delete[] hprevious;
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
  delete[] uprevious;
  delete[] utprevious;
  //delete[] uGrid;
  //delete[] uZ;
  //delete[] nHZ;
  //delete[] heatZ;
  //delete[] coolZ;

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  cudaFree(d_vx);
  cudaFree(d_vy);
  cudaFree(d_vz);
  cudaFree(d_mass);
  cudaFree(d_h);
  cudaFree(d_hprevious);
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
  cudaFree(d_uprevious);
  cudaFree(d_utprevious);
  //cudaFree(d_uGrid);
  //cudaFree(d_uZ);
  //cudaFree(d_nHZ);
  //cudaFree(d_heatZ);
  //cudaFree(d_coolZ);
}
