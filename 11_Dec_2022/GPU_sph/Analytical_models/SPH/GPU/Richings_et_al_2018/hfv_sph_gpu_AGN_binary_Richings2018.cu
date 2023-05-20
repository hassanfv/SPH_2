//%%writefile test.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include "myCppSPHLibs_v7_t6_exclude_BH.h"
#include <curand_kernel.h>
using namespace std;

// smoothing func. updated + my_pi is replaced by M_PI. (28 Jan - 2023)
// In this version, we implemented the Restart possibility! (24 Jan - 2023)
// In this version, we use CLOUDY cooling & heating!
// In this version, we also include cooling.
// In this version, the output file also contains the velocity components.

string RESTART = "no"; // options = yes or no (note they are in lower case letters)!!!!!!!!!!!!!!!!!



// Function to save the OUTPUT Snap-Shots!!
void saveArraysToBinary(const std::string& filename, float* x, float* y, float* z, float* vx, float* vy, float* vz, int N, float* rho, float* h, float* u, int NG)
{
    // Open the file in binary mode
    std::ofstream file(filename, std::ios::binary);
    
    // Check if the file was opened successfully
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write N and NG to the file
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    file.write(reinterpret_cast<const char*>(&N), sizeof(int));
    
    // Write the arrays to the file
    file.write(reinterpret_cast<const char*>(x), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(y), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(z), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(vx), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(vy), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(vz), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(rho), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(h), N * sizeof(float));
    file.write(reinterpret_cast<const char*>(u), N * sizeof(float));
    
    // Close the file
    file.close();
}



//========================================
//======= Smoothing Length (BH) =========
//========================================

float smoothing_h_BH(float *x, float *y, float *z, float hBH,
                                int N, int Ndown, int Nup, float coeff)
{

  float h_new = 2.0f * hBH;
  float h_tmp = h_new;
  int N_iter = 0;
  int k = 0;
  
  float xBH = 0.0f;
  float yBH = 0.0f;
  float zBH = 0.0f;

  float dx, dy, dz;
  while ((k < Ndown) || (k > Nup))
  {

    k = 0;

    for (int j = 0; j < N; j++)
    {
      dx = x[j] - xBH;
      dy = y[j] - yBH;
      dz = z[j] - zBH;
      float rr = sqrt(dx * dx + dy * dy + dz * dz);

      if (rr <= h_new)
      {
        k++;
      }
    }

    if (k < Ndown)
    {
      h_new = h_new + coeff * 2.0f * hBH;
    }

    if (k > Nup)
    {
      h_new = h_new - coeff * 2.0f * hBH;
    }

    if (h_new > h_tmp)
    {
      h_tmp = h_new;
    }

    N_iter++;
    if (N_iter > 1000)
    {
      h_new = h_tmp;
      break;
    }
  }
  hBH = 0.5f * h_new;
  return hBH;
}







//---------------------
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

  float dt = 1e-4; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.001f; // used for smoothing length.

  //***************************************
  //********** Reading params.txt file ************
  //***************************************

  // Open the file
  std::ifstream filex("params.txt");

  // Check if file was successfully opened
  if (!filex) {
    std::cerr << "Unable to open file param.txt";
    return 1; // return with error code 1
  }

  // Variables to store the values
  int N_tot, N, N_blank;
  float G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res;

  // Read the values from the file
  filex >> N_tot;
  filex >> N;
  filex >> N_blank;
  filex >> G;
  filex >> L_AGN_code_unit;
  filex >> M_dot_in;
  filex >> v_in;
  filex >> u_for_10K_Temp;
  filex >> m_sph_high_res;

  // Close the file
  filex.close();


  int N_real = N;
  
  N = N_tot;


  //***************************************
  //********** Reading IC file ************
  //***************************************

  std::vector<float> xvec(N_tot);
  std::vector<float> yvec(N_tot);
  std::vector<float> zvec(N_tot);
  std::vector<float> vxvec(N_tot);
  std::vector<float> vyvec(N_tot);
  std::vector<float> vzvec(N_tot);
  std::vector<float> massvec(N_tot);
  std::vector<float> hpreviousvec(N_tot);
  std::vector<float> epsvec(N_tot);
  std::vector<float> uvec(N_tot);

  // Check if the binary file exists
  std::ifstream file("IC_Richings2018_110k.bin", std::ios::binary);
  if (!file) {
    std::cout << "Could not open the binary file." << std::endl;
  }
  file.close();
  
  // Open the binary file for reading
  file.open("IC_Richings2018_110k.bin", std::ios::binary);
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
  float *weightsBH, *d_weightsBH, *dt_particles, *d_dt_particles;

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
  dt_particles = new float[N];

  u = new float[N];
  dudt = new float[N];
  uprevious = new float[N];
  utprevious = new float[N];

  Nngb_previous = new float[N];

  weightsBH = new float[N];

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
  cudaMalloc(&d_dt_particles, N * sizeof(float));

  cudaMalloc(&d_u, N * sizeof(float));
  cudaMalloc(&d_dudt, N * sizeof(float));
  cudaMalloc(&d_uprevious, N * sizeof(float));
  cudaMalloc(&d_utprevious, N * sizeof(float));

  cudaMalloc(&d_Nngb_previous, N * sizeof(float));

  cudaMalloc(&d_weightsBH, N * sizeof(float));

  /* IC file
  0  1  2  3   4   5   6  7          8    9
  x, y, z, vx, vy, vz, m, hprevious, eps, u
  */

  /* RESTART file
  0  1  2  3   4   5   6  7          7    9   10
  x, y, z, vx, vy, vz, m, hprevious, eps, u,  dudt (u wil be uprevious and dudt will be utprevious)!
  */

  // Initialize x, y, and z on the Host (All particles, i.e. Gas + DM + BH).
  for (int i = 0; i < N; i++)
  {
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
    //NGroupz[i] = i;
    
  }
  
  
  // Initialize x, y, and z on the Host (Only Gas particles).
  for (int i = 0; i < N; i++)
  {
    hprevious[i] = hpreviousvec[i];
    h[i] = hpreviousvec[i];    // place holder.
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
    uprevious[i] = 0.0f;
    utprevious[i] = 0.0f;
    
    dt_particles[i] = 0.0f;

    Nngb_previous[i] = Nngb_f;
    
    weightsBH[i] = 0.0f;
    
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
  cudaMemcpy(d_dt_particles, dt_particles, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_u, u, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dudt, dudt, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uprevious, uprevious, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_utprevious, utprevious, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_Nngb_previous, Nngb_previous, N * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_weightsBH, weightsBH, N * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  const float visc_alpha = 1.0f;
  //const float mH = 1.6726e-24; // gram
  //const float kB = 1.3807e-16; // cm2 g s-2 K-1
  //const float XH = 0.76;

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
  
  N = N_real; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  //-----------------------------------------------
  //-------------- Smoothing Length ---------------
  //-----------------------------------------------
  if (RESTART == "no")
  {
    smoothing_h_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_hprevious,
                                         N, Ndown, Nup, coeff,
                                         Nngb_f, d_Nngb_previous, d_divV, dt);
    cudaDeviceSynchronize();
  }

  //-----------------------------------------------
  //----------------- getDensity ------------------
  //-----------------------------------------------
  getDensity_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_mass,
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
  getPressure_Adiabatic_exBH<<<gridSize, blockSize>>>(d_P, d_rho, d_u, gammah, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //----------------- getCsound -------------------
  //-----------------------------------------------
  getCsound_Adiabatic_exBH<<<gridSize, blockSize>>>(d_csnd, d_u, gammah, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //----------------- div_curlV -------------------
  //-----------------------------------------------
  div_curlVel_exBH<<<gridSize, blockSize>>>(d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                       d_rho, d_mass, d_h, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ acc_sph --------------------
  //-----------------------------------------------
  acc_sph_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
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
    get_dU_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                    d_divV, d_curlV, d_mass, d_P, d_dudt,
                                    visc_alpha, N);
    cudaDeviceSynchronize();
  }

  //-----------------------------------------------
  //---------------- u evolution ------------------
  //-----------------------------------------------
  if (RESTART == "no")
  {
    u_updater1_exBH<<<gridSize, blockSize>>>(d_u, d_dudt, dt, N);
    cudaDeviceSynchronize();
  }


  //-----------------------------------------------
  //-------- updating uprevious, utprevious -------
  //-----------------------------------------------
  if (RESTART == "no")
  {
    u_ut_previous_updater_exBH<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                                   d_utprevious, N);
    cudaDeviceSynchronize();
  }

  const float C_CFL = 0.25;

  float dt_max;
  //float dt_B0, dt_B1, dt_B2, dt_B3;
  //int jG1, jG2, jG3, NG1, NG2, NG3, NN, jj;
  //NG1 = NG; // Note that NG1, NG2, and NG3 are just accidentally named like NG!!!
  //jj = 1;

  float h_min, h_max;
  
  float leftover_mass = 0.0f;

  //int Energy_checker = 0;


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

    //****************** position evolution (BH fixed at [0, 0, 0]) *******************

    r_evolve_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, N);
    cudaDeviceSynchronize();

    //****************** Smoothing Length *********************

    smoothing_h_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_hprevious,
                                          N, Ndown, Nup, coeff,
                                          Nngb_f, d_Nngb_previous, d_divV,
                                          dt);
    cudaDeviceSynchronize();


    //****************** updating hprevious ***************
    hprevious_updater<<<gridSize, blockSize>>>(d_hprevious,
                                               d_h, N);
    cudaDeviceSynchronize();

    //****************** getDensity ***********************
    getDensity_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_mass,
                                        d_rho, d_h, N);
    cudaDeviceSynchronize();

    //****************** getAcc_gX *************************
    auto T_acc_g = std::chrono::high_resolution_clock::now();
    acc_g<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                    d_mass, G, N);
    cudaDeviceSynchronize();
    auto end_acc_g = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_g = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_g - T_acc_g);
    cout << "T_acc_g = " << elapsed_acc_g.count() * 1e-9 << endl;

    //****************** getPressure **********************
    getPressure_Adiabatic_exBH<<<gridSize, blockSize>>>(d_P, d_rho, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** getCsound ************************
    getCsound_Adiabatic_exBH<<<gridSize, blockSize>>>(d_csnd, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** div_curlVX ************************ 
    auto T_divCurl = std::chrono::high_resolution_clock::now();
    div_curlVel_exBH<<<gridSize, blockSize>>>(d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                          d_rho, d_mass, d_h, N);
    cudaDeviceSynchronize();
    auto end_divCurl = std::chrono::high_resolution_clock::now();
    auto elapsed_divCurl = std::chrono::duration_cast<std::chrono::nanoseconds>(end_divCurl - T_divCurl);
    cout << "T_divCurl = " << elapsed_divCurl.count() * 1e-9 << endl;

    //****************** acc_sphX ************************** 
    auto T_acc_sph = std::chrono::high_resolution_clock::now();
    acc_sph_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                      d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                      d_accz_sph, visc_alpha, N);
    cudaDeviceSynchronize();
    auto end_acc_sph = std::chrono::high_resolution_clock::now();
    auto elapsed_acc_sph = std::chrono::duration_cast<std::chrono::nanoseconds>(end_acc_sph - T_acc_sph);
    cout << "T_acc_sph = " << elapsed_acc_sph.count() * 1e-9 << endl;

    //****************** acc_tot **************************
    auto T_acc_tot = std::chrono::high_resolution_clock::now();
    acc_g_sph_exBH<<<gridSize, blockSize>>>(d_accx_tot, d_accy_tot, d_accz_tot,
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

    //******************** get_dUX (du_dt) *********************
    auto T_dU = std::chrono::high_resolution_clock::now();
    get_dU_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                     d_divV, d_curlV, d_mass, d_P, d_dudt,
                                     visc_alpha, N);
    cudaDeviceSynchronize();
    auto end_dU = std::chrono::high_resolution_clock::now();
    auto elapsed_dU = std::chrono::duration_cast<std::chrono::nanoseconds>(end_dU - T_dU);
    cout << "T_dU = " << elapsed_dU.count() * 1e-9 << endl;

    //******************** u evolution *********************
    u_updater_main_exBH<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                            d_utprevious, dt, N);
    cudaDeviceSynchronize();




    
    
    //**************** AGN Wind Injection *******************
    agn_wind_injection<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, 
                                                d_u, d_mass, d_h, d_eps, M_dot_in, v_in, dt,
                                                leftover_mass, m_sph_high_res, u_for_10K_Temp, N);
    
    N = N + static_cast<int>(u[0]); // Updating the number of particles as new particles may be created above!
    leftover_mass = u[N+1];


    cudaMemcpy(u, d_u, N*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
    {
      cout << "uuuuu = " << u[i] << endl;
    }


    //-------------------------------------------------

    h_min = min_finder(h, N);
    h_max = max_finder(h, N);
    cout << "min_h = " << h_min << endl;
    cout << "max_h = " << h_max << endl;

    

    cudaMemcpy(rho, d_rho, N*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 5; i++)
    {
      cout << "AAA = " << rho[i] << endl;
    }



    
    //------------ SAVING SNAP-SHOTS ------------
    if(!(counter % 20)){
      cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(vx, d_vx, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vz, d_vz, N*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(rho, d_rho, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h, d_h, N*sizeof(float), cudaMemcpyDeviceToHost);

      cudaMemcpy(u, d_u, N*sizeof(float), cudaMemcpyDeviceToHost);

      // Specify the output file name
      std::string filename = "./Outputs/G-"+ to_string(t*1) + ".bin";
      // Save the arrays to binary format
      saveArraysToBinary(filename, x, y, z, vx, vy, vz, N, rho, h, u, N);
    }

    //******* updating uprevious, utprevious ********
    auto T_u_ut = std::chrono::high_resolution_clock::now();
    u_ut_previous_updater_exBH<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                                   d_utprevious, N);
    cudaDeviceSynchronize();
    auto end_u_ut = std::chrono::high_resolution_clock::now();
    auto elapsed_u_ut = std::chrono::duration_cast<std::chrono::nanoseconds>(end_u_ut - T_u_ut);
    cout << "T_update_u_ut = " << elapsed_u_ut.count() * 1e-9 << endl;


    //auto end_FOR = std::chrono::high_resolution_clock::now();
    //auto elapsed_FOR = std::chrono::duration_cast<std::chrono::nanoseconds>(end_FOR - T_FOR);
    //cout << "T_FOR = " << elapsed_FOR.count() * 1e-9 << endl;

    auto T_NG = std::chrono::high_resolution_clock::now();
    //cudaMemcpy(d_NGroupz, NGroupz, NG * sizeof(int), cudaMemcpyHostToDevice);
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
    
    dt_array_indiv_dt_exBH<<<gridSize, blockSize>>>(d_x, d_y, d_z,
                                               d_vx, d_vy, d_vz,
                                               d_accx, d_accy, d_accz,
                                               d_accx_tot, d_accy_tot, d_accz_tot,
                                               d_h, d_csnd, d_dt_particles,
                                               d_abs_acc_g, d_abs_acc_tot,
                                               d_divV, d_dh_dt, C_CFL,
                                               visc_alpha, d_eps, N);
    cudaDeviceSynchronize();

    cudaMemcpy(dt_particles, d_dt_particles, N * sizeof(float), cudaMemcpyDeviceToHost);

    dt = min_finder(dt_particles, N);

    dt_max = max_finder(dt_particles, N);
    
    
    

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
}
