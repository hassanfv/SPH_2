#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include "myCppSPHLibs_v4.h"
#include "myPhotoLibsGPU_v2.h"
using namespace std;


// In this version, we also include cooling.
// In this version, the output file also contains the velocity components.


float max_finder(float *arr, int N){

      float max_val = 0.0;
      for(int i = 0; i < N; i++){
        if(arr[i] >= max_val){
          max_val = arr[i];
        }
      }
      return max_val;
    }

    float min_finder(float *arr, int N){

      float min_val = arr[0];
      for(int i = 0; i < N; i++){
        if(arr[i] <= min_val){
          min_val = arr[i];
        }
      }
      return min_val;
    }


int main(){

  // Reading params.GPU file.
  ifstream infile;
  infile.open("params.GPU");

  int N;
  float c_0, gammah, Rcld_in_pc, Rcld_in_cm, Mcld_in_g, muu, Mach;
  float grav_const_in_cgs, G;

  if (infile.is_open()){
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
  }else{
    cout << "File Not Found !!!" << endl;
  }

  long double UnitRadius_in_cm = Rcld_in_cm;
  long double UnitRadius_in_cm_2 = UnitRadius_in_cm*UnitRadius_in_cm;

  long double UnitMass_in_g = Mcld_in_g;
  //-------------------------
  long double UnitDensity_in_cgsT = UnitMass_in_g / pow(UnitRadius_in_cm,3);
  //-------------------------
  long double Unit_u_in_cgsT = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm;
  long double Unit_P_in_cgsT = UnitDensity_in_cgsT * Unit_u_in_cgsT;
  long double unitVelocityT = sqrt(grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm);
  long double unitTime_in_sT = sqrt(pow(UnitRadius_in_cm,3) / grav_const_in_cgs/UnitMass_in_g);

  float UnitDensity_in_cgs = (float) UnitDensity_in_cgsT;
  float Unit_u_in_cgs = (float) Unit_u_in_cgsT;
  float Unit_P_in_cgs = (float) Unit_P_in_cgsT;
  float unitVelocity = (float) unitVelocityT;
  float unitTime_in_s = (float) unitTime_in_sT;

  cout << "UnitDensity_in_cgs = " << UnitDensity_in_cgs << endl;
  cout << "Unit_u_in_cgs = " << Unit_u_in_cgs << endl;
  cout << "Unit_P_in_cgs = " << Unit_P_in_cgs << endl;
  cout << "unitVelocity = " << unitVelocity << endl;
  cout << "unitTime_in_s = " << unitTime_in_s << endl;
  cout << endl;

  // Reading Hydra file.
  string fname = "GPU_IC_DLA_120k.csv"; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  vector<vector<string>> content;
  vector<string> row;
  string line, word;
  
  fstream file (fname, ios::in);
  if(file.is_open())
  {
  while(getline(file, line))
  {
  row.clear();
  
  stringstream str(line);
  
  while(getline(str, word, ','))
  row.push_back(word);
  content.push_back(row);
  }
  }
  else
  cout<<"Could not open the file\n";

  //***************************************
  //******* READING COOLING GRID **********
  //***************************************
  // Reading Cooling Grid file.

  string fnamex = "sortedCoolingGrid_J_0.00001.csv";
  const int N_u = 1000;
  const int N_rho = 1000;
  const int NGrid = N_u * N_rho;
  float ref_dt_cgs = 200.0f * 365.24f * 24.0f * 3600.0f; // i.e 200 years.

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

  float *uadT = new float[NGrid];
  float *rhoT = new float[NGrid];
  float *delta_u = new float[NGrid];

  for (int i = 0; i < NGrid; i++)
  {

      uadT[i] = stof(contentx[i][0]);
      rhoT[i] = stof(contentx[i][1]);
      delta_u[i] = stof(contentx[i][3]); // Note that it is the fourth index (i.e. 3) !
  }

  /*
  float Tmin = 1e4;
  float Tmax = 1e8;

  float stp_T = (log10(Tmax) - log10(Tmin)) / N_u;
  float *Tgrid = new float[N_u];

  for (int i = 0; i < N_u; i++)
  {
      Tgrid[i] = pow(10, (log10(Tmin) + i * stp_T));
  }

  //-------- Converting T to u.
  const float XH = 0.76;
  const float mH = 1.6726e-24; // gram
  float nHcgs = 1.0;             //  cm^-3
  float *uGrid = new float[N_u];

  for (int i = 0; i < N_u; i++)
  {
      uGrid[i] = convert_Temp_to_u(Tgrid[i], nHcgs, XH);
  }
  */

  

  //delete[] Tgrid;

  /*
  float nH_min = 1e-4;
  float nH_max = 1e2;
  float rho_min = nH_min * mH;
  float rho_max = nH_max * mH;

  float stp_rho = (log10(rho_max) - log10(rho_min)) / N_rho;
  float *rhoGrid = new float[N_rho];

  for (int i = 0; i < N_rho; i++)
  {
      rhoGrid[i] = pow(10, (log10(rho_min) + i * stp_rho));
  }

  */

  float *uGrid = new float[N_u];
  float *rhoGrid = new float[N_rho];
  
  for (int i = 0; i < N_u; i++) // Note: N_u = N_rho = 1000 !
  {
    uGrid[i] = uadT[i * 1000];
    rhoGrid[i] = rhoT[i];
  }
  


  float MIN_uad = uGrid[0];
  float MAX_uad = uGrid[N_u - 1];

  float MIN_rho = rhoGrid[0];
  float MAX_rho = rhoGrid[N_rho - 1];
  //****** END OF READING THE COOLING GRID *********

  float *d_uGrid, *d_rhoGrid, *d_uadT, *d_rhoT, *d_delta_u;

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

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));
  cudaMalloc(&d_z, N*sizeof(float));

  cudaMalloc(&d_vx, N*sizeof(float));
  cudaMalloc(&d_vy, N*sizeof(float));
  cudaMalloc(&d_vz, N*sizeof(float));

  cudaMalloc(&d_accx, N*sizeof(float));
  cudaMalloc(&d_accy, N*sizeof(float));
  cudaMalloc(&d_accz, N*sizeof(float));

  cudaMalloc(&d_mass, N*sizeof(float));
  cudaMalloc(&d_h, N*sizeof(float));
  cudaMalloc(&d_hprevious, N*sizeof(float));
  cudaMalloc(&d_rho, N*sizeof(float));
  cudaMalloc(&d_eps, N*sizeof(float));
  cudaMalloc(&d_P, N*sizeof(float));
  cudaMalloc(&d_csnd, N*sizeof(float));

  cudaMalloc(&d_divV, N*sizeof(float));
  cudaMalloc(&d_curlV, N*sizeof(float));

  cudaMalloc(&d_accx_sph, N*sizeof(float));
  cudaMalloc(&d_accy_sph, N*sizeof(float));
  cudaMalloc(&d_accz_sph, N*sizeof(float));

  cudaMalloc(&d_accx_tot, N*sizeof(float));
  cudaMalloc(&d_accy_tot, N*sizeof(float));
  cudaMalloc(&d_accz_tot, N*sizeof(float));

  cudaMalloc(&d_abs_acc_g, N*sizeof(float));
  cudaMalloc(&d_abs_acc_tot, N*sizeof(float));
  cudaMalloc(&d_v_sig, N*sizeof(float));
  cudaMalloc(&d_dh_dt, N*sizeof(float));

  cudaMalloc(&d_u, N*sizeof(float));
  cudaMalloc(&d_dudt, N*sizeof(float));
  cudaMalloc(&d_uprevious, N*sizeof(float));
  cudaMalloc(&d_utprevious, N*sizeof(float));

  cudaMalloc(&d_uGrid, N_u*sizeof(float));
  cudaMalloc(&d_rhoGrid, N_rho*sizeof(float));
  cudaMalloc(&d_uadT, NGrid*sizeof(float));
  cudaMalloc(&d_rhoT, NGrid*sizeof(float));
  cudaMalloc(&d_delta_u, NGrid*sizeof(float));

  // 0  1  2  3   4   5   6  7          8
  // x, y, z, vx, vy, vz, m, hprevious, eps

  // Initialize x, y, and z on the Host.
  for(int i = 0; i < N; i++){
    x[i] = stof(content[i][0]);
    y[i] = stof(content[i][1]);
    z[i] = stof(content[i][2]);

    vx[i] = stof(content[i][3]);
    vy[i] = stof(content[i][4]);
    vz[i] = stof(content[i][5]);

    mass[i] = stof(content[i][6]);
    hprevious[i] = stof(content[i][7]);
    eps[i] = stof(content[i][8]);
    h[i] = 0.0f; // place holder.
    rho[i] = 0.0f; // place holder.
    P[i] = 0.0f; // placeholder.
    csnd[i] = 0.0f; // placeholder.

    divV[i] = 0.0f; // placeholder.
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

    u[i] = stof(content[i][9]);
    dudt[i] = 0.0f;
    uprevious[i] = 0.0f;
    utprevious[i] = 0.0f;
  }

  // Copy from Host to Device.
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_vx, vx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vy, vy, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vz, vz, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx, accx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy, accy, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz, accz, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_mass, mass, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_h, h, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_hprevious, hprevious, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rho, rho, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_eps, eps, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_P, P, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_csnd, csnd, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_divV, divV, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_curlV, curlV, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx_sph, accx_sph, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy_sph, accy_sph, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz_sph, accz_sph, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_accx_tot, accx_tot, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accy_tot, accy_tot, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_accz_tot, accz_tot, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_abs_acc_g, abs_acc_g, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_abs_acc_tot, abs_acc_tot, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_sig, v_sig, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dh_dt, dh_dt, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_u, u, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dudt, dudt, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uprevious, uprevious, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_utprevious, utprevious, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_uGrid, uGrid, N_u*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rhoGrid, rhoGrid, N_rho*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uadT, uadT, NGrid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rhoT, rhoT, NGrid*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_delta_u, delta_u, NGrid*sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 256; // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid
  
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.001f; // used for smoothing length.

  const float visc_alpha = 1.0f;

  //float kB = 1.3807e-16; // cm2 g s-2 K-1
  //float mH2 = muu * mH;

  //float kBmH2 = kB/mH2;

  const float my_pi = 3.141592f;

  // We set MAX_dt_code_unit to avoid negative u !
  float MAX_dt_code_unit = ref_dt_cgs / unitTime_in_s;

  float t = 0.0f;
  float dt = MAX_dt_code_unit; //2e-4;
  float tEnd = 2.0f;
  float Nt = ceil(tEnd/dt) + 1;

  //-----------------------------------------------
  //-------------- Smoothing Length ---------------
  //-----------------------------------------------
  smoothing_h<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_hprevious,
                                       N, Ndown, Nup, coeff);
  cudaDeviceSynchronize();
  
  //-----------------------------------------------
  //----------------- getDensity ------------------
  //-----------------------------------------------
  getDensity<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_mass,
                                      d_rho, d_h, my_pi, N);
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
                                       d_rho, d_mass, d_h, my_pi, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //------------------ acc_sph --------------------
  //-----------------------------------------------
  acc_sph<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                   d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                   d_accz_sph, my_pi, visc_alpha, N);
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
  get_dU<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                  d_divV, d_curlV, d_mass, d_P, d_dudt,
                                  my_pi, visc_alpha, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //---------------- u evolution ------------------
  //-----------------------------------------------
  u_updater1<<<gridSize, blockSize>>>(d_u, d_dudt, dt, N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //---------------- applyCooling -----------------
  //-----------------------------------------------
  float current_dt_cgs = dt * unitTime_in_s;

  applyCooling<<<gridSize, blockSize>>>(d_uadT, d_rhoT, d_delta_u,
                                        ref_dt_cgs, d_uGrid, d_rhoGrid,
                                        MIN_uad, MAX_uad, MIN_rho,
                                        MAX_rho, Unit_u_in_cgs,
                                        UnitDensity_in_cgs, d_u, d_rho,
                                        current_dt_cgs, N, N_u, N_rho, NGrid);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //-------- updating uprevious, utprevious -------
  //-----------------------------------------------
  u_ut_previous_updater<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                                 d_utprevious, N);
  cudaDeviceSynchronize();

  float v_signal, min_h, dt_cfl, max_abs_acc_g, max_abs_acc_tot;
  float dt_f, dt_kin, min_h_dh_dt, dt_dens;
  float dtz[4];

  const float C_CFL = 0.25;


  // ************************************************************** 
  // *********************** MAIN LOOP **************************** 
  // **************************************************************
  
  int counter = 0;

  while(t < tEnd){

    //****************** velocity evolution *******************
    v_evolve<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_accx_tot, d_accy_tot,
                                      d_accz_tot, dt, N);
    cudaDeviceSynchronize();
    
    //****************** position evolution *******************
    r_evolve<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, dt, N);
    cudaDeviceSynchronize();

    //****************** Smoothing Length *********************
    smoothing_h<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_hprevious,
                                        N, Ndown, Nup, coeff);
    cudaDeviceSynchronize();

    //****************** updating hprevious ***************
    hprevious_updater<<<gridSize, blockSize>>>(d_hprevious,
                                               d_h, N);
    cudaDeviceSynchronize();

    //****************** getDensity ***********************
    getDensity<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_mass,
                                        d_rho, d_h, my_pi, N);
    cudaDeviceSynchronize();

    //****************** getAcc_g *************************
    acc_g<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_eps, d_accx, d_accy, d_accz,
                                  d_mass, G, N);
    cudaDeviceSynchronize();

    //****************** getPressure **********************
    getPressure_Adiabatic<<<gridSize, blockSize>>>(d_P, d_rho, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** getCsound ************************
    getCsound_Adiabatic<<<gridSize, blockSize>>>(d_csnd, d_u, gammah, N);
    cudaDeviceSynchronize();

    //****************** div_curlV ************************
    div_curlVel<<<gridSize, blockSize>>>(d_divV, d_curlV, d_x, d_y, d_z, d_vx, d_vy, d_vz,
                                        d_rho, d_mass, d_h, my_pi, N);
    cudaDeviceSynchronize();

    //****************** acc_sph ************************** 
    acc_sph<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                    d_divV, d_curlV, d_mass, d_P, d_accx_sph, d_accy_sph,
                                    d_accz_sph, my_pi, visc_alpha, N);
    cudaDeviceSynchronize();

    //****************** acc_tot **************************
    acc_g_sph<<<gridSize, blockSize>>>(d_accx_tot, d_accy_tot, d_accz_tot,
                                      d_accx, d_accy, d_accz,
                                      d_accx_sph, d_accy_sph, d_accz_sph, N);
    cudaDeviceSynchronize();

    //****************** velocity evolution *******************
    v_evolve<<<gridSize, blockSize>>>(d_vx, d_vy, d_vz, d_accx_tot, d_accy_tot,
                                      d_accz_tot, dt, N);
    cudaDeviceSynchronize();

    //******************** get_dU (du_dt) *********************
    get_dU<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_csnd, d_rho,
                                    d_divV, d_curlV, d_mass, d_P, d_dudt,
                                    my_pi, visc_alpha, N);
    cudaDeviceSynchronize();

    //******************** u evolution *********************
    u_updater_main<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                            d_utprevious, dt, N);
    cudaDeviceSynchronize();

    //******************** applyCooling *********************
        
    /*
    float *uBeforeCooling = new float[N];
    
    cudaMemcpy(u, d_u, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int k = 0; k < N; k++)
    {
      uBeforeCooling[k] = u[k];
    }
    */
    
    current_dt_cgs = dt * unitTime_in_s;
    applyCooling<<<gridSize, blockSize>>>(d_uadT, d_rhoT, d_delta_u,
                                          ref_dt_cgs, d_uGrid, d_rhoGrid,
                                          MIN_uad, MAX_uad, MIN_rho,
                                          MAX_rho, Unit_u_in_cgs,
                                          UnitDensity_in_cgs, d_u, d_rho,
                                          current_dt_cgs, N, N_u, N_rho, NGrid);
    cudaDeviceSynchronize();

    

    if(!(counter % 15)){
      cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaMemcpy(vx, d_vx, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vy, d_vy, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(vz, d_vz, N*sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaMemcpy(rho, d_rho, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h, d_h, N*sizeof(float), cudaMemcpyDeviceToHost);
      
      cudaMemcpy(u, d_u, N*sizeof(float), cudaMemcpyDeviceToHost);

      ofstream outfile("./Outputs/G-"+ to_string(t*1000) + ".csv");
      
      outfile << "x" << "," << "y" << "," << "z" << "," << "vx" << "," << "vy" << ","
              << "vz" << "," << "h" << "," << "rho" << "," << "u" << endl; // this will be the header !

      for(int i = 0; i < N; i++){
        outfile << x[i] << "," << y[i] << "," << z[i] << ","
                << vx[i] << "," << vy[i] << "," << vz[i] << ","
                << h[i] << "," << rho[i] << "," << u[i] << endl;
      }
    }
    
    
    //******* updating uprevious, utprevious ********
    u_ut_previous_updater<<<gridSize, blockSize>>>(d_u, d_dudt, d_uprevious,
                                                   d_utprevious, N);
    cudaDeviceSynchronize();
    
    dt_array<<<gridSize, blockSize>>>(d_accx, d_accy, d_accz, d_accx_tot,
                                      d_accy_tot, d_accz_tot, d_h, d_csnd,
                                      d_abs_acc_g, d_abs_acc_tot, d_v_sig,
                                      d_divV, d_dh_dt, N);
    cudaDeviceSynchronize();

    cudaMemcpy(abs_acc_g, d_abs_acc_g, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(abs_acc_tot, d_abs_acc_tot, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_sig, d_v_sig, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dh_dt, d_dh_dt, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h, d_h, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(divV, d_divV, N*sizeof(float), cudaMemcpyDeviceToHost);

    v_signal = max_finder(v_sig, N);
    min_h = min_finder(h, N);
    dt_cfl = C_CFL * min_h / v_signal;

    max_abs_acc_g = max_finder(abs_acc_g, N);
    dt_f = sqrt(min_h/max_abs_acc_g);

    max_abs_acc_tot = max_finder(abs_acc_tot, N);
    dt_kin = sqrt(min_h/max_abs_acc_tot);

    min_h_dh_dt = min_finder(dh_dt, N);
    dt_dens = C_CFL * min_h_dh_dt;

    dtz[0] = dt_f; dtz[1] = dt_kin; dtz[2] = dt_cfl; dtz[3] = dt_dens;
    dt = 0.25 * min_finder(dtz, 4);   // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if(!(counter % 5)){
      cout << "v_signal, min_h, C_CFL = " << v_signal<<" "<<min_h << " "<< C_CFL<<endl;
      cout << "dt_f = " << dt_f << endl;
      cout << "dt_kin = " << dt_kin << endl;
      cout << "dt_cfl = " << dt_cfl << endl;
      cout << "dt_dens = " << dt_dens << endl;
      cout << endl;
    }

    if(dt > MAX_dt_code_unit){
      dt = MAX_dt_code_unit;
    }

    t += dt;

    if(!(counter % 5)){
      cout << "Adopted dt = " << dt << endl;
      cout << "current t = " << t << endl;
      cout << "*****************************" << endl;
      cout << endl;
    }

    counter ++;

  }

  delete[] x; delete[] y; delete[] z; delete[] vx; delete[] vy; delete[] vz;
  delete[] mass; delete[] h; delete[] hprevious; delete[] rho;
  delete[] accx; delete[] accy; delete[] accz; delete[] eps;
  delete[] P; delete[] csnd; delete[] divV; delete[] curlV;
  delete[] accx_sph; delete[] accy_sph; delete[] accz_sph;
  delete[] accx_tot; delete[] accy_tot; delete[] accz_tot;
  delete[] abs_acc_g; delete[] abs_acc_tot; delete[] v_sig;
  delete[] dh_dt; delete[] u; delete[] dudt; delete[] uprevious;
  delete[] utprevious; delete[] uGrid; delete[] rhoGrid;
  delete[] uadT; delete[] rhoT; delete[] delta_u;

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
  cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
  cudaFree(d_mass); cudaFree(d_h); cudaFree(d_hprevious);
  cudaFree(d_rho); cudaFree(d_accx); cudaFree(d_accy); cudaFree(d_accz);
  cudaFree(d_P); cudaFree(d_csnd); cudaFree(d_divV); cudaFree(d_curlV);
  cudaFree(d_accx_sph); cudaFree(d_accy_sph); cudaFree(d_accz_sph);
  cudaFree(d_accx_tot); cudaFree(d_accy_tot); cudaFree(d_accz_tot);
  cudaFree(d_abs_acc_g); cudaFree(d_abs_acc_tot); cudaFree(d_v_sig);
  cudaFree(d_dh_dt); cudaFree(d_u); cudaFree(d_dudt); cudaFree(d_uprevious);
  cudaFree(d_utprevious); cudaFree(d_uGrid); cudaFree(d_rhoGrid);
  cudaFree(d_uadT); cudaFree(d_rhoT); cudaFree(d_delta_u);

}
