#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include "myCppSPHLibs.h"
using namespace std;


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

  float UnitRadius_in_cm = Rcld_in_cm;
  float UnitRadius_in_cm_2 = UnitRadius_in_cm*UnitRadius_in_cm;

  float UnitMass_in_g = Mcld_in_g;
  //--------------------- To avoid getting inf ! --------
  float UnitDensity_in_cgs = UnitMass_in_g / UnitRadius_in_cm_2;
  UnitDensity_in_cgs = UnitDensity_in_cgs / UnitRadius_in_cm;
  //-------------------------
  float Unit_u_in_cgs = grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm;
  float Unit_P_in_cgs = UnitDensity_in_cgs * Unit_u_in_cgs;
  float unitVelocity = sqrt(grav_const_in_cgs * UnitMass_in_g / UnitRadius_in_cm);

  // Reading Hydra file.
  string fname = "GPU_IC_Kitsonias_600k_RAND.csv"; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

  // declaring the arrays.
  float *x, *d_x, *y, *d_y, *z, *d_z, *vx, *d_vx, *vy, *d_vy, *vz, *d_vz;
  float *mass, *d_mass, *h, *d_h, *hprevious, *d_hprevious, *rho, *d_rho;
  float *accx, *accy, *accz, *d_accx, *d_accy, *d_accz, *eps, *d_eps;
  float *P, *d_P, *csnd, *d_csnd, *divV, *d_divV, *curlV, *d_curlV;
  float *accx_sph, *accy_sph, *accz_sph, *d_accx_sph, *d_accy_sph, *d_accz_sph;
  float *accx_tot, *accy_tot, *accz_tot, *d_accx_tot, *d_accy_tot, *d_accz_tot;
  float *abs_acc_g, *abs_acc_tot, *v_sig, *dh_dt, *d_abs_acc_g, *d_abs_acc_tot;
  float *d_v_sig, *d_dh_dt;

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
    h[i] = 10.0f; // place holder.
    rho[i] = 11.0f; // place holder.
    P[i] = 12.0f; // placeholder.
    csnd[i] = 13.0f; // placeholder.

    divV[i] = 14.0f; // placeholder.
    curlV[i] = 15.0f; // placeholder.

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

  int blockSize = 256; // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid
  
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.001; // used for smoothing length.

  const float visc_alpha = 1.0f;

  float mH = 1.6726e-24; // gram
  float kB = 1.3807e-16; // cm2 g s-2 K-1
  float mH2 = muu * mH;

  float kBmH2 = kB/mH2;

  float T_cld = 377.0f;
  float T_ps = 10.0f;
  float T_0 = 10.0f;

  const float my_pi = 3.141592f;

  float t = 0.0f;
  float dt = 2e-4;
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

  //getPressure<<<gridSize, blockSize>>>(d_P, d_rho, T_cld, T_ps, T_0, kBmH2,
  //                                     UnitDensity_in_cgs, Unit_P_in_cgs,
  //                                     gammah, N);
  getPressure_Kitsonias<<<gridSize, blockSize>>>(d_P, d_rho, UnitDensity_in_cgs, 
                                                 Unit_P_in_cgs, gammah,
                                                 N);
  cudaDeviceSynchronize();

  //-----------------------------------------------
  //----------------- getCsound -------------------
  //-----------------------------------------------

  //getCsound<<<gridSize, blockSize>>>(d_csnd, d_rho, T_cld, T_ps, T_0, kBmH2,
  //                                     UnitDensity_in_cgs, unitVelocity,
  //                                     gammah, N);
  getCsound_Kitsonias<<<gridSize, blockSize>>>(d_csnd, d_rho, UnitDensity_in_cgs,
                                               unitVelocity, gammah, N);
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
    //getPressure<<<gridSize, blockSize>>>(d_P, d_rho, T_cld, T_ps, T_0, kBmH2,
    //                                    UnitDensity_in_cgs, Unit_P_in_cgs,
    //                                    gammah, N);
    getPressure_Kitsonias<<<gridSize, blockSize>>>(d_P, d_rho, UnitDensity_in_cgs, 
                                                 Unit_P_in_cgs, gammah,
                                                 N);
    cudaDeviceSynchronize();

    //****************** getCsound ************************
    //getCsound<<<gridSize, blockSize>>>(d_csnd, d_rho, T_cld, T_ps, T_0, kBmH2,
    //                                    UnitDensity_in_cgs, unitVelocity,
    //                                    gammah, N);
    getCsound_Kitsonias<<<gridSize, blockSize>>>(d_csnd, d_rho, UnitDensity_in_cgs,
                                               unitVelocity, gammah, N);
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

    counter ++;

    if(!(counter % 25)){
      cout << "t = " << t << endl;
    }

    if(!(counter % 25)){
      cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(z, d_z, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(rho, d_rho, N*sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h, d_h, N*sizeof(float), cudaMemcpyDeviceToHost);

      ofstream outfile("G-"+ to_string(t) + ".csv");
      
      outfile << "x" << "," << "y" << "," << "z" << "," << "h" << ","
              << "rho" << endl; // this will be the header !

      for(int i = 0; i < N; i++){
        outfile << x[i] << "," << y[i] << "," << z[i] << "," << h[i] << "," 
                << rho[i] << endl;
      }
    }
    
    dt_array<<<gridSize, blockSize>>>(d_accx, d_accy, d_accz, d_accx_tot,
                                      d_accy_tot, d_accz_tot, d_h, d_csnd,
                                      d_abs_acc_g, d_abs_acc_tot, d_v_sig,
                                      d_divV, d_dh_dt, N);

    cudaMemcpy(abs_acc_g, d_abs_acc_g, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(abs_acc_tot, d_abs_acc_tot, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v_sig, d_v_sig, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dh_dt, d_dh_dt, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h, d_h, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(divV, d_divV, N*sizeof(float), cudaMemcpyDeviceToHost);


    v_signal = max_finder(v_sig, N);
    min_h = min_finder(h, N);
    dt_cfl = C_CFL * min_h / v_signal;

    cout << "v_signal, min_h, C_CFL = " << v_signal<<" "<<min_h << " "<< C_CFL<<endl;

    max_abs_acc_g = max_finder(abs_acc_g, N);
    dt_f = sqrt(min_h/max_abs_acc_g);

    max_abs_acc_tot = max_finder(abs_acc_tot, N);
    dt_kin = sqrt(min_h/max_abs_acc_tot);

    min_h_dh_dt = min_finder(dh_dt, N);
    dt_dens = C_CFL * min_h_dh_dt;

    dtz[0] = dt_f; dtz[1] = dt_kin; dtz[2] = dt_cfl; dtz[3] = dt_dens;
    dt = 0.25 * min_finder(dtz, 4);

    cout << "dt_f = " << dt_f << endl;
    cout << "dt_kin = " << dt_kin << endl;
    cout << "dt_cfl = " << dt_cfl << endl;
    cout << "dt_dens = " << dt_dens << endl;

    /*
    if(dt > 0.0005){
      dt = 0.0005;
    }
    */

    if(dt < 0.00001){
      dt = 0.00001;
    }

    cout << "Adopted dt = " << dt << endl;
    cout << "current t = " << t << endl;
    cout << "*****************************" << endl;
    cout << endl;

    t += dt;

  }


  delete[] x; delete[] y; delete[] z; delete[] vx; delete[] vy; delete[] vz;
  delete[] mass; delete[] h; delete[] hprevious; delete[] rho;
  delete[] accx; delete[] accy; delete[] accz; delete[] eps;
  delete[] P; delete[] csnd; delete[] divV; delete[] curlV;
  delete[] accx_sph; delete[] accy_sph; delete[] accz_sph;
  delete[] accx_tot; delete[] accy_tot; delete[] accz_tot;
  delete[] abs_acc_g; delete[] abs_acc_tot; delete[] v_sig;
  delete[] dh_dt;

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
  cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
  cudaFree(d_mass); cudaFree(d_h); cudaFree(d_hprevious);
  cudaFree(d_rho); cudaFree(d_accx); cudaFree(d_accy); cudaFree(d_accz);
  cudaFree(d_P); cudaFree(d_csnd); cudaFree(d_divV); cudaFree(d_curlV);
  cudaFree(d_accx_sph); cudaFree(d_accy_sph); cudaFree(d_accz_sph);
  cudaFree(d_accx_tot); cudaFree(d_accy_tot); cudaFree(d_accz_tot);
  cudaFree(abs_acc_g); cudaFree(abs_acc_tot); cudaFree(v_sig);
  cudaFree(dh_dt);

}
