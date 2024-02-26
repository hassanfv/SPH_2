
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <tuple>

using namespace std;



float hcoolingZ(float u, float *uArr, float rho, float *metalz, float Zmetal, float dt, // Zmetal is the gas metallicity assumed.
               float *nHGrid, float *ZGrid, float *uEvol, float *ionFrac, float x, float y, float z,
               float *muA, float *logT, float *kpcGrid, float unit_density_in_cgs, float unit_time_in_sec, 
               float unit_u_in_cgs, float unitLength_in_cm, float kpc_in_cm, float GAMMA_MINUS1,
               float *tGrid, int N_kpc, int N_nH, int N_Z, int N_T, int N_M, int N_t, int N)
{

  int ndx_kpc = -1;
  int ndx_nH = -1;
  int ndx_u = -1;
  int ndx_t = -1;
  
  float delta1 = 0.0f;
  float delta2 = 0.0f;
  
  float uEv = 0.0f;
  
  int index = 0;
  
  float dt_sec = dt * unit_time_in_sec;
  
  //======== kpc =========
  float r_i = sqrt(x * x + y * y + z * z);
  r_i = r_i * unitLength_in_cm / kpc_in_cm; // converting r_i from code unit to kpc!
  
  for (int j = 0; j < N_kpc; j++)
  {
    if (ndx_kpc == -1 && r_i <= kpcGrid[j])
    {
      ndx_kpc = j;
    }
  }
  
  if (ndx_kpc > 0) 
  {
    delta1 = abs(kpcGrid[ndx_kpc] - r_i);
    delta2 = abs(kpcGrid[ndx_kpc - 1] - r_i);
    
    if (delta2 < delta1)
    {
      ndx_kpc -= 1;
    }
  }
  
  if (ndx_kpc == -1) // This happens when r_i is greater than max(kpcGrid) !
  {
    ndx_kpc = N_kpc - 1;
  }
  
  //======== nH ==========
  //---- convert rho to nH
  float mH = 1.6726e-24;
  float XH = 0.70;
  float rho_cgs = rho * unit_density_in_cgs;
  float nH = rho_cgs * XH / mH;
  
  for (int j = 0; j < N_nH; j++)
  {
    if (ndx_nH == -1 && nH <= nHGrid[j])
    {
      ndx_nH = j;
    }
  }

  if (ndx_nH > 0) 
  {
    delta1 = abs(nHGrid[ndx_nH] - nH);
    delta2 = abs(nHGrid[ndx_nH - 1] - nH);
    
    if (delta2 < delta1)
    {
      ndx_nH -= 1;
    }
  }
  
  if (ndx_nH == -1) // This happens when nH is greater than max(nHGrid) !
  {
    ndx_nH = N_nH - 1;
  }

  //======== Z =========
  int ndx_Z = 1; // Assuming [Z/H] -1. Recall that Z = [-2, -1, 0].
  
  //======== t ========= time!
  for (int j = 0; j < N_t; j++)
  {
    if (ndx_t == -1 && dt_sec <= tGrid[j])
    {
      ndx_t = j;
    }
  }

  if (ndx_t > 0) 
  {
    delta1 = abs(tGrid[ndx_t] - dt_sec);
    delta2 = abs(tGrid[ndx_t - 1] - dt_sec);
    
    if (delta2 < delta1)
    {
      ndx_t -= 1;
    }
  }
  
  if (ndx_t == -1) // This happens when dt_sec is greater than max(tGrid) !
  {
    ndx_t = N_t - 1;
  }
  
  //========= u =========
  // Calculate the strides
  int stride_Z0 = 1;
  int stride_nH0 = N_Z * stride_Z0;
  int stride_T0 = N_nH * stride_nH0;
  int stride_kpc0 = N_T * stride_T0;
  
  float u_i = u * unit_u_in_cgs;
  
  for (int j = 0; j < N_T; j++)
  {
    index = (ndx_kpc * stride_kpc0) + (j * stride_T0) + (ndx_nH * stride_nH0) + ndx_Z; // uArr values are from t = 0! so no need for ndx_t!
    if (ndx_u == -1 && u_i <= uArr[index])
    {
      ndx_u = j;
    }
  }
  
  if (ndx_u == -1) // This happens when u_1 is greater than max(U) !
  {
    ndx_u = N_T - 1;
  }

  // Calculate the strides
  int stride_t = 1;
  int stride_Z = N_t * stride_t;
  int stride_nH = N_Z * stride_Z;
  int stride_T = N_nH * stride_nH;
  int stride_kpc = N_T * stride_T;
  
  int ndx = (ndx_kpc * stride_kpc) + (ndx_u * stride_T) + (ndx_nH * stride_nH) + (ndx_Z * stride_Z) + ndx_t;
  
  if (ndx_u > 0)
  {
    int ndx_minus1 = (ndx_kpc * stride_kpc) + ((ndx_u-1) * stride_T) + (ndx_nH * stride_nH) + (ndx_Z * stride_Z) + ndx_t;
    float uEv_1 = uEvol[ndx_minus1];
    float uEv_2 = uEvol[ndx];
    
    int index_U = (ndx_kpc * stride_kpc0) + (ndx_u * stride_T0) + (ndx_nH * stride_nH0) + ndx_Z;
    int index_U_minus1 = (ndx_kpc * stride_kpc0) + ((ndx_u - 1) * stride_T0) + (ndx_nH * stride_nH0) + ndx_Z;
    
    float diff = uArr[index_U] - uArr[index_U_minus1];
    float fhi = (u_i - uArr[index_U_minus1]) / diff;
    float flow = 1.0 - fhi;
    
    uEv = flow * uEv_1 + fhi * uEv_2;      
  }
  else
  {
    uEv = uEvol[ndx];
  }
  
  
  float u_after_hcooling = uEv / unit_u_in_cgs; // uEv is the u after heating and/or cooling!

  // Calculate the strides for Metalz
  int stride_tx = 1; // stride for t dimension
  int stride_Mx = N_t * stride_tx;
  int stride_Zx = N_M * stride_Mx;
  int stride_nHx = N_Z * stride_Zx;
  int stride_Tx = N_nH * stride_nHx;
  int stride_kpcx = N_T * stride_Tx;
  
  for (int k = 0; k < N_M; k++)
  {
    int ndx_M = (ndx_kpc * stride_kpcx) + (ndx_u * stride_Tx) + (ndx_nH * stride_nHx) + (ndx_Z * stride_Zx) + (k * stride_Mx) + (ndx_t * stride_tx);
    
    ionFrac[k] = metalz[ndx_M]; // k is species index! Here we use flattened version of a 2D array of ionFrac(N, N_M) dimension
  }
  
  return u_after_hcooling;
  
}



//******************************************
//********** Reading params.txt ************
//******************************************

void readParams(std::string &filename, int &N_tot, int &ndx_BH, float &G, float &L_AGN_code_unit,
                float &M_dot_in_code_unit, float &vin_in_code_unit,
                float &u_for_10K_Temp, float &m_sph_high_res, float &sigma,
                float &UnitDensity_in_cgs, float &Unit_u_in_cgs, float &unitTime_in_s,
                float &unitLength_in_cm)
{
  std::ifstream file("params.txt");
  if (file.is_open())
  {
    std::getline(file, filename); // Read filename string
    file >> N_tot;                // Read N_tot
    file >> ndx_BH;               // Read ndx_BH
    file >> G;                    // Read G
    file >> L_AGN_code_unit;      // Read L_AGN_code_unit
    file >> M_dot_in_code_unit;   // Read M_dot_in_code_unit
    file >> vin_in_code_unit;     // Read vin_in_code_unit
    file >> u_for_10K_Temp;       // Read u_for_10K_Temp
    file >> m_sph_high_res;       // Read m_sph_high_res
    file >> sigma;                // Read sigma
    file >> UnitDensity_in_cgs;                // Read UnitDensity_in_cgs
    file >> Unit_u_in_cgs;                // Read Unit_u_in_cgs
    file >> unitTime_in_s;                // Read unitTime_in_s
    file >> unitLength_in_cm;                // Read unitLength_in_cm
  }
  else
  {
    std::cout << "Unable to open params.txt file";
  }
  file.close();
}


const float kpc_in_cm = 3.086e21;
const float gammah = 5.0f / 3.0f;
const float GAMMA_MINUS1 = gammah - 1.0f;
const float Zmetal = 0.1; // ==> [Z/H] = -1.


int main()
{


  float dt = 2e-7; // in code unit!


  //*******************************************************************
  //******************* Reading Cooling File **************************
  //*******************************************************************
  ifstream file("coolHeatGridJan2024FineGrid.bin", ios::binary);

  if (!file) {
    cerr << "Failed to open coolHeatGridJan2024FineGrid.bin file." << endl;
    return 1;
  }

  // Read dimensions
  int N_kpc, N_nH, N_Z, N_T, N_M, N_t;
  file.read(reinterpret_cast<char*>(&N_kpc), sizeof(N_kpc));
  file.read(reinterpret_cast<char*>(&N_T), sizeof(N_T));
  file.read(reinterpret_cast<char*>(&N_nH), sizeof(N_nH));
  file.read(reinterpret_cast<char*>(&N_Z), sizeof(N_Z));
  file.read(reinterpret_cast<char*>(&N_M), sizeof(N_M));
  file.read(reinterpret_cast<char*>(&N_t), sizeof(N_t));

  // Allocate memory for 1D arrays
  float* kpcsF = new float[N_kpc];
  float* densities = new float[N_nH];
  float* metallicities = new float[N_Z];
  float* temperatures = new float[N_T];
  float* tArrX = new float[N_t];

  float* uEvolutionX = new float[N_kpc * N_T * N_nH * N_Z * N_t];
  float* muArrX = new float[N_kpc * N_T * N_nH * N_Z * N_t];
  float* metalzX = new float[N_kpc * N_T * N_nH * N_Z * N_M * N_t];
  float* uArrX = new float[N_kpc * N_T * N_nH * N_Z];

  // Read data into arrays
  file.read(reinterpret_cast<char*>(kpcsF), N_kpc * sizeof(float));
  file.read(reinterpret_cast<char*>(densities), N_nH * sizeof(float));
  file.read(reinterpret_cast<char*>(metallicities), N_Z * sizeof(float));
  file.read(reinterpret_cast<char*>(temperatures), N_T * sizeof(float));
  file.read(reinterpret_cast<char*>(tArrX), N_t * sizeof(float));
  
  file.read(reinterpret_cast<char*>(uEvolutionX), N_kpc * N_T * N_nH * N_Z * N_t * sizeof(float));
  file.read(reinterpret_cast<char*>(muArrX), N_kpc * N_T * N_nH * N_Z * N_t * sizeof(float));
  file.read(reinterpret_cast<char*>(metalzX), N_kpc * N_T * N_nH * N_Z * N_M * N_t * sizeof(float));
  file.read(reinterpret_cast<char*>(uArrX), N_kpc * N_T * N_nH * N_Z * sizeof(float));

  file.close();
  
  //------------- Just for testing ---------
  int stride_t = 1;
  int stride_Z = N_t * stride_t;
  int stride_nH = N_Z * stride_Z;
  int stride_T = N_nH * stride_nH;
  int stride_kpc = N_T * stride_T;
  
  int ndx_kpc =1;
  int ndx_u = 29;
  int ndx_nH = 80;
  int ndx_Z = 1;
  int ndx_t = 3;
  
  cout << "kpc = " << kpcsF[ndx_kpc] << endl;
  cout << "nH = " << densities[ndx_nH] << " cm^-3" << endl;
  cout << "T = " << pow(10.0f, temperatures[ndx_u]) << " K" << endl;
  cout << "time [yrs] = " << tArrX[ndx_t]/3600.0/24.0/365.25 << " yrs" << endl;
  cout << endl;
  
  int ndx0 = (ndx_kpc * stride_kpc) + (ndx_u * stride_T) + (ndx_nH * stride_nH) + (ndx_Z * stride_Z) + 0;
  int ndx1 = (ndx_kpc * stride_kpc) + (ndx_u * stride_T) + (ndx_nH * stride_nH) + (ndx_Z * stride_Z) + ndx_t;
  cout << "u befor evolution = " << uEvolutionX[ndx0] << endl;
  cout << "u after evolving " << tArrX[ndx_t]/3600.0/24.0/365.25 << " yrs = " << uEvolutionX[ndx1] << endl;
  cout << "mu (after time yrs passed) = " << muArrX[ndx1] << endl;
  
  //---------
  int stride_tx = 1; // stride for t dimension
  int stride_Mx = N_t * stride_tx;
  int stride_Zx = N_M * stride_Mx;
  int stride_nHx = N_Z * stride_Zx;
  int stride_Tx = N_nH * stride_nHx;
  int stride_kpcx = N_T * stride_Tx;
  
  int ndx_M = 0; // corresponding to HI !
  int ndx = (ndx_kpc * stride_kpcx) + (ndx_u * stride_Tx) + (ndx_nH * stride_nHx) + (ndx_Z * stride_Zx) + (ndx_M * stride_Mx) + (ndx_t * stride_tx);
  cout << "HI fraction = " << metalzX[ndx] << endl;
  cout << endl;

  //--------------------------------------------
  
  float *Temp, *uArr, *nH, *Z, *HCool, *muA, *kpc, *metalz, *tArr;
  
  int N_HCool = N_kpc * N_T * N_nH * N_Z * N_t;
  int N_metalz = N_kpc * N_T * N_nH * N_Z * N_M * N_t;
  
  kpc = new float[N_kpc];
  Temp = new float[N_T];
  nH = new float[N_nH];
  Z = new float[N_Z];
  tArr = new float[N_t];
  
  HCool = new float[N_HCool];
  muA = new float[N_HCool];
  metalz = new float[N_metalz];
  uArr = new float[N_kpc * N_T * N_nH * N_Z];
  
  for (int i = 0; i < N_kpc; i++)
  {
    kpc[i] = kpcsF[i];
  }
  
  for (int i = 0; i < N_nH; i++)
  {
    nH[i] = densities[i];
  }
  
  for (int i = 0; i < N_Z; i++)
  {
    Z[i] = metallicities[i];
  }
  
  for (int i = 0; i < N_t; i++)
  {
    tArr[i] = tArrX[i];
  }

  for (int i = 0; i < N_HCool; i++)
  {
    HCool[i] = uEvolutionX[i];
  }
  
  for (int i = 0; i < N_HCool; i++)
  {
    muA[i] = muArrX[i];
  }
  
  for (int i = 0; i < (N_kpc * N_T * N_nH * N_Z); i++)
  {
    uArr[i] = uArrX[i];
  }

  for (int i = 0; i < N_metalz; i++)
  {
    metalz[i] = metalzX[i];
  }


  
  //********************************************************************
  //**************** Reading the params.txt file ***********************
  //********************************************************************
  std::string filename;
  int N, ndx_BH;
  float GG, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
        unitLength_in_cm;

  readParams(filename, N, ndx_BH, GG, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
             unitLength_in_cm);

  std::cout << "filename: " << filename << "\n";
  std::cout << "N: " << N << "\n";
  std::cout << "ndx_BH: " << ndx_BH << "\n";
  std::cout << "GG: " << GG << "\n";
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
  
  cout << endl;
  cout << "******************************************" << endl;
  cout << "******************************************" << endl;
  cout << "Addopted Time-step: dt_yrs = " << dt * unitTime_in_s / 3600.0 / 24.0 / 365.25 << " yrs" << endl;
  cout << "******************************************" << endl;
  cout << "******************************************" << endl;

  float *ionFrac = new float[N_M];
  
  float u = 0.109076; // in code unit!
  float rho = 1021.2034; // in code unit!

  float x = -0.2270;  // in code unit!
  float y = -0.0165; // in code unit!
  float z = -0.0363; // in code unit!
  
  float uAhcooling = 0.0f;
  uAhcooling = hcoolingZ(u, uArr, rho, metalz, Zmetal, dt, // Zmetal is the gass metallicity assumed.
                         nH, Z, HCool, ionFrac, x, y, z,
                         muA, Temp, kpc, UnitDensity_in_cgs, unitTime_in_s, 
                         Unit_u_in_cgs, unitLength_in_cm, kpc_in_cm, GAMMA_MINUS1,
                         tArr, N_kpc, N_nH, N_Z, N_T, N_M, N_t, N);


  cout << endl;
  cout << "u before hcooling = " << u << endl;
  cout << "u After hcooling = " << uAhcooling << endl;

  cout << endl;
  for (int i = 0; i < N_M; i++)
  {
    cout << "ionFrac = " << ionFrac[i] << endl;
  }


}




