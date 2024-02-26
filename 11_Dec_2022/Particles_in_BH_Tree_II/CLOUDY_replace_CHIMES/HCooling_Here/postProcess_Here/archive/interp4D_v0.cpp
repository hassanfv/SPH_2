
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;


//===== interpolate_4d_hypercube
float interpolate_4d_hypercube(float nH_p, float T_p, float r_p, float NH_p, float *nH, float *T, float *r, float *NH,
                               float *Gam, float *Lam, int nxnH0, int nxT0, int nxr0, int nxNH0, int nxnH1, int nxT1, int nxr1, int nxNH1,
                               int N_nH, int N_T, int N_r, int N_NH)
{

  float dx = (nH_p - nH[nxnH0]) / (nH[nxnH1] - nH[nxnH0]);
  float dy = (T_p - T[nxT0]) / (T[nxT1] - T[nxT0]);
  float dz = (r_p - r[nxr0]) / (r[nxr1] - r[nxr0]);
  float dw = (NH_p - NH[nxNH0]) / (NH[nxNH1] - NH[nxNH0]);
  
  cout << "dx, dy, dz, dw = " << dx << ", " << dy << ", " << dz << ", " << dw << endl;
  cout << "nH = " << nH_p << ", " << nH[nxnH0] << ", " << nH[nxnH1] << ", " << nH[nxnH0] << endl;
  cout << "T = " << T_p << ", " << T[nxT0] << ", " << T[nxT1] << ", " << T[nxT0] << endl;
  cout << "r = " << r_p << ", " << r[nxr0] << ", " << r[nxr1] << ", " << r[nxr0] << endl;
  cout << "NH = " << NH_p << ", " << NH[nxNH0] << ", " << NH[nxNH1] << ", " << NH[nxNH0] << endl << endl;
  cout << "nxnH0 = " << nxnH0 << endl; 
  cout << "nxT0 = " << nxT0 << endl; 
  cout << "nxr0 = " << nxr0 << endl; 
  cout << "nxNH0 = " << nxNH0 << endl << endl; 
  
  float pG[16];
  float pL[16];
  
  int stride_nH = N_T * N_r * N_NH;
  int stride_T = N_r * N_NH;
  int stride_r = N_NH;
  
  int nxs = 0;
  for (int i = nxnH0; i <= nxnH1; i++)
  {
    for (int j = nxT0; j <= nxT1; j++)
    {
      for (int k = nxr0; k <= nxr1; k++)
      {
        for (int l = nxNH0; l<= nxNH1; l++)
        {
          pG[nxs] = Gam[i * stride_nH + j * stride_T + k * stride_r + l]; // heating!
          pL[nxs] = Lam[i * stride_nH + j * stride_T + k * stride_r + l]; // cooling!
          nxs += 1;
        }
      }
    }
  }
  
  //-------------> HEATING <---------------
  // Interpolate along the x-axis
  float c00 = pG[0] * (1.0f - dx) + pG[8] * dx;
  float c01 = pG[1] * (1.0f - dx) + pG[9] * dx;
  float c10 = pG[2] * (1.0f - dx) + pG[10] * dx;
  float c11 = pG[3] * (1.0f - dx) + pG[11] * dx;
  float c20 = pG[4] * (1.0f - dx) + pG[12] * dx;
  float c21 = pG[5] * (1.0f - dx) + pG[13] * dx;
  float c30 = pG[6] * (1.0f - dx) + pG[14] * dx;
  float c31 = pG[7] * (1.0f - dx) + pG[15] * dx;

  // Correct interpolation along the y-axis
  float c0 = c00 * (1.0f - dy) + c10 * dy;
  float c1 = c01 * (1.0f - dy) + c11 * dy;
  float c2 = c20 * (1.0f - dy) + c30 * dy;
  float c3 = c21 * (1.0f - dy) + c31 * dy;

  // Interpolate along the z-axis
  float c = c0 * (1.0f - dz) + c2 * dz;
  float d = c1 * (1.0f - dz) + c3 * dz;

  // Finally, interpolate along the w-axis
  float GamH = c * (1.0f - dw) + d * dw;
  
  //-------------> COOLING <---------------
  // Interpolate along the x-axis
  c00 = pL[0] * (1.0f - dx) + pL[8] * dx;
  c01 = pL[1] * (1.0f - dx) + pL[9] * dx;
  c10 = pL[2] * (1.0f - dx) + pL[10] * dx;
  c11 = pL[3] * (1.0f - dx) + pL[11] * dx;
  c20 = pL[4] * (1.0f - dx) + pL[12] * dx;
  c21 = pL[5] * (1.0f - dx) + pL[13] * dx;
  c30 = pL[6] * (1.0f - dx) + pL[14] * dx;
  c31 = pL[7] * (1.0f - dx) + pL[15] * dx;

  // Correct interpolation along the y-axis
  c0 = c00 * (1.0f - dy) + c10 * dy;
  c1 = c01 * (1.0f - dy) + c11 * dy;
  c2 = c20 * (1.0f - dy) + c30 * dy;
  c3 = c21 * (1.0f - dy) + c31 * dy;

  // Interpolate along the z-axis
  c = c0 * (1.0f - dz) + c2 * dz;
  d = c1 * (1.0f - dz) + c3 * dz;

  // Finally, interpolate along the w-axis
  float LamC = c * (1.0f - dw) + d * dw;

  cout << "Gam = " << GamH << endl;
  cout << "Lam = " << LamC << endl;

  return pow(10, GamH) - pow(10, LamC);
}


//===== doCloudyHCooling
void doCloudyHCooling(float nH_p, float u_p, float r_p, float NH_p, float *nH, float *T, float *r, float *NH, float *uarr, float *muarr,
                      float *Gam, float *Lam, int N_nH, int N_T, int N_r, int N_NH, float gamma, float kB, float mH)
{
  /*
  uarr is what we get by converting T to u using mu from the main data for which Gam and Lam are located. So uarr and muarr have the
  same len Gam and Lam! But nH, T, r, and NH are the grid with the step of 0.1 and/or 0.2 that we defined in "createInputList.py".
  The distinction between these arrays is very important in order to understand the functionality of these codes!
  */

  //----> nH <----
  int nxnH0 = -1;
  for (int j = 0; j < N_nH; j++)
  {
    if (nxnH0 == -1 && nH_p >= nH[j]-0.01 && nH_p <= nH[j]+0.01)
      nxnH0 = j;
  }
  
  if (nxnH0 > 0) // To handle cases in which nH_p <= min(nH)
    nxnH0 = nxnH0 - 1;

  if (nxnH0 == -1) // To handle cases in which nH_p >= max(nH)
    nxnH0 = N_nH - 2; // 1 will later be added to nxnH0 to make nxnH1!


  //----> rkpc <----
  int nxr0 = -1;
  for (int j = 0; j < N_r; j++)
  {
    if (nxr0 == -1 && r_p >= r[j]-0.01 && r_p <= r[j]+0.01)
      nxr0 = j;
  }
  
  if (nxr0 > 0) // To handle cases in which r_p <= min(r)
    nxr0 = nxr0 - 1;

  if (nxr0 == -1) // To handle cases in which r_p >= max(r)
    nxr0 = N_r - 2;


  //----> NH <----
  int nxNH0 = -1;
  for (int j = 0; j < N_NH; j++)
  {
    if (nxNH0 == -1 && NH_p >= (NH[j]-0.01) && NH_p <= (NH[j]+0.01))
      nxNH0 = j;
  }
  
  if (nxNH0 > 0) // To handle cases in which NH_p < min(NH)
    nxNH0 = nxNH0 - 1;

  if (nxNH0 == -1) // To handle cases in which NH_p > max(NH)
    nxNH0 = N_NH - 2;


  //----> T <----
  // First we need to convert u_p to T_p. For this, we need to have its mu, i.e. mu_p. So first we find mu_p!
  //----> Finding mu_p <----
  u_p = log10(u_p);
  
  int S1 = N_T * N_r * N_NH;
  int S2 = N_r * N_NH;
  int S3 = N_NH;

  int nxnH1 = nxnH0 + 1;
  int nxr1 = nxr0 + 1;
  int nxNH1 = nxNH0 + 1;

  int ndx = 0;
  int nxu0 = -1;
  for (int j = 0; j < N_T; j++)
  {
    ndx = (nxnH1 * S1) + (j * S2) + (nxr1 * S3) + nxNH1;
    if (nxu0 == -1  && u_p <= uarr[ndx])
        nxu0 = ndx;
  }
  
  float mu_p = muarr[nxu0];
  cout << "mu_p = " << mu_p << endl;
  
  float T_p = (gamma - 1.0f) * mu_p * mH / kB * pow(10, u_p);
  T_p = log10(T_p);
  
  cout << "T_p = " << T_p << endl;
  
  //--- Now that we have T_p, we proceed to find nxT0!
  int nxT0 = -1;
  for (int j = 0; j < N_T; j++)
  {
    if (nxT0 == -1 && T_p >= (T[j]-0.01) && T_p <= (T[j]+0.01))
      nxT0 = j;
  }

  if (nxT0 > 0) // To handle cases in which T_p <= min(T)
    nxT0 = nxT0 - 1;

  if (nxT0 == -1) // To handle cases in which T_p >= max(T)
    nxT0 = N_T - 2;

  int nxT1 = nxT0 + 1;

  float Gam_minus_Lam = interpolate_4d_hypercube(nH_p, T_p, r_p, NH_p, nH, T, r, NH, Gam, Lam, nxnH0, nxT0, nxr0, nxNH0,
                                                 nxnH1, nxT1, nxr1, nxNH1, N_nH, N_T, N_r, N_NH);

}





int main()
{

  float kB = 1.3807e-16;
  float mH = 1.673534e-24;
  float gamma = 5.0f/3.0f;

  std::ifstream bin_file("HCoolMu.bin", std::ios::binary);

  // Read scalar values
  int N_nH, N_T, N_r, N_NH;
  bin_file.read(reinterpret_cast<char*>(&N_nH), sizeof(N_nH));
  bin_file.read(reinterpret_cast<char*>(&N_T), sizeof(N_T));
  bin_file.read(reinterpret_cast<char*>(&N_r), sizeof(N_r));
  bin_file.read(reinterpret_cast<char*>(&N_NH), sizeof(N_NH));

  // Allocate memory for arrays using float
  float* nHGrid = new float[N_nH];
  float* TGrid = new float[N_T];
  float* rGrid = new float[N_r];
  float* NHGrid = new float[N_NH];
  float* Heat = new float[N_nH*N_T*N_r*N_NH];
  float* Cool = new float[N_nH*N_T*N_r*N_NH];
  float* muarr = new float[N_nH*N_T*N_r*N_NH];
  float* uarr = new float[N_nH*N_T*N_r*N_NH];

  // Read arrays from the binary file
  bin_file.read(reinterpret_cast<char*>(nHGrid), N_nH * sizeof(float));
  bin_file.read(reinterpret_cast<char*>(TGrid), N_T * sizeof(float));
  bin_file.read(reinterpret_cast<char*>(rGrid), N_r * sizeof(float));
  bin_file.read(reinterpret_cast<char*>(NHGrid), N_NH * sizeof(float));
  bin_file.read(reinterpret_cast<char*>(Heat), N_nH*N_T*N_r*N_NH * sizeof(float));
  bin_file.read(reinterpret_cast<char*>(Cool), N_nH*N_T*N_r*N_NH * sizeof(float));
  bin_file.read(reinterpret_cast<char*>(muarr), N_nH*N_T*N_r*N_NH * sizeof(float));
  bin_file.read(reinterpret_cast<char*>(uarr), N_nH*N_T*N_r*N_NH * sizeof(float));

  // Example usage
  cout << "First element of nHGrid: " << nHGrid[0] << endl;

  //2.4,0.3,19.4,2.9

  float nH_p = 2.43;
  float T_p = 2.9;
  float r_p = 0.3;
  float NH_p = 19.4;
  float mu_tmp = 0.6051;
  
  float u_p = kB * pow(10, T_p) / (gamma - 1.0) / mu_tmp / mH; // log10 of it will be taken inside the "doCloudyHCooling" function!
  
  cout << "u_p = " << u_p << endl;
  
  doCloudyHCooling(nH_p, u_p, r_p, NH_p, nHGrid, TGrid, rGrid, NHGrid, uarr, muarr,
                   Heat, Cool, N_nH, N_T, N_r, N_NH, gamma, kB, mH);

}




