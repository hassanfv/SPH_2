
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib> // This is ONLY used for the "exit(0)" function !!

using namespace std;


//===== findClosestIndex
int find_nx0(float *A, int N, float x)
{
  int nxClosest = 0; // Start with the first element as the closest
  float smallestDiff = abs(A[0] - x); // Initialize the smallest difference

  for (int i = 1; i < N; ++i)
  {
    float currentDiff = abs(A[i] - x); // Compute the current difference
    if (currentDiff < smallestDiff) { // If the current difference is smaller, update the closest element
      smallestDiff = currentDiff;
      nxClosest = i;
    }
  }
  
  int nx0 = 0; // just a place holder. 
  
  if (x >= A[nxClosest])
    nx0 = nxClosest;
  else
    nx0 = nxClosest - 1;
  
  if (nx0 == N - 1) // To handle cases in which nH_p >= max(nH)
    nx0 = nxClosest - 1;
  
  if (nx0 == -1) // if nH_p is less than min(nH) then the "else" part will subtract 1 from nxClosest which is already 0, making it -1!!!!
    nx0 = 0;
  
  
  return nx0; // Return nx0. nx1 will be created in the main function!
}



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
  int nxnH0 = find_nx0(nH, N_nH, nH_p);

  cout << "nxnH0 = " << nxnH0 << endl;
  cout << "nH_p = " << nH_p << ",  nH[nxnH0] = " << nH[nxnH0] << ",  nH[nxnH1] = " << nH[nxnH0+1] << endl << endl;


  //----> rkpc <----
  int nxr0 = find_nx0(r, N_r, r_p);
  
  cout << "nxr0 = " << nxr0 << endl;
  cout << "r_p = " << r_p << ",  r[nxr0] = " << r[nxr0] << ",  r[nxr1] = " << r[nxr0+1] << endl << endl;
  

  //----> NH <----
  int nxNH0 = find_nx0(NH, N_NH, NH_p);
  
  cout << "nxNH0 = " << nxNH0 << endl;
  cout << "NH_p = " << NH_p << ",  NH[nxNH0] = " << NH[nxNH0] << ",  NH[nxNH1] = " << NH[nxNH0+1] << endl << endl;
  

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
  
  if (nxu0 == -1) // This happens if u_p is greater than max(uarr)!
    nxu0 = N_nH * N_T * N_r * N_NH - 1;
  
  cout << "nxu0 = " << nxu0 << endl;
  
  float mu_p = muarr[nxu0];
  cout << "mu_p = " << mu_p << endl;
  
  float T_p = (gamma - 1.0f) * mu_p * mH / kB * pow(10, u_p);
  T_p = log10(T_p);
  
  cout << "T_p = " << T_p << endl;
  
  //--- Now that we have T_p, we proceed to find nxT0!
  int nxT0 = find_nx0(T, N_T, T_p);
  
  cout << "nxT0 = " << nxT0 << endl;
  cout << "T_p = " << T_p << ",  T[nxT0] = " << T[nxT0] << ",  T[nxT1] = " << T[nxT0+1] << endl << endl;

  int nxT1 = nxT0 + 1;

  float Gam_minus_Lam = interpolate_4d_hypercube(nH_p, T_p, r_p, NH_p, nH, T, r, NH, Gam, Lam, nxnH0, nxT0, nxr0, nxNH0,
                                                 nxnH1, nxT1, nxr1, nxNH1, N_nH, N_T, N_r, N_NH);

  cout << "Gam - Lam = " << Gam_minus_Lam << endl << endl;

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


  //-0.1	0.3	19.8	4.6	-22.5319	-23.0429	0.5124

  float nH_p = 3.83;
  float T_p = 4.07;
  float r_p = 0.17;
  float NH_p = 21.17;
  float mu_tmp = 0.6124;
  
  float u_p = kB * pow(10, T_p) / (gamma - 1.0) / mu_tmp / mH; // log10 of it will be taken inside the "doCloudyHCooling" function!
  
  cout << "u_p = " << u_p << endl;
  
  doCloudyHCooling(nH_p, u_p, r_p, NH_p, nHGrid, TGrid, rGrid, NHGrid, uarr, muarr,
                   Heat, Cool, N_nH, N_T, N_r, N_NH, gamma, kB, mH);

}




