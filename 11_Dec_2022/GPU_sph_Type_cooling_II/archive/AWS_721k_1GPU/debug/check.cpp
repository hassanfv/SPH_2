#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <tuple>
//#include "hfvCLibs_v4.h"
#include <cstdlib> // This is ONLY used for the "exit(0)" function !!


using namespace std;


//***************************************
//********** Reading IC file ************
//***************************************

std::tuple<std::vector<int>, std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>, std::vector<float>>
readVectorsFromFile(int N, const std::string &filename)
{
  std::vector<int> Typvec(N);
  std::vector<float> xvec(N);
  std::vector<float> yvec(N);
  std::vector<float> zvec(N);
  std::vector<float> vxvec(N);
  std::vector<float> vyvec(N);
  std::vector<float> vzvec(N);
  std::vector<float> massvec(N);
  std::vector<float> hvec(N);
  std::vector<float> epsvec(N);
  std::vector<float> uvec(N);

  // Check if the binary file exists
  std::ifstream file(filename, std::ios::binary);
  if (!file)
  {
    std::cout << "Could not open the IC file." << std::endl;
  }
  else
  {
    // Close and reopen the file
    file.close();
    file.open(filename, std::ios::binary);

    if (file)
    {
      // Read the first array
      file.read(reinterpret_cast<char *>(Typvec.data()), sizeof(int) * Typvec.size());

      file.read(reinterpret_cast<char *>(xvec.data()), sizeof(float) * xvec.size());
      file.read(reinterpret_cast<char *>(yvec.data()), sizeof(float) * yvec.size());
      file.read(reinterpret_cast<char *>(zvec.data()), sizeof(float) * zvec.size());

      file.read(reinterpret_cast<char *>(vxvec.data()), sizeof(float) * vxvec.size());
      file.read(reinterpret_cast<char *>(vyvec.data()), sizeof(float) * vyvec.size());
      file.read(reinterpret_cast<char *>(vzvec.data()), sizeof(float) * vzvec.size());

      file.read(reinterpret_cast<char *>(massvec.data()), sizeof(float) * massvec.size());
      file.read(reinterpret_cast<char *>(hvec.data()), sizeof(float) * hvec.size());
      file.read(reinterpret_cast<char *>(epsvec.data()), sizeof(float) * epsvec.size());
      file.read(reinterpret_cast<char *>(uvec.data()), sizeof(float) * uvec.size());

      // Close the file
      file.close();
    }
    else
    {
      std::cout << "Failed to open the IC file." << std::endl;
    }
  }

  return std::make_tuple(Typvec, xvec, yvec, zvec, vxvec, vyvec, vzvec, massvec, hvec, epsvec, uvec);
}


//========================================
//========== Smoothing Length ============ Updated: 28 Jan 2023. h_new adopted from eq.31 in Gadget2 Paper
//========================================
void smoothing_h(int *Typ, float *x, float *y, float *z, float *h,
                            int N, int Ndown, int Nup, float coeff,
                            float Nngb_f, float *Nngb_previous, float *divV, float dt)
{
  
  for (int i = 0; i < N; i++)
  {
  float h_new = 2.f * (0.5f * h[i] * (1.0f + pow((Nngb_f / Nngb_previous[i]), 1.0f / 3.0f)) +
                        1.0f / 3.0f * h[i] * divV[i] * dt);
  float h_tmp = h_new;
  int N_iter = 0;
  int k = 0;
  int k_init = 0;
  
  coeff = 0.2f;
  
  int k_pre = Nup; // Just a choice so that it does not activate that if condition in the first run!
  
  int reset = 0;

  //cout << "h_new = " << h_new << endl;
  
  if (i%1000 == 0)
  {
    cout << "current i = " << i << endl;
  }

  float dx, dy, dz;
  while ((k < Ndown) || (k > Nup))
  {

    k = 0;

    for (int j = 0; j < N; j++)
    {
    
      if (Typ[j] == 0)
      {
      
        //cout << "i, k inside = " << i << ", " << k << endl;
        
        dx = x[j] - x[i];
        dy = y[j] - y[i];
        dz = z[j] - z[i];
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          k++;
        }
        
      }
      
      if (k > Nup)
      {
        break;
      }
      
    }
    
    if (reset == 0)
    {
      k_init = k;
      reset++;
    }

    //-----------
    if (((k < Ndown) && (k_pre > Nup)) || ((k > Nup) && (k_pre < Ndown)))
    {
      coeff = coeff / 2.0f;
      cout << "i, k, k_pre, k_init, coeff = " << i << ", " << k << ", " << k_pre << ", " << k_init << ", " << coeff << endl;
    }
    //-----------

    if (k < Ndown)
    {
      h_new = h_new + coeff * 2.0f * h[i];
    }

    if (k > Nup)
    {
      h_new = h_new - coeff * 2.0f * h[i];
    }

    //cout << "i, k, k_pre, coeff = " << i << ", " << k << ", " << k_pre << ", " << coeff << endl;

    k_pre = k;

    N_iter++;
    if ((N_iter > 5) || (k_init > 66) || (k_init < 61))
    {
      cout << "Max Iteration Reached for i = " << i << endl;
      cout << "initial Nngb = " << k_init << endl;
      cout << "final Nngb = " << k << endl;
      k = 0;
      for (int j = 0; j < N; j++)
      {
        if (Typ[j] == 0)
        {
          dx = x[j] - x[i];
          dy = y[j] - y[i];
          dz = z[j] - z[i];
          float rr = sqrt(dx * dx + dy * dy + dz * dz);

          if (rr <= h_new)
          {
            k++;
          }
        }
      }
      cout << "final Nngb II = " << k << endl;
      cout << "initial h = " << h[i] << endl;
      cout << "final interrupted h = " << 0.5f * h_new << endl;
      cout << "dist = " << sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]) << endl;
      cout << "pow = " << 1.0f + pow((Nngb_f / Nngb_previous[i]), 1.0f / 3.0f) << endl;
      cout << "============================" << endl << endl;
      h_new = h_tmp;
      break;
    }
  }
  
  if (i == 17435)
  {
    cout << "initial h = " << h[i] << endl;
    cout << "final h = " << 0.5 * h_new << endl;
  }
  
  //Nngb_previous[i] = k;
  h[i] = 0.5 * h_new;
  
  }
  
}


int main()
{


  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.05f; // used for smoothing length.

  const int N = 1359855; // 721663;
  

  //*********************************************************************
  //******************** Reading the IC file ****************************
  //*********************************************************************
  auto data = readVectorsFromFile(N, "IC_R_1359k.bin");

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
  int *Typ;
  float *x, *y, *z;
  float *Nngb_previous, *divV, *h; // Note that both are floats and not int! check smoothing func. to see why!

  Typ = new int[N];

  x = new float[N];
  y = new float[N];
  z = new float[N];

  h = new float[N];
  Nngb_previous = new float[N];
  divV = new float[N];

  for (int i = 0; i < N; i++)
  {
    Typ[i] = Typvec[i];

    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];

    h[i] = hvec[i];
    Nngb_previous[i] = Nngb_f;
    divV[i] = 0.0f;
  }

  float dt = 2e-7;
  
  //float h_p, Nngb_p;
  
  int i = 100;
  
  auto T_hh = std::chrono::high_resolution_clock::now();
  smoothing_h(Typ, x, y, z, h, N, Ndown, Nup, coeff, Nngb_f, Nngb_previous, divV, dt);
  auto end_hh = std::chrono::high_resolution_clock::now();
  auto elapsed_hh = std::chrono::duration_cast<std::chrono::nanoseconds>(end_hh - T_hh);
  cout << "T_h = " << elapsed_hh.count() * 1e-9 << endl;

  cout << "h = " << h[i] << endl;
  //cout << "Nngb_p = " << Nngb_p << endl;


}
