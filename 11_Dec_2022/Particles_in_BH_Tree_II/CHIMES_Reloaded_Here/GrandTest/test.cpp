#include <iostream>
#include <fstream>

using namespace std;


int main()
{
  std::ifstream file("HCoolChimes.bin", std::ios::binary);
    
  // Reading integers
  int N_T, N_nH, N_r, N_NH, N_t;
  file.read(reinterpret_cast<char*>(&N_T), sizeof(N_T));
  file.read(reinterpret_cast<char*>(&N_nH), sizeof(N_nH));
  file.read(reinterpret_cast<char*>(&N_r), sizeof(N_r));
  file.read(reinterpret_cast<char*>(&N_NH), sizeof(N_NH));
  file.read(reinterpret_cast<char*>(&N_t), sizeof(N_t));

  // Reading floats
  float nHLowBound, nHUpBound, rLowBound, rUpBound, NHLowBound, NHUpBound, timeLowBound, timeUpBound;
  file.read(reinterpret_cast<char*>(&nHLowBound), sizeof(nHLowBound));
  file.read(reinterpret_cast<char*>(&nHUpBound), sizeof(nHUpBound));
  file.read(reinterpret_cast<char*>(&rLowBound), sizeof(rLowBound));
  file.read(reinterpret_cast<char*>(&rUpBound), sizeof(rUpBound));
  file.read(reinterpret_cast<char*>(&NHLowBound), sizeof(NHLowBound));
  file.read(reinterpret_cast<char*>(&NHUpBound), sizeof(NHUpBound));
  file.read(reinterpret_cast<char*>(&timeLowBound), sizeof(timeLowBound));
  file.read(reinterpret_cast<char*>(&timeUpBound), sizeof(timeUpBound));

  // Reading arrays
  float* uarr = new float[N_T * N_nH * N_r * N_NH * N_t];
  file.read(reinterpret_cast<char*>(uarr), (N_T * N_nH * N_r * N_NH * N_t) * sizeof(float));

  float* nHGrid = new float[N_nH];
  file.read(reinterpret_cast<char*>(nHGrid), N_nH * sizeof(float));

  float* rGrid = new float[N_r];
  file.read(reinterpret_cast<char*>(rGrid), N_r * sizeof(float));

  float* NHGrid = new float[N_NH];
  file.read(reinterpret_cast<char*>(NHGrid), N_NH * sizeof(float));

  file.close();

  
  for (int i = 0; i < 100; i++)
  {
    cout << uarr[i] << endl;
  }




}


