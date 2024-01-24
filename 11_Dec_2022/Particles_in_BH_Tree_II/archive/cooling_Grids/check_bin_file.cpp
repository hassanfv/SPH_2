#include <iostream>
#include <fstream>
#include <cstring>

using namespace std;

int main() {
  std::ifstream file("coolHeatGridDec2023.bin", std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Unable to open file" << std::endl;
    return 1;
  }

  // Read dimensions
  int N_kpc, N_nH, N_Z, N_T, N_M;
  file.read(reinterpret_cast<char*>(&N_kpc), sizeof(N_kpc));
  file.read(reinterpret_cast<char*>(&N_T), sizeof(N_T));
  file.read(reinterpret_cast<char*>(&N_nH), sizeof(N_nH));
  file.read(reinterpret_cast<char*>(&N_Z), sizeof(N_Z));
  file.read(reinterpret_cast<char*>(&N_M), sizeof(N_M));

  // Allocate memory for 1D arrays
  float* kpcsF = new float[N_kpc];
  float* densities = new float[N_nH];
  float* metallicities = new float[N_Z];
  float* temperatures = new float[N_T];
  float* uEvolutionX = new float[N_kpc * N_T * N_nH * N_Z];
  float* muArrX = new float[N_kpc * N_T * N_nH * N_Z];
  float* metalzX = new float[N_kpc * N_T * N_nH * N_Z * N_M];
  float* uArrX = new float[N_kpc * N_T * N_nH * N_Z];

  // Read data into arrays
  file.read(reinterpret_cast<char*>(kpcsF), N_kpc * sizeof(float));
  file.read(reinterpret_cast<char*>(densities), N_nH * sizeof(float));
  file.read(reinterpret_cast<char*>(metallicities), N_Z * sizeof(float));
  file.read(reinterpret_cast<char*>(temperatures), N_T * sizeof(float));
  
  file.read(reinterpret_cast<char*>(uEvolutionX), N_kpc * N_T * N_nH * N_Z * sizeof(float));
  file.read(reinterpret_cast<char*>(muArrX), N_kpc * N_T * N_nH * N_Z * sizeof(float));
  file.read(reinterpret_cast<char*>(metalzX), N_kpc * N_T * N_nH * N_Z * N_M * sizeof(float));
  file.read(reinterpret_cast<char*>(uArrX), N_kpc * N_T * N_nH * N_Z * sizeof(float));

  int i = 2; // kpc
  int j = 5; // T
  int k = 6; // nH
  int t = 1; // Z
  
  int index = i * (N_T * N_nH * N_Z) + j * (N_nH * N_Z) + k * (N_Z) + t;
  
  cout << uEvolutionX[index] << endl;
  cout << uArrX[index] << endl;

  // Clean up
  delete[] kpcsF;
  delete[] densities;
  delete[] metallicities;
  delete[] temperatures;
  delete[] uEvolutionX;
  delete[] muArrX;
  delete[] metalzX;

  file.close();
  return 0;
}

