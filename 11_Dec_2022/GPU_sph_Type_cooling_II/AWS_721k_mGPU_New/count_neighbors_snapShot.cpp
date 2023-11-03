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


//----- loadArraysFromBinary
bool loadArraysFromBinary(const std::string &filename, float *&x, float *&y, float *&z, float *&vx, float *&vy, float *&vz,
                          float *&rho, float *&h, float *&u, float *&mass, float *&ionFrac, int *&Typ, int &N, int &N_ionFrac)
{
    // Open the file in binary mode
    std::ifstream file(filename, std::ios::binary);

    // Check if the file was opened successfully
    if (!file)
    {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return false;
    }

    // Read N and N_ionFrac from the file
    file.read(reinterpret_cast<char *>(&N), sizeof(int));
    file.read(reinterpret_cast<char *>(&N_ionFrac), sizeof(int));

    // Allocate memory for the arrays
    Typ = new int[N];
    x = new float[N];
    y = new float[N];
    z = new float[N];
    vx = new float[N];
    vy = new float[N];
    vz = new float[N];
    rho = new float[N];
    h = new float[N];
    u = new float[N];
    mass = new float[N];
    ionFrac = new float[N_ionFrac];

    // Read the arrays from the file
    file.read(reinterpret_cast<char *>(Typ), N * sizeof(int));
    file.read(reinterpret_cast<char *>(x), N * sizeof(float));
    file.read(reinterpret_cast<char *>(y), N * sizeof(float));
    file.read(reinterpret_cast<char *>(z), N * sizeof(float));
    file.read(reinterpret_cast<char *>(vx), N * sizeof(float));
    file.read(reinterpret_cast<char *>(vy), N * sizeof(float));
    file.read(reinterpret_cast<char *>(vz), N * sizeof(float));
    file.read(reinterpret_cast<char *>(rho), N * sizeof(float));
    file.read(reinterpret_cast<char *>(h), N * sizeof(float));
    file.read(reinterpret_cast<char *>(u), N * sizeof(float));
    file.read(reinterpret_cast<char *>(mass), N * sizeof(float));
    file.read(reinterpret_cast<char *>(ionFrac), N_ionFrac * sizeof(float));

    // Close the file
    file.close();

    return true;
}



//----- distance
float distance(float x1, float y1, float z1, float x2, float y2, float z2) {
    return std::sqrt(
        (x1 - x2) * (x1 - x2) +
        (y1 - y2) * (y1 - y2) +
        (z1 - z2) * (z1 - z2)
    );
}


//----- countNearby
int countNearby(float x[], float y[], float z[], int Typ[], int index, int size, float dist) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (i == index) continue; // skip comparing with itself
        if (Typ[i] == 0 && distance(x[index], y[index], z[index], x[i], y[i], z[i]) <= dist) {
            count++;
        }
    }
    return count;
}



int main()
{  

  //*********************************************************************
  //******************** Reading the IC file ****************************
  //*********************************************************************
  std::string filename = "./Out712k_mgpu/G-0.019400.bin";

  float *x = nullptr, *y = nullptr, *z = nullptr;
  float *vx = nullptr, *vy = nullptr, *vz = nullptr;
  float *rho = nullptr, *h = nullptr, *u = nullptr, *mass = nullptr, *ionFrac = nullptr;
  int *Typ = nullptr;
  int N, N_ionFrac;
  
  if (loadArraysFromBinary(filename, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac, Typ, N, N_ionFrac))
  {
    std::cout << "Data loaded successfully from " << filename << std::endl;
  }
  else 
  {
    std::cerr << "Failed to load data from " << filename << std::endl;
    return 1;  // Exit with an error code
  }

  int i = 253000;

  float dist = (2.0f * h[i]);
  
  int N_ngb = countNearby(x, y, z, Typ, i, N, dist);

  cout << endl;
  cout << endl;
  cout << "with h[" << i << "] = " << h[i] << " we find " << N_ngb << " neighboring particles !" << endl;

}






