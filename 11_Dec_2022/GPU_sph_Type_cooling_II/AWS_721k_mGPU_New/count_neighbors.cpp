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

std::tuple<std::vector<int>, 
           std::vector<float>, std::vector<float>, std::vector<float>, 
           std::vector<float>, std::vector<float>, std::vector<float>, 
           std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
readVectorsFromFile(const std::string &filename) 
{
    int N;
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Could not open the IC file for reading." << std::endl;
        exit(1); // or other error handling
    }

    // Read N_tot from the start of the file
    file.read(reinterpret_cast<char*>(&N), sizeof(int));

    // Create and resize vectors based on N
    std::vector<int> Typvec(N);
    std::vector<float> xvec(N);
    std::vector<float> yvec(N);
    std::vector<float> zvec(N);
    std::vector<float> vxvec(N);
    std::vector<float> vyvec(N);
    std::vector<float> vzvec(N);
    std::vector<float> uvec(N);
    std::vector<float> hvec(N);
    std::vector<float> epsvec(N);
    std::vector<float> massvec(N);

    file.read(reinterpret_cast<char*>(Typvec.data()), sizeof(int) * N);
    file.read(reinterpret_cast<char*>(xvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(yvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(zvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(vxvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(vyvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(vzvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(uvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(hvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(epsvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(massvec.data()), sizeof(float) * N);

    file.close();

    return std::make_tuple(Typvec, xvec, yvec, zvec, vxvec, vyvec, vzvec, uvec, hvec, epsvec, massvec);
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

  const int N = 721166; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  

  //*********************************************************************
  //******************** Reading the IC file ****************************
  //*********************************************************************
  auto data = readVectorsFromFile("IC_R_721k.bin");

  std::vector<int> &Typvec = std::get<0>(data);
  std::vector<float> &xvec = std::get<1>(data);
  std::vector<float> &yvec = std::get<2>(data);
  std::vector<float> &zvec = std::get<3>(data);
  std::vector<float> &vxvec = std::get<4>(data);
  std::vector<float> &vyvec = std::get<5>(data);
  std::vector<float> &vzvec = std::get<6>(data);
  std::vector<float> &uvec = std::get<7>(data);
  std::vector<float> &hvec = std::get<8>(data);
  std::vector<float> &epsvec = std::get<9>(data);
  std::vector<float> &massvec = std::get<10>(data);

  // declaring the arrays.
  int *Typ;
  float *x, *y, *z;
  float *Nngb_previous, *divV, *h; // Note that both are floats and not int! check smoothing func. to see why!

  Typ = new int[N];

  x = new float[N];
  y = new float[N];
  z = new float[N];

  h = new float[N];
  

  for (int i = 0; i < N; i++)
  {
    Typ[i] = Typvec[i];

    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];

    h[i] = hvec[i];
  }

  int i = 222000;

  float dist = (2.0f * h[i]);
  
  int N_ngb = countNearby(x, y, z, Typ, i, N, dist);

  cout << "with h[" << i << "] = " << h[i] << " we find " << N_ngb << " neighboring particles !" << endl;

}






