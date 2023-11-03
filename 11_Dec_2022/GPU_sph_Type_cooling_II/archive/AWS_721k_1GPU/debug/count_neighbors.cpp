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



float distance(float x1, float y1, float z1, float x2, float y2, float z2) {
    return std::sqrt(
        (x1 - x2) * (x1 - x2) +
        (y1 - y2) * (y1 - y2) +
        (z1 - z2) * (z1 - z2)
    );
}

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

  const int N = 1359855;
  

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

  for (int i = 0; i < N; i++)
  {
    Typ[i] = Typvec[i];

    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];

    h[i] = hvec[i];
  }

  int i = 242400;

  float dist = (2.0f * h[i]);
  
  int N_ngb = countNearby(x, y, z, Typ, i, N, dist);

  cout << "with h[" << i << "] = " << h[i] << " we find " << N_ngb << " neighboring particles !" << endl;

}






