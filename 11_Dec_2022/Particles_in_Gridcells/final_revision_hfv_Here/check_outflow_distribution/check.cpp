#include <iostream>
#include <cmath>
#include <random>
#include <fstream>


using namespace std;


//==============================================================
//====================== hash function =========================
//==============================================================
unsigned long long my_hash(unsigned long long x, unsigned long long y)
{
    return x * 6364136223846793005ULL + y;
}



int main()
{


  int jj = -1;
  
  int N = 50000;
  
  
  float *x = new float[N];
  float *y = new float[N];
  float *z = new float[N];
  
  float *vx = new float[N];
  float *vy = new float[N];
  float *vz = new float[N];
  
  
  float v_in = 300000.0; // km/s
  
  int seed = 0;
  

  for (int j = 0; j < N/2; j++)
  {
  
      seed = 10;
  
      // Initialize random number generator
      std::mt19937_64 rng(my_hash(seed, j));
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);

      const float one_pc_in_code_unit = 1.0f;  // Replace with your value

      float rt = dist(rng) * one_pc_in_code_unit;
      float theta = dist(rng) * 2.0f * M_PI;
      //float phi = dist(rng) * M_PI;; // This was non symmetric!!
      float phi = acos(2.0f * dist(rng) - 1.0f);

      float xt = rt * sin(phi) * cos(theta);
      float yt = rt * sin(phi) * sin(theta);
      float zt = rt * cos(phi);

      float rr = sqrt(xt * xt + yt * yt + zt * zt);

      float vxt = xt / rr * v_in;
      float vyt = yt / rr * v_in;
      float vzt = zt / rr * v_in;

      //---- Injecting the first particle of the pair ----
      x[jj + 1] = xt;
      y[jj + 1] = yt;
      z[jj + 1] = zt;

      vx[jj + 1] = vxt;
      vy[jj + 1] = vyt;
      vz[jj + 1] = vzt;

      //---- Injecting the second particle of the pair ----
      x[jj + 2] = -xt;
      y[jj + 2] = -yt;
      z[jj + 2] = -zt;

      vx[jj + 2] = -vxt;
      vy[jj + 2] = -vyt;
      vz[jj + 2] = -vzt;

      jj += 2;     
  }

  // Save data to a binary file
  std::ofstream outFile("data.bin", std::ios::binary);

  // Write N
  outFile.write(reinterpret_cast<const char*>(&N), sizeof(N));

  // Write arrays
  outFile.write(reinterpret_cast<const char*>(x), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(y), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(z), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(vx), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(vy), N * sizeof(float));
  outFile.write(reinterpret_cast<const char*>(vz), N * sizeof(float));

  outFile.close();

  // Don't forget to release the memory using delete[].
  delete[] x;
  delete[] y;
  delete[] z;
  delete[] vx;
  delete[] vy;
  delete[] vz;

}






