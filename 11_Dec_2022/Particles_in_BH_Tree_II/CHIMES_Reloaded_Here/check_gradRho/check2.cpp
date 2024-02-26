
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <tuple>
#include <cstdlib> // This is ONLY used for the "exit(0)" function !!

using namespace std;


// Function to read binary file
void readBinaryFile(const std::string& filename, int& N, int& N_ionFrac, int*& Typ, float*& x, float*& y, float*& z, float*& vx, float*& vy, float*& vz, float*& rho, float*& h, float*& u, float*& mass, float*& ionFrac) {
  // Open file
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
      std::cerr << "Failed to open file" << std::endl;
      return;
  }

  // Read N and N_ionFrac
  file.read(reinterpret_cast<char*>(&N), sizeof(N));
  file.read(reinterpret_cast<char*>(&N_ionFrac), sizeof(N_ionFrac));

  // Allocate memory for arrays
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

  // Read arrays
  file.read(reinterpret_cast<char*>(Typ), N * sizeof(int));
  file.read(reinterpret_cast<char*>(x), N * sizeof(float));
  file.read(reinterpret_cast<char*>(y), N * sizeof(float));
  file.read(reinterpret_cast<char*>(z), N * sizeof(float));
  file.read(reinterpret_cast<char*>(vx), N * sizeof(float));
  file.read(reinterpret_cast<char*>(vy), N * sizeof(float));
  file.read(reinterpret_cast<char*>(vz), N * sizeof(float));
  file.read(reinterpret_cast<char*>(rho), N * sizeof(float));
  file.read(reinterpret_cast<char*>(h), N * sizeof(float));
  file.read(reinterpret_cast<char*>(u), N * sizeof(float));
  file.read(reinterpret_cast<char*>(mass), N * sizeof(float));
  file.read(reinterpret_cast<char*>(ionFrac), N_ionFrac * sizeof(float));

  // Close file
  file.close();
}

//=========================================================
//================== grad_rho_norm_ngb ====================
//=========================================================
float grad_rho_norm_ngb(int i, int *Typ, float *x, float *y, float *z, float *rho, float *mass,
                                  float *h, int N)
{

  float dx, dy, dz, rr, hij, q, hij5, sig;
  float nW = 0.0f;
  float gWx = 0.0f;
  float gWy = 0.0f;
  float gWz = 0.0f;
  
  float gradrhox = 0.0f;
  float gradrhoy = 0.0f;
  float gradrhoz = 0.0f;

  for (int j = 0; j < N; j++)
  {
    dx = x[j] - x[i];
    dy = y[j] - y[i];
    dz = z[j] - z[i];

    rr = sqrt(dx * dx + dy * dy + dz * dz);
    hij = 0.5f * (h[i] + h[j]);
    q = rr / hij;

    if (q <= 2.0f)
    {

      nW = 0.0f;
      gWx = 0.0f;
      gWy = 0.0f;
      gWz = 0.0f;

      sig = 1.0f / M_PI;
      hij5 = hij * hij * hij * hij * hij;

      if (q <= 1.0f)
      {
        nW = sig / hij5 * (-3.0f + (9.0f / 4.0f) * q);
        gWx = nW * dx;
        gWy = nW * dy;
        gWz = nW * dz;
      }

      if ((q > 1.0f) && (q <= 2.0f))
      {
        nW = -3.0f * sig / (4.0f * hij5) * (2.0f - q) * (2.0f - q) / (q + 1e-10);
        gWx = nW * dx;
        gWy = nW * dy;
        gWz = nW * dz;
      }

      gradrhox += mass[j] * gWx;
      gradrhoy += mass[j] * gWy;
      gradrhoz += mass[j] * gWz;
    }
  }
  float gradRhoNorm = sqrt(gradrhox * gradrhox + gradrhoy * gradrhoy + gradrhoz * gradrhoz);
  
  return gradRhoNorm;
}





int main() 
{

  int N, N_ionFrac;
  int* Typ = nullptr;
  float* x = nullptr, *y = nullptr, *z = nullptr, *vx = nullptr, *vy = nullptr, *vz = nullptr, *rho = nullptr, *h = nullptr;
  float *u = nullptr, *mass = nullptr, *ionFrac = nullptr;

  std::string filename = "G-0.089583.bin";
  readBinaryFile(filename, N, N_ionFrac, Typ, x, y, z, vx, vy, vz, rho, h, u, mass, ionFrac);


  float unitLength_in_cm = 3.086e+21;
  float unit_rho = 1.62276e-23;
  float mH = 1.673534e-24;
  float XH = 0.7;
  
  //int i = 447896;//2824624; //1598369;
  
  for (int i = 0; i < N; i += 2)
  { 
    //cout << "rho[i] = " << rho[i] << endl;
    float gradRho = grad_rho_norm_ngb(i, Typ, x, y, z, rho, mass, h, N);
    
    float rho_gradRho = rho[i] / gradRho * unitLength_in_cm;
    
    //cout << "rho/gradRho = " << rho_gradRho << " cm" << endl;
    
    float nH_cgs = rho[i] * unit_rho * XH / mH;
    
    //cout << "nH_cgs = " << nH_cgs << endl;
    
    float NH_Shield = log10(rho_gradRho * nH_cgs);
    
    //cout << "NH_shield = " << NH_Shield << endl;
    //cout << "====================================" << endl << endl;
    
    if (!(i % 2000))
      cout << "We are at i = " << i << endl << endl;
    
    if (NH_Shield < 19.0)
    {
      cout << "NH_shield = " << NH_Shield << endl;
      cout << "i = " << i << endl;
      cout << endl << endl;
    }
  }
}






