//%%writefile test.cu
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <tuple>
#include "multi_libs.h"

using namespace std;

// Function to save results
void saveArraysToBinary(const std::string &filename, float *accx, float *accy, float *accz, size_t N)
{
  std::ofstream out(filename, std::ios::binary);
  out.write(reinterpret_cast<const char *>(accx), N * sizeof(float));
  out.write(reinterpret_cast<const char *>(accy), N * sizeof(float));
  out.write(reinterpret_cast<const char *>(accz), N * sizeof(float));
  out.close();
}

int main()
{

  // Reading the params.txt file
  std::string filename;
  int N;
  float G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma;

  readParams(filename, N, G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma);

  std::cout << "filename: " << filename << "\n";
  std::cout << "N: " << N << "\n";
  std::cout << "G: " << G << "\n";
  std::cout << "L_AGN_code_unit: " << L_AGN_code_unit << "\n";
  std::cout << "M_dot_in_code_unit: " << M_dot_in << "\n";
  std::cout << "vin_in_code_unit: " << v_in << "\n";
  std::cout << "u_for_10K_Temp: " << u_for_10K_Temp << "\n";
  std::cout << "m_sph_high_res: " << m_sph_high_res << "\n";
  std::cout << "sigma: " << sigma << "\n";

  // Reading the IC file
  auto data = readVectorsFromFile(N, filename);

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

  // Allocate and initialize host memory for full arrays
  int *Typ = new int[N];
  float *x = new float[N];
  float *y = new float[N];
  float *z = new float[N];
  float *eps = new float[N];
  float *mass = new float[N];

  float *accx = new float[N];
  float *accy = new float[N];
  float *accz = new float[N];

  for (int i = 0; i < N; i++)
  {
    Typ[i] = Typvec[i];
    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];
    eps[i] = epsvec[i];
    mass[i] = massvec[i];
    accx[i] = 0.0f;
    accy[i] = 0.0f;
    accz[i] = 0.0f;
  }

  int devCount;
  cudaGetDeviceCount(&devCount);

  // Pointers for device memory
  int *dev_Typ[devCount];
  float *dev_x[devCount], *dev_y[devCount], *dev_z[devCount], *dev_eps[devCount];
  float *dev_accx[devCount], *dev_accy[devCount], *dev_accz[devCount], *dev_mass[devCount];

  for (int i = 0; i < devCount; i++)
  {
    cudaSetDevice(i);

    // Allocate device memory for full arrays
    cudaMalloc(&dev_Typ[i], N * sizeof(int));
    cudaMalloc(&dev_x[i], N * sizeof(float));
    cudaMalloc(&dev_y[i], N * sizeof(float));
    cudaMalloc(&dev_z[i], N * sizeof(float));
    cudaMalloc(&dev_eps[i], N * sizeof(float));
    cudaMalloc(&dev_accx[i], N * sizeof(float));
    cudaMalloc(&dev_accy[i], N * sizeof(float));
    cudaMalloc(&dev_accz[i], N * sizeof(float));
    cudaMalloc(&dev_mass[i], N * sizeof(float));

    // Copy full arrays to each device
    cudaMemcpy(dev_Typ[i], Typ, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x[i], x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y[i], y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z[i], z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eps[i], eps, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mass[i], mass, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_accx[i], accx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_accy[i], accy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_accz[i], accz, N * sizeof(float), cudaMemcpyHostToDevice);
  }

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  //------------------------------------------------------------------
  // Determining the beg and end index of the array for each GPU.

  int nGPUs = 4; // devCount;

  cout << "nGPUs = " << nGPUs << endl;

  const int N_per_GPU = N / nGPUs;
  const int remainder = N % nGPUs;

  int *beg, *end, *MLen; // MLen means Memory length which is end - beg!

  beg = new int[nGPUs];
  end = new int[nGPUs];
  MLen = new int[nGPUs];

  for (int rank = 0; rank < nGPUs; rank++)
  {
    if (rank < remainder)
    {
      beg[rank] = rank * (N_per_GPU + 1);
      end[rank] = beg[rank] + N_per_GPU + 1;
      MLen[rank] = end[rank] - beg[rank];
    }
    else
    {
      beg[rank] = rank * N_per_GPU + remainder;
      end[rank] = beg[rank] + N_per_GPU;
      MLen[rank] = end[rank] - beg[rank];
    }
  }
  //------------------------

  int iterations = 10;

  for (int iter = 0; iter < iterations; iter++)
  {

    //================ acc_g ====================
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      acc_g_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], dev_Typ[i], dev_x[i], dev_y[i], dev_z[i],
                                          dev_eps[i], dev_accx[i], dev_accy[i], dev_accz[i], dev_mass[i],
                                          G, N);
    }

    // Synchronize
    for (int i = 0; i < nGPUs; i++)
    {
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    // Now dev_accx[i], dev_accy[i], dev_accz[i] for each GPU, i, contain the results of their own portion of the data.

    // Copy the results to main GPU.
    int NN = MLen[0];
    for (int i = 1; i < nGPUs; i++)
    {

      cudaMemcpyPeer(dev_accx[0] + NN, 0, dev_accx[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(dev_accy[0] + NN, 0, dev_accy[i] + NN, i, MLen[i] * sizeof(float));
      cudaMemcpyPeer(dev_accz[0] + NN, 0, dev_accz[i] + NN, i, MLen[i] * sizeof(float));

      NN = NN + MLen[i];
    }

    // The main GPU now has the full results in dev_accx[0], dev_accy[0], dev_accz[0]

    // Copy the full results back to each device for next iteration (i.e. time-step).
    for (int i = 1; i < devCount; i++) // Note that GPU 0 already hass the data!
    {
      cudaMemcpyPeer(dev_accx[i], i, dev_accx[0], 0, N * sizeof(float));
      cudaMemcpyPeer(dev_accy[i], i, dev_accy[0], 0, N * sizeof(float));
      cudaMemcpyPeer(dev_accz[i], i, dev_accz[0], 0, N * sizeof(float));
    }

  } // ===> The end of iteration!

} // The end of main().
