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
  int *dev_Typ;
  float *dev_x, *dev_y, *dev_z, *dev_eps;
  float *dev_accx, *dev_accy, *dev_accz, *dev_mass;

  for (int i = 0; i < devCount; i++)
  {
    //cudaSetDevice(i);

    // Allocate device memory for full arrays
    cudaMalloc(&dev_Typ, N * sizeof(int));
    cudaMalloc(&dev_x, N * sizeof(float));
    cudaMalloc(&dev_y, N * sizeof(float));
    cudaMalloc(&dev_z, N * sizeof(float));
    cudaMalloc(&dev_eps, N * sizeof(float));
    cudaMalloc(&dev_accx, N * sizeof(float));
    cudaMalloc(&dev_accy, N * sizeof(float));
    cudaMalloc(&dev_accz, N * sizeof(float));
    cudaMalloc(&dev_mass, N * sizeof(float));

    // Copy full arrays to each device
    cudaMemcpy(dev_Typ, Typ, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_z, z, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_eps, eps, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_mass, mass, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_accx, accx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_accy, accy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_accz, accz, N * sizeof(float), cudaMemcpyHostToDevice);
  }

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  //------------------------------------------------------------------
  int beg = 0;
  int end = N - 1;


  int iterations = 10;

  for (int iter = 0; iter < iterations; iter++)
  {

    auto start_loop = std::chrono::high_resolution_clock::now();

      auto start_kernel = std::chrono::high_resolution_clock::now();

      // Run the kernel only on a portion of the arrays
      acc_g_mgpu<<<gridSize, blockSize>>>(beg, end, dev_Typ, dev_x, dev_y, dev_z,
                                          dev_eps, dev_accx, dev_accy, dev_accz, dev_mass,
                                          G, N);

      auto end_kernel = std::chrono::high_resolution_clock::now();
      auto elapsed_kernel = std::chrono::duration_cast<std::chrono::nanoseconds>(end_kernel - start_kernel);
      cout << "Kernel time = " << elapsed_kernel.count() * 1e-9 << endl;

      cudaDeviceSynchronize();

    // Save to a file
    // Copy data from GPU to host
    cudaMemcpy(accx, dev_accx, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(accy, dev_accy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(accz, dev_accz, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Save the arrays to binary format
    std::string filename = "outSingle.bin";
    saveArraysToBinary(filename, accx, accy, accz, N);

    auto end_loop = std::chrono::high_resolution_clock::now();
    auto elapsed_loop = std::chrono::duration_cast<std::chrono::nanoseconds>(end_loop - start_loop);
    cout << "Elapsed time = " << elapsed_loop.count() * 1e-9 << endl;

  } // ===> The end of iteration!

  // Free device memory
    cudaFree(dev_Typ);
    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_z);
    cudaFree(dev_eps);
    cudaFree(dev_accx);
    cudaFree(dev_accy);
    cudaFree(dev_accz);
    cudaFree(dev_mass);

  // Free host memory
  free(Typ);
  free(x);
  free(y);
  free(z);
  free(eps);
  free(mass);

} // The end of main().
