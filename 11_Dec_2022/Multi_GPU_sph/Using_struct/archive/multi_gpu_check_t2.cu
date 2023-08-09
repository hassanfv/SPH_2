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

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include "multi_libs_struct.h"

using namespace std;

// Function to save results
void saveArraysToBinary(const std::string &filename, float *h, size_t N)
{
  std::ofstream out(filename, std::ios::binary);
  out.write(reinterpret_cast<const char *>(h), N * sizeof(float));
  out.close();
}

int main()
{

  float dt = 1e-7; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is only the first time step !!

  const int Nngb_f = 64.0f; // used in smoothing func.
  const int Nngb = 64;
  const int Ndown = Nngb - 5;
  const int Nup = Nngb + 5;
  const float coeff = 0.001f; // used for smoothing length.
  float gammah = 5.0f / 3.0f;
  const float C_CFL = 0.25;

  float h_min, h_max, h_mean;

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

  Particle *p = new Particle[N];

  for (int i = 0; i < N; i++)
  {

    if (i == 200000)
      cout << "check" << endl;

    p[i].Type = Typvec[i];

    p[i].x = xvec[i];
    p[i].y = yvec[i];
    p[i].z = zvec[i];

    p[i].h = hvec[i];

    p[i].divV = 0.0f;

    if (p[i].Type == 0)
      p[i].Nngb_previous = Nngb_f;
    else
      p[i].Nngb_previous = 0.0f;
  }

  // int nGPUs;
  // cudaGetDeviceCount(&nGPUs);

  int nGPUs = 4;

  cout << "nGPUs = " << nGPUs << endl;

  // Pointers for device memory
  Particle *d_p[nGPUs];

  for (int i = 0; i < nGPUs; i++)
  {

    cout << "i = " << i << endl;

    cudaSetDevice(i);

    // Allocate memory for N particles on device i
    cudaMalloc(&d_p[i], N * sizeof(Particle));

    // Copy full arrays to each device
    cudaMemcpy(d_p[i], p, N * sizeof(Particle), cudaMemcpyHostToDevice);
  }

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  const float visc_alpha = 1.0f;

  float t = 0.0f;

  float tEnd = 5.0f;
  float Nt = ceil(tEnd / dt) + 1;

  //------------------------------------------------------------------
  // Determining the beg and end index of the array for each GPU.

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

  auto T_NG = std::chrono::high_resolution_clock::now();

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!! START of smoothing_h !!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    cout << "AA" << endl;

    // Run the kernel only on a portion of the arrays
    smoothing_h_mgpu_struct<<<gridSize, blockSize>>>(beg[i], end[i], d_p[i], Ndown, Nup, coeff, Nngb_f, dt, N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Copy results to main GPU.
  int NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_p[0] + NN, 0, d_p[i] + NN, i, MLen[i] * sizeof(Particle));

    NN = NN + MLen[i];
  }

  // The main GPU now has the full results in d_accx[0], d_accy[0], d_accz[0]

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already has the data!
  {

    cudaMemcpyPeer(d_p[i], i, d_p[0], 0, N * sizeof(Particle));
  }

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF smoothing_h !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  auto end_NG = std::chrono::high_resolution_clock::now();
  auto elapsed_NG = std::chrono::duration_cast<std::chrono::nanoseconds>(end_NG - T_NG);
  cout << "T_NG = " << elapsed_NG.count() * 1e-9 << endl;

} // The end of main().
