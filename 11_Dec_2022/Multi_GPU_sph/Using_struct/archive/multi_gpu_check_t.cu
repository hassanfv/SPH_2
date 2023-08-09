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
#include "multi_libs.h"

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
  int *Typ = new int[N];

  float *x = new float[N];
  float *y = new float[N];
  float *z = new float[N];

  float *vx = new float[N];
  float *vy = new float[N];
  float *vz = new float[N];

  float *h = new float[N];
  float *eps = new float[N];
  float *mass = new float[N];
  float *u = new float[N];

  float *accx = new float[N];
  float *accy = new float[N];
  float *accz = new float[N];

  float *accx_tot = new float[N];
  float *accy_tot = new float[N];
  float *accz_tot = new float[N];

  float *abs_acc_g = new float[N];
  float *abs_acc_tot = new float[N];
  float *v_sig = new float[N];

  float *rho = new float[N];
  float *P = new float[N];
  float *csnd = new float[N];

  float *divV = new float[N];
  float *curlV = new float[N];

  float *accx_sph = new float[N];
  float *accy_sph = new float[N];
  float *accz_sph = new float[N];

  float *dh_dt = new float[N];

  float *dudt = new float[N];
  float *utprevious = new float[N];

  float *dt_particles = new float[N];

  float *Nngb_previous = new float[N];

  for (int i = 0; i < N; i++)
  {

    if (i == 200000)
      cout << "check" << endl;

    Typ[i] = Typvec[i];

    x[i] = xvec[i];
    y[i] = yvec[i];
    z[i] = zvec[i];

    vx[i] = vxvec[i];
    vy[i] = vyvec[i];
    vz[i] = vzvec[i];

    h[i] = hvec[i];
    eps[i] = epsvec[i];
    mass[i] = massvec[i];
    u[i] = uvec[i];

    accx[i] = 0.0f;
    accy[i] = 0.0f;
    accz[i] = 0.0f;

    accx_tot[i] = 0.0f;
    accy_tot[i] = 0.0f;
    accz_tot[i] = 0.0f;

    abs_acc_g[i] = 0.0f;
    abs_acc_tot[i] = 0.0f;
    v_sig[i] = 0.0f;

    rho[i] = 0.0f;  // place holder.
    P[i] = 0.0f;    // placeholder.
    csnd[i] = 0.0f; // placeholder.

    divV[i] = 0.0f;  // placeholder.
    curlV[i] = 0.0f; // placeholder.

    if (Typ[i] == 0)
    {
      Nngb_previous[i] = Nngb_f;
    }
    else
    {
      Nngb_previous[i] = 0.0f;
    }
  }

  int nGPUs;
  cudaGetDeviceCount(&nGPUs);

  cout << "nGPUs = " << nGPUs << endl;

  // Pointers for device memory
  int *d_Typ[nGPUs];
  float *d_x[nGPUs], *d_y[nGPUs], *d_z[nGPUs], *d_eps[nGPUs];
  float *d_accx[nGPUs], *d_accy[nGPUs], *d_accz[nGPUs], *d_mass[nGPUs];

  float *d_vx[nGPUs], *d_vy[nGPUs], *d_vz[nGPUs];
  float *d_h[nGPUs], *d_rho[nGPUs];

  float *d_P[nGPUs], *d_csnd[nGPUs], *d_divV[nGPUs], *d_curlV[nGPUs];
  float *d_accx_sph[nGPUs], *d_accy_sph[nGPUs], *d_accz_sph[nGPUs];
  float *d_accx_tot[nGPUs], *d_accy_tot[nGPUs], *d_accz_tot[nGPUs];
  float *d_abs_acc_g[nGPUs], *d_abs_acc_tot[nGPUs];
  float *d_v_sig[nGPUs], *d_dh_dt[nGPUs], *d_u[nGPUs], *d_dudt[nGPUs];
  float *d_utprevious[nGPUs];
  float *d_Nngb_previous[nGPUs]; // Note that both are floats and not int! check smoothing func. to see why!
  float *d_dt_particles[nGPUs];

  for (int i = 0; i < nGPUs; i++)
  {

    cout << "i = " << i << endl;

    cudaSetDevice(i);

    // Allocate device memory for full arrays
    cudaMalloc(&d_Typ[i], N * sizeof(int));

    cudaMalloc(&d_x[i], N * sizeof(float));
    cudaMalloc(&d_y[i], N * sizeof(float));
    cudaMalloc(&d_z[i], N * sizeof(float));

    cudaMalloc(&d_vx[i], N * sizeof(float));
    cudaMalloc(&d_vy[i], N * sizeof(float));
    cudaMalloc(&d_vz[i], N * sizeof(float));

    cudaMalloc(&d_accx[i], N * sizeof(float));
    cudaMalloc(&d_accy[i], N * sizeof(float));
    cudaMalloc(&d_accz[i], N * sizeof(float));

    cudaMalloc(&d_eps[i], N * sizeof(float));
    cudaMalloc(&d_mass[i], N * sizeof(float));

    cudaMalloc(&d_h[i], N * sizeof(float));
    cudaMalloc(&d_rho[i], N * sizeof(float));
    cudaMalloc(&d_P[i], N * sizeof(float));
    cudaMalloc(&d_csnd[i], N * sizeof(float));

    cudaMalloc(&d_divV[i], N * sizeof(float));
    cudaMalloc(&d_curlV[i], N * sizeof(float));

    cudaMalloc(&d_accx_sph[i], N * sizeof(float));
    cudaMalloc(&d_accy_sph[i], N * sizeof(float));
    cudaMalloc(&d_accz_sph[i], N * sizeof(float));

    cudaMalloc(&d_accx_tot[i], N * sizeof(float));
    cudaMalloc(&d_accy_tot[i], N * sizeof(float));
    cudaMalloc(&d_accz_tot[i], N * sizeof(float));

    cudaMalloc(&d_abs_acc_g[i], N * sizeof(float));
    cudaMalloc(&d_abs_acc_tot[i], N * sizeof(float));
    cudaMalloc(&d_v_sig[i], N * sizeof(float));
    cudaMalloc(&d_dh_dt[i], N * sizeof(float));
    cudaMalloc(&d_dt_particles[i], N * sizeof(float));

    cudaMalloc(&d_u[i], N * sizeof(float));
    cudaMalloc(&d_dudt[i], N * sizeof(float));
    cudaMalloc(&d_utprevious[i], N * sizeof(float));

    cudaMalloc(&d_Nngb_previous[i], N * sizeof(float));

    // Copy full arrays to each device
    cudaMemcpy(d_Typ[i], Typ, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x[i], x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y[i], y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z[i], z, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_vx[i], vx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy[i], vy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz[i], vz, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_accx[i], accx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accy[i], accy, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accz[i], accz, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_eps[i], eps, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass[i], mass, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_h[i], h, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_rho[i], rho, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_P[i], P, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csnd[i], csnd, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_divV[i], divV, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_curlV[i], curlV, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_accx_sph[i], accx_sph, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accy_sph[i], accy_sph, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accz_sph[i], accz_sph, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_accx_tot[i], accx_tot, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accy_tot[i], accy_tot, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accz_tot[i], accz_tot, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_abs_acc_g[i], abs_acc_g, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_abs_acc_tot[i], abs_acc_tot, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_sig[i], v_sig, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dh_dt[i], dh_dt, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dt_particles[i], dt_particles, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_u[i], u, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dudt[i], dudt, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_utprevious[i], utprevious, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Nngb_previous[i], Nngb_previous, N * sizeof(float), cudaMemcpyHostToDevice);
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

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!! START of smoothing_h !!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);

    cout << "AA" << endl;

    // Run the kernel only on a portion of the arrays
    smoothing_h_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], d_Typ[i], d_x[i], d_y[i], d_z[i], d_h[i],
                                              Ndown, Nup, coeff, Nngb_f, d_Nngb_previous[i], d_divV[i], dt, N);
  }

  // Synchronize to make sure computation is done before proceeding
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  // Now d_h[i] for each GPU, i, contains the results for its own portion portion of the data!

  // Copy results to main GPU.
  int NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {

    cudaMemcpyPeer(d_h[0] + NN, 0, d_h[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // The main GPU now has the full results in d_accx[0], d_accy[0], d_accz[0]

  // Copy results back to each device for next iteration
  for (int i = 1; i < nGPUs; i++) // Note that GPU 0 already has the data!
  {
    cudaMemcpyPeer(d_h[i], i, d_h[0], 0, N * sizeof(float));
  }

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!! END OF smoothing_h !!!!!!!!!!!!!!!!!!!
  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


} // The end of main().
