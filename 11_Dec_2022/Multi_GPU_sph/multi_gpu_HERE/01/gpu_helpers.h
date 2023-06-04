#ifndef GPU_HELPERS_H
#define GPU_HELPERS_H
#include <cuda.h>
#include "multi_libs.h"

void acc_g_on_multi_gpus(int nGPUs, int gridSize, int blockSize,
                         int *beg, int *end, int **dev_Typ, float **dev_x, float **dev_y, float **dev_z,
                         float **dev_eps, float **dev_accx, float **dev_accy, float **dev_accz, float **dev_mass,
                         const float G, const int N, int *MLen, const int devCount)
{
  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    acc_g_mgpu<<<gridSize, blockSize>>>(beg[i], end[i], dev_Typ[i], dev_x[i], dev_y[i], dev_z[i],
                                        dev_eps[i], dev_accx[i], dev_accy[i], dev_accz[i], dev_mass[i],
                                        G, N);
  }

  for (int i = 0; i < nGPUs; i++)
  {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

  int NN = MLen[0];
  for (int i = 1; i < nGPUs; i++)
  {
    cudaMemcpyPeer(dev_accx[0] + NN, 0, dev_accx[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(dev_accy[0] + NN, 0, dev_accy[i] + NN, i, MLen[i] * sizeof(float));
    cudaMemcpyPeer(dev_accz[0] + NN, 0, dev_accz[i] + NN, i, MLen[i] * sizeof(float));

    NN = NN + MLen[i];
  }

  // Copy the full results back to each device for next iteration (i.e. time-step).
  for (int i = 1; i < devCount; i++) // Note that GPU 0 already hass the data that is why i starts from 1!
  {
    cudaMemcpyPeer(dev_accx[i], i, dev_accx[0], 0, N * sizeof(float));
    cudaMemcpyPeer(dev_accy[i], i, dev_accy[0], 0, N * sizeof(float));
    cudaMemcpyPeer(dev_accz[i], i, dev_accz[0], 0, N * sizeof(float));
  }
}

#endif // GPU_HELPERS_H
