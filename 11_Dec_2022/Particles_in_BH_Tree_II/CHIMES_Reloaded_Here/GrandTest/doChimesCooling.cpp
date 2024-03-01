


//===== findClosestIndex
__device__ int find_nx0(float *A, int N, float x)
{
  int nxClosest = 0; // Start with the first element as the closest
  float smallestDiff = fabsf(A[0] - x); // Initialize the smallest difference using fabsf for float absolute value in CUDA

  for (int i = 1; i < N; ++i)
  {
    float currentDiff = fabsf(A[i] - x); // Compute the current difference using fabsf
    if (currentDiff < smallestDiff) { // If the current difference is smaller, update the closest element
      smallestDiff = currentDiff;
      nxClosest = i;
    }
  }

  int nx0 = 0; // just a place holder.

  if (x >= A[nxClosest])
    nx0 = nxClosest;
  else
    nx0 = nxClosest - 1;

  if (nx0 == N - 1) // To handle cases in which the index is at the last element
    nx0 = nxClosest - 1;

  if (nx0 == -1) // Handle case where computed index is below 0
    nx0 = 0;

  return nx0; // Return nx0. nx1 will be computed elsewhere!
}



//===== interpolate_4d_hypercube
__device__ float interpolate_4d_hypercube(float nH_p, float T_p, float r_p, float NH_p, float *nH, float *T, float *r, float *NH,
                                          float *Gam, float *Lam, int nxnH0, int nxT0, int nxr0, int nxNH0, int nxnH1, int nxT1, int nxr1, int nxNH1,
                                          int N_nH, int N_T, int N_r, int N_NH)
{
  float dx = (nH_p - nH[nxnH0]) / (nH[nxnH1] - nH[nxnH0]);
  float dy = (T_p - T[nxT0]) / (T[nxT1] - T[nxT0]);
  float dz = (r_p - r[nxr0]) / (r[nxr1] - r[nxr0]);
  float dw = (NH_p - NH[nxNH0]) / (NH[nxNH1] - NH[nxNH0]);

  float pG[16];

  int stride_nH = N_T * N_r * N_NH;
  int stride_T = N_r * N_NH;
  int stride_r = N_NH;

  int nxs = 0;
  for (int i = nxnH0; i <= nxnH1; i++)
  {
    for (int j = nxT0; j <= nxT1; j++)
    {
      for (int k = nxr0; k <= nxr1; k++)
      {
        for (int l = nxNH0; l<= nxNH1; l++)
        {
          pG[nxs] = Gam[i * stride_nH + j * stride_T + k * stride_r + l]; // heating!
          pL[nxs] = Lam[i * stride_nH + j * stride_T + k * stride_r + l]; // cooling!
          nxs += 1;
        }
      }
    }
  }

  //-------------> HEATING <---------------
  // Interpolate along the x-axis
  float c00 = pG[0] * (1.0f - dx) + pG[8] * dx;
  float c01 = pG[1] * (1.0f - dx) + pG[9] * dx;
  float c10 = pG[2] * (1.0f - dx) + pG[10] * dx;
  float c11 = pG[3] * (1.0f - dx) + pG[11] * dx;
  float c20 = pG[4] * (1.0f - dx) + pG[12] * dx;
  float c21 = pG[5] * (1.0f - dx) + pG[13] * dx;
  float c30 = pG[6] * (1.0f - dx) + pG[14] * dx;
  float c31 = pG[7] * (1.0f - dx) + pG[15] * dx;

  // Correct interpolation along the y-axis
  float c0 = c00 * (1.0f - dy) + c10 * dy;
  float c1 = c01 * (1.0f - dy) + c11 * dy;
  float c2 = c20 * (1.0f - dy) + c30 * dy;
  float c3 = c21 * (1.0f - dy) + c31 * dy;

  // Interpolate along the z-axis
  float c = c0 * (1.0f - dz) + c2 * dz;
  float d = c1 * (1.0f - dz) + c3 * dz;

  // Finally, interpolate along the w-axis
  float GamH = c * (1.0f - dw) + d * dw;

  //-------------> COOLING <---------------
  // Interpolate along the x-axis
  c00 = pL[0] * (1.0f - dx) + pL[8] * dx;
  c01 = pL[1] * (1.0f - dx) + pL[9] * dx;
  c10 = pL[2] * (1.0f - dx) + pL[10] * dx;
  c11 = pL[3] * (1.0f - dx) + pL[11] * dx;
  c20 = pL[4] * (1.0f - dx) + pL[12] * dx;
  c21 = pL[5] * (1.0f - dx) + pL[13] * dx;
  c30 = pL[6] * (1.0f - dx) + pL[14] * dx;
  c31 = pL[7] * (1.0f - dx) + pL[15] * dx;

  // Correct interpolation along the y-axis
  c0 = c00 * (1.0f - dy) + c10 * dy;
  c1 = c01 * (1.0f - dy) + c11 * dy;
  c2 = c20 * (1.0f - dy) + c30 * dy;
  c3 = c21 * (1.0f - dy) + c31 * dy;

  // Interpolate along the z-axis
  c = c0 * (1.0f - dz) + c2 * dz;
  d = c1 * (1.0f - dz) + c3 * dz;

  // Finally, interpolate along the w-axis
  float LamC = c * (1.0f - dw) + d * dw;

  return powf(10, GamH) - powf(10, LamC); // Use powf for floating-point exponentiation
}



//===== getGamMinusLam
__device__ float getGamMinusLam(float nH_p, float u_p, float r_p, float NH_p, float *nH, float *T, float *r, float *NH, float *uarr, float *muarr,
                                float *Gam, float *Lam, int N_nH, int N_T, int N_r, int N_NH, float gamma, float kB, float mH,
                                float nHLowBound, float nHUpBound, float TLowBound, float TUpBound, float rLowBound, float rUpBound,
                                float NHLowBound, float NHUpBound)
{
  // Adjusting input parameters to be within the bounds
  nH_p = max(nHLowBound + 0.0001f, min(nH_p, nHUpBound - 0.0001f));
  r_p = max(rLowBound + 0.0001f, min(r_p, rUpBound - 0.0001f));
  NH_p = max(NHLowBound + 0.0001f, min(NH_p, NHUpBound - 0.0001f));

  // Finding indexes for interpolation
  int nxnH0 = find_nx0(nH, N_nH, nH_p);
  int nxr0 = find_nx0(r, N_r, r_p);
  int nxNH0 = find_nx0(NH, N_NH, NH_p);

  // Convert u_p to T_p
  int S1 = N_nH * N_r * N_NH * N_t;
  int S2 = N_r * N_NH * N_t;
  int S3 = N_NH * N_t;
  int S4 = N_t;

  u_p = u_p * unit_u;

  nxmu = 1 * S1 + nxnH0 * S2 + nx_r0 * S3 + nx_NH0 * S4 + 1
  mu_p = muarr[nxmu] // We assume mu does not vary too much around these indices!!
  float T_p = (gamma - 1.0f) * mu_p * mH / kB * u_p;
  T_p = log10(T_p);

  // Ensuring T_p is within bounds
  T_p = max(TLowBound + 0.0001f, min(T_p, TUpBound - 0.0001f));
  int nxT0 = find_nx0(T, N_T, T_p);

 
  float dt_yrs = dt * unit_time_sec / 3600.0 / 24.0 / 365.25;
  dt_yrs = max(timeLowBound + 0.0001f, min(dt_yrs, timeUpBound - 0.0001f)); 
  int nxtime0 = find_nx0(tarr, N_t, dt_yrs)

  int nxnH1 = nxnH0 + 1;
  int nxr1 = nxr0 + 1;
  int nxNH1 = nxNH0 + 1;
  int nxT1 = nxT0 + 1;
  int nxtime1 = nxtime0 + 1;


  // Performing 4D interpolation
  float Gam_minus_Lam = interpolate_4d_hypercube(nH_p, T_p, r_p, NH_p, nH, T, r, NH, Gam, Lam, nxnH0, nxT0, nxr0, nxNH0,
                                                 nxnH1, nxT1, nxr1, nxNH1, N_nH, N_T, N_r, N_NH);

  return Gam_minus_Lam;
}
