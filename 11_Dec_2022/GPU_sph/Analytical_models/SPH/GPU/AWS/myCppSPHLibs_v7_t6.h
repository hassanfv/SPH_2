#ifndef MYCPPSPHLIBS_H
#define MYCPPSPHLIBS_H

// Several BH related functions added here. (15 May 2023).
// acc_g_sph_gas_dm function is added to account for the presence of DM particles with different number than gas particles (14 May 2023).
// A particle can maximally lose half its internal energy in a time step. Gadget 2 code sec. 5.5.
// my_pi is replaced by M_PI. (28 Jan 2023).

//========================================
//====== smoothing_h_individual_dt =======  Updated: 28 Jan 2023. h_new adopted from eq.31 in Gadget2 Paper
//========================================
__global__ void smoothing_h_individual_dt(float *x, float *y, float *z, float *hres, float *hprevious,
                                          int N, int Ndown, int Nup, float coeff,
                                          float Nngb_f, float *Nngb_previous, float *divV, float dt,
                                          int *activeParticles)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float h_new = 2.0f * (0.5f * hprevious[i] * (1.0f + pow((Nngb_f / Nngb_previous[i]), 1.0f / 3.0f)) +
                          1.0f / 3.0f * hprevious[i] * divV[i] * dt);
    float h_tmp = h_new;
    int N_iter = 0;
    int k = 0;

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {
        dx = x[j] - x[i];
        dy = y[j] - y[i];
        dz = z[j] - z[i];
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          k++;
        }
      }

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * hprevious[i];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * hprevious[i];
      }

      if (h_new > h_tmp)
      {
        h_tmp = h_new;
      }

      N_iter++;
      if (N_iter > 100)
      {
        h_new = h_tmp;
        break;
      }
    }
    Nngb_previous[i] = k;
    hres[i] = 0.5 * h_new;
  }
}

//========================================
//======== extrapolate_smoothing_h =======
//========================================
__global__ void extrapolate_smoothing_h(float *hres, float *hprevious, float *dh_dt,
                                        float *Nngb_previous, float *divV, float Nngb_f,
                                        int NGx, int *NGroupx, float dt)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < NGx)
  {
    int ii = NGroupx[i];
    hres[ii] = (0.5f * hprevious[ii] * (1.0f + pow((Nngb_f / Nngb_previous[ii]), 1.0f / 3.0f)) +
                1.0f / 3.0f * hprevious[ii] * divV[ii] * dt);
  }
}

//======================================== (3 - Feb - 2023): imporoved the approach!
//========= Smoothing LengthX2 =========== (1 - Feb - 2023): Modified for individual timeSteps.
//========================================
__global__ void smoothing_hX2(float *x, float *y, float *z, float *hres, float *hprevious,
                              int N, int Ndown, int Nup, float coeff,
                              float Nngb_f, float *Nngb_previous, float *divV,
                              int NN, int *NGroupx, float dt)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < NN)
  {

    int ii = NGroupx[i];
    float h_new = 2.0f * (0.5f * hprevious[ii] * (1.0f + pow((Nngb_f / Nngb_previous[ii]), 1.0f / 3.0f)) +
                          1.0f / 3.0f * hprevious[ii] * divV[ii] * dt);
    float h_tmp = h_new;
    int N_iter = 0;
    int k = 0;

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {
        dx = x[j] - x[ii];
        dy = y[j] - y[ii];
        dz = z[j] - z[ii];
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          k++;
        }
      }

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * hprevious[ii];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * hprevious[ii];
      }

      if (h_new > h_tmp)
      {
        h_tmp = h_new;
      }

      N_iter++;
      if (N_iter > 100)
      {
        h_new = h_tmp;
        break;
      }
    }

    Nngb_previous[ii] = k;
    hres[ii] = 0.5 * h_new;
  }

  if ((i >= NN) && (i < N)) // Extrapolate!
  {
    int ii = NGroupx[i];
    hres[ii] = (0.5f * hprevious[ii] * (1.0f + pow((Nngb_f / Nngb_previous[ii]), 1.0f / 3.0f)) +
                1.0f / 3.0f * hprevious[ii] * divV[ii] * dt);
  }
}

//========================================
//========== Smoothing LengthX ============ (1 - Feb - 2023): Modified for individual timeSteps.
//========================================
__global__ void smoothing_hX(float *x, float *y, float *z, float *hres, float *hprevious,
                             int N, int Ndown, int Nup, float coeff,
                             float Nngb_f, float *Nngb_previous, float *divV,
                             int NGx, int *NGroupx, float dt)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < NGx)
  {

    int ii = NGroupx[i];
    float h_new = 2.0f * (0.5f * hprevious[ii] * (1.0f + pow((Nngb_f / Nngb_previous[ii]), 1.0f / 3.0f)) +
                          1.0f / 3.0f * hprevious[ii] * divV[ii] * dt);
    float h_tmp = h_new;
    int N_iter = 0;
    int k = 0;

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {
        dx = x[j] - x[ii];
        dy = y[j] - y[ii];
        dz = z[j] - z[ii];
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          k++;
        }
      }

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * hprevious[ii];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * hprevious[ii];
      }

      if (h_new > h_tmp)
      {
        h_tmp = h_new;
      }

      N_iter++;
      if (N_iter > 100)
      {
        h_new = h_tmp;
        break;
      }
    }
    Nngb_previous[ii] = k;
    hres[ii] = 0.5 * h_new;
  }
}

//========================================
//========== Smoothing Length ============ Updated: 28 Jan 2023. h_new adopted from eq.31 in Gadget2 Paper
//========================================
__global__ void smoothing_h(float *x, float *y, float *z, float *hres, float *hprevious,
                            int N, int Ndown, int Nup, float coeff,
                            float Nngb_f, float *Nngb_previous, float *divV, float dt)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float h_new = 2.0f * (0.5f * hprevious[i] * (1.0f + pow((Nngb_f / Nngb_previous[i]), 1.0f / 3.0f)) +
                          1.0f / 3.0f * hprevious[i] * divV[i] * dt);
    float h_tmp = h_new;
    int N_iter = 0;
    int k = 0;

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {
        dx = x[j] - x[i];
        dy = y[j] - y[i];
        dz = z[j] - z[i];
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          k++;
        }
      }

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * hprevious[i];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * hprevious[i];
      }

      if (h_new > h_tmp)
      {
        h_tmp = h_new;
      }

      N_iter++;
      if (N_iter > 100)
      {
        h_new = h_tmp;
        break;
      }
    }
    Nngb_previous[i] = k;
    hres[i] = 0.5 * h_new;
  }
}

//========================================
//======= Smoothing Length (old) =========
//========================================
__global__ void smoothing_h_old(float *x, float *y, float *z, float *hres, float *hprevious,
                                int N, int Ndown, int Nup, float coeff)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float h_new = 2.0f * hprevious[i]; // Change it to 2.0 in REAL App !!!!!!!!
    float h_tmp = h_new;
    int N_iter = 0;
    int k = 0;

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {
        dx = x[j] - x[i];
        dy = y[j] - y[i];
        dz = z[j] - z[i];
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          k++;
        }
      }

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * hprevious[i];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * hprevious[i];
      }

      if (h_new > h_tmp)
      {
        h_tmp = h_new;
      }

      N_iter++;
      if (N_iter > 100)
      {
        h_new = h_tmp;
        break;
      }
    }
    hres[i] = 0.5 * h_new;
  }
}



/*
//========================================
//======= Smoothing Length (BH) =========
//========================================
__global__ void smoothing_h_BH(float *x, float *y, float *z, float &hBH,
                                int N, int Ndown, int Nup, float coeff)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i == 0) // Just for one particle, i.e. BH!
  {

    float h_new = 2.0f * hBH;
    float h_tmp = h_new;
    int N_iter = 0;
    int k = 0;
    
    float xBH = 0.0f;
    float yBH = 0.0f;
    float zBH = 0.0f;

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {
        dx = x[j] - xBH;
        dy = y[j] - yBH;
        dz = z[j] - zBH;
        float rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          k++;
        }
      }

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * hBH;
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * hBH;
      }

      if (h_new > h_tmp)
      {
        h_tmp = h_new;
      }

      N_iter++;
      if (N_iter > 100)
      {
        h_new = h_tmp;
        break;
      }
    }
    hBH = 0.5 * h_new;
  }
}
*/



//==========================================
//============ getDensity ==================
//==========================================
__global__ void getDensity(float *x, float *y, float *z, float *mass,
                           float *rho, float *h, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float dx, dy, dz, rr, hij, sig, q, hij3;
    float WIij;
    float ss = 0.0f;

    for (int j = 0; j < N; j++)
    {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];

      rr = sqrt(dx * dx + dy * dy + dz * dz);
      hij = 0.5f * (h[i] + h[j]);

      if (rr <= 2.0f * hij)
      {

        sig = 1.0 / M_PI;
        q = rr / hij;
        hij3 = hij * hij * hij;
        WIij = 0.0f;

        if (q <= 1.0)
        {
          WIij = sig / hij3 * (1.0f - (3.0f / 2.0f) * q * q + (3.0f / 4.0f) * q * q * q);
        }

        if ((q > 1.0f) && (q <= 2.0))
        {
          WIij = sig / hij3 * (1.0f / 4.0f) * (2.0f - q) * (2.0f - q) * (2.0f - q);
        }

        ss += mass[j] * WIij;
      }
    }
    rho[i] = ss;
  }
}


//==========================================
//============ getWeights_BH ==================
//==========================================
__global__ void getWeights_BH(float *x, float *y, float *z, float *mass,
                           float *weightsBH, float hBH, float *h, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float dx, dy, dz, rr, hij, sig, q, hij3;
    float WIij;
    float ss = 0.0f;
    
    float xBH = 0.0f;
    float yBH = 0.0f;
    float zBH = 0.0f;

    dx = x[i] - xBH;
    dy = y[i] - yBH;
    dz = z[i] - zBH;

    rr = sqrt(dx * dx + dy * dy + dz * dz);
    hij = 0.5f * (h[i] + hBH);

    if (rr <= 2.0f * hij)
    {

      sig = 1.0 / M_PI;
      q = rr / hij;
      hij3 = hij * hij * hij;
      WIij = 0.0f;

      if (q <= 1.0)
      {
        WIij = sig / hij3 * (1.0f - (3.0f / 2.0f) * q * q + (3.0f / 4.0f) * q * q * q);
      }

      if ((q > 1.0f) && (q <= 2.0))
      {
        WIij = sig / hij3 * (1.0f / 4.0f) * (2.0f - q) * (2.0f - q) * (2.0f - q);
      }

      weightsBH[i] = WIij;
    }
  }
}





//==============================================
//================= acc_gX =====================
//==============================================
__global__ void acc_gX(float *x, float *y, float *z, float *eps, float *accx,
                       float *accy, float *accz, float *mass, float G, int N,
                       int NN, int *NGroupx)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < NN) // Only updating for active particles !!
  {

    int ii = NGroupx[i];

    float dx, dy, dz, rr, inv_r3, epsij, q, q2, q3, q4, q5, q6, fk;
    float accxt = 0.0f, accyt = 0.0f, acczt = 0.0f;
    for (int j = 0; j < N; j++)
    {
      dx = x[j] - x[ii];
      dy = y[j] - y[ii];
      dz = z[j] - z[ii];

      rr = sqrt(dx * dx + dy * dy + dz * dz);
      inv_r3 = 1.0f / (rr * rr * rr + 1e-5);
      epsij = 0.5f * (eps[ii] + eps[j]);
      q = rr / epsij;
      q2 = q * q;
      q3 = q2 * q;
      q4 = q3 * q;
      q5 = q4 * q;
      q6 = q5 * q;

      if (q <= 1.0f)
      {
        fk = (1.0f / (epsij * epsij * epsij)) * ((4.0f / 3.0f) - (6.0f / 5.0f) * q2 + (1.0f / 2.0f) * q3);
      }

      if ((q > 1.0f) && (q <= 2.0f))
      {
        fk = inv_r3 * ((-1.0f / 15.0f) + (8.0f / 3.0f) * q3 - 3.0f * q4 + (6.0f / 5.0f) * q5 - (1.0f / 6.0f) * q6);
      }

      if (q > 2.0f)
      {
        fk = inv_r3;
      }

      accxt += G * fk * dx * mass[j];
      accyt += G * fk * dy * mass[j];
      acczt += G * fk * dz * mass[j];
    }
    accx[ii] = accxt;
    accy[ii] = accyt;
    accz[ii] = acczt;
  }
}

//==============================================
//================= acc_g ======================
//==============================================
__global__ void acc_g(float *x, float *y, float *z, float *eps, float *accx,
                      float *accy, float *accz, float *mass, float G, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float dx, dy, dz, rr, inv_r3, epsij, q, q2, q3, q4, q5, q6, fk;
    float accxt = 0.0f, accyt = 0.0f, acczt = 0.0f;
    for (int j = 0; j < N; j++)
    {
      dx = x[j] - x[i];
      dy = y[j] - y[i];
      dz = z[j] - z[i];

      rr = sqrt(dx * dx + dy * dy + dz * dz);
      inv_r3 = 1.0f / (rr * rr * rr + 1e-5);
      epsij = 0.5f * (eps[i] + eps[j]);
      q = rr / epsij;
      q2 = q * q;
      q3 = q2 * q;
      q4 = q3 * q;
      q5 = q4 * q;
      q6 = q5 * q;

      if (q <= 1.0f)
      {
        fk = (1.0f / (epsij * epsij * epsij)) * ((4.0f / 3.0f) - (6.0f / 5.0f) * q2 + (1.0f / 2.0f) * q3);
      }

      if ((q > 1.0f) && (q <= 2.0f))
      {
        fk = inv_r3 * ((-1.0f / 15.0f) + (8.0f / 3.0f) * q3 - 3.0f * q4 + (6.0f / 5.0f) * q5 - (1.0f / 6.0f) * q6);
      }

      if (q > 2.0f)
      {
        fk = inv_r3;
      }

      accxt += G * fk * dx * mass[j];
      accyt += G * fk * dy * mass[j];
      acczt += G * fk * dz * mass[j];
    }
    accx[i] = accxt;
    accy[i] = accyt;
    accz[i] = acczt;
  }
}

//===================================================
//================== getPressure ====================
//===================================================
__global__ void getPressure(float *P, float *rho, float T_cld, float T_ps,
                            float T_0, float kBmH2, float UnitDensity_in_cgs,
                            float Unit_P_in_cgs, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float rhot = rho[i] * UnitDensity_in_cgs;

    if (rhot <= 1e-21)
    {
      P[i] = rhot * kBmH2 * T_cld / Unit_P_in_cgs;
    }

    if ((rhot > 1e-21) && (rhot <= 2e-21))
    {
      P[i] = rhot * kBmH2 * gammah * T_cld * pow((rhot / 2e-21), (gammah - 1.0f)) / Unit_P_in_cgs;
    }

    if ((rhot > 2e-21) && (rhot <= 1e-18))
    {
      P[i] = rhot * kBmH2 * T_ps / Unit_P_in_cgs;
    }

    if (rhot > 1e-18)
    {
      P[i] = rhot * kBmH2 * T_0 * (1.0f + gammah * pow((rhot / 1e-14), (gammah - 1.0f))) / Unit_P_in_cgs;
    }
  }
}

//===============================================
//================= getCsound ===================
//===============================================
__global__ void getCsound(float *csnd, float *rho, float T_cld, float T_ps, float T_0,
                          float kBmH2, float UnitDensity_in_cgs, float unitVelocity,
                          float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float rhot = rho[i] * UnitDensity_in_cgs;

    if (rhot <= 1e-21)
    {
      csnd[i] = sqrt(kBmH2 * T_cld) / unitVelocity;
    }

    if ((rhot > 1e-21) && (rhot <= 2e-21))
    {
      csnd[i] = sqrt(kBmH2 * gammah * T_cld * pow((rhot / 2e-21), (gammah - 1.0f))) / unitVelocity;
    }

    if ((rhot > 2e-21) && (rhot <= 1e-18))
    {
      csnd[i] = sqrt(kBmH2 * T_ps) / unitVelocity;
    }

    if (rhot > 1e-18)
    {
      csnd[i] = sqrt(kBmH2 * T_0 * (1.0f + gammah * pow((rhot / 1e-14), (gammah - 1.0f)))) / unitVelocity;
    }
  }
}

//=====================================================
//================== div_curlVelX ======================
//=====================================================
__global__ void div_curlVelX(float *divV, float *curlV, float *x, float *y, float *z,
                             float *vx, float *vy, float *vz, float *rho, float *mass,
                             float *h, int N, int NN, int *NGroupx)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < NN)
  {

    int ii = NGroupx[i];

    float dx, dy, dz, rr, hij, q, vxji, vyji, vzji, hij5, sig;
    float nW = 0.0f;
    float gWx = 0.0f;
    float gWy = 0.0f;
    float gWz = 0.0f;
    float vxij, vyij, vzij;
    float ss = 0.0f;
    float curlVx = 0.0f;
    float curlVy = 0.0f;
    float curlVz = 0.0f;

    for (int j = 0; j < N; j++)
    {
      dx = x[j] - x[ii];
      dy = y[j] - y[ii];
      dz = z[j] - z[ii];

      rr = sqrt(dx * dx + dy * dy + dz * dz);
      hij = 0.5f * (h[ii] + h[j]);
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

        vxji = vx[j] - vx[ii];
        vyji = vy[j] - vy[ii];
        vzji = vz[j] - vz[ii];

        ss += mass[j] / rho[ii] * (vxji * gWx + vyji * gWy + vzji * gWz);

        vxij = vx[ii] - vx[j]; //-vxji;
        vyij = vy[ii] - vy[j]; //-vyji;
        vzij = vz[ii] - vz[j]; //-vzji;

        curlVx += mass[j] / rho[ii] * (vyij * gWz - vzij * gWy); // eq. 18 in Beck et al. 2016.
        curlVy += mass[j] / rho[ii] * (vzij * gWx - vxij * gWz);
        curlVz += mass[j] / rho[ii] * (vxij * gWy - vyij * gWx);
      }
    }
    divV[ii] = ss; // abs(ss);
    curlV[ii] = sqrt(curlVx * curlVx + curlVy * curlVy + curlVz * curlVz);
  }
}

//=====================================================
//================== div_curlVel ======================
//=====================================================
__global__ void div_curlVel(float *divV, float *curlV, float *x, float *y, float *z,
                            float *vx, float *vy, float *vz, float *rho, float *mass,
                            float *h, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float dx, dy, dz, rr, hij, q, vxji, vyji, vzji, hij5, sig;
    float nW = 0.0f;
    float gWx = 0.0f;
    float gWy = 0.0f;
    float gWz = 0.0f;
    float vxij, vyij, vzij;
    float ss = 0.0f;
    float curlVx = 0.0f;
    float curlVy = 0.0f;
    float curlVz = 0.0f;

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

        vxji = vx[j] - vx[i];
        vyji = vy[j] - vy[i];
        vzji = vz[j] - vz[i];

        ss += mass[j] / rho[i] * (vxji * gWx + vyji * gWy + vzji * gWz);

        vxij = vx[i] - vx[j]; //-vxji;
        vyij = vy[i] - vy[j]; //-vyji;
        vzij = vz[i] - vz[j]; //-vzji;

        curlVx += mass[j] / rho[i] * (vyij * gWz - vzij * gWy); // eq. 18 in Beck et al. 2016.
        curlVy += mass[j] / rho[i] * (vzij * gWx - vxij * gWz);
        curlVz += mass[j] / rho[i] * (vxij * gWy - vyij * gWx);
      }
    }
    divV[i] = ss; // abs(ss);
    curlV[i] = sqrt(curlVx * curlVx + curlVy * curlVy + curlVz * curlVz);
  }
}

//===========================================================
//====================== acc_sphX ===========================
//===========================================================
__global__ void acc_sphX(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                         float *h, float *c, float *rho, float *divV, float *curlV,
                         float *mass, float *P, float *ax, float *ay, float *az,
                         float visc_alpha, int N, int NN, int *NGroupx)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < NN)
  {

    int ii = NGroupx[i];

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float axt = 0.0f;
    float ayt = 0.0f;
    float azt = 0.0f;

    for (int j = 0; j < N; j++)
    {
      dx = x[ii] - x[j];
      dy = y[ii] - y[j];
      dz = z[ii] - z[j];

      rr = sqrt(dx * dx + dy * dy + dz * dz);

      hij = 0.5f * (h[ii] + h[j]);

      if (rr < 2.0f * hij)
      {

        nW = 0.0f;
        gWx = 0.0f;
        gWy = 0.0f;
        gWz = 0.0f;
        sig = 1.0f / M_PI;
        hij5 = hij * hij * hij * hij * hij;
        q = rr / hij;

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

        //-------- PIij ---------
        vxij = vx[ii] - vx[j];
        vyij = vy[ii] - vy[j];
        vzij = vz[ii] - vz[j];

        vij_rij = vxij * dx + vyij * dy + vzij * dz;

        float cij = 0.5f * (c[ii] + c[j]);

        wij = vij_rij / (rr + 1e-5);
        vij_sig = c[ii] + c[j] - 3.0f * wij;
        rhoij = 0.5f * (rho[ii] + rho[j]);

        PIij = 0.0f;
        if (vij_rij <= 0.0f)
        {

          PIij = -0.5f * visc_alpha * vij_sig * wij / rhoij;

          //------- Shear-viscosity correction -------
          fi = abs(divV[ii]) / (abs(divV[ii]) + curlV[ii] + 0.0001 * c[ii] / h[ii]);
          fj = abs(divV[j]) / (abs(divV[j]) + curlV[j] + 0.0001 * c[j] / h[j]);
          fij = 0.5f * (fi + fj);
          PIij = fij * PIij;
          //------- End of Shear-visc. correction -----
        }

        axt -= mass[j] * (P[ii] / rho[ii] / rho[ii] + P[j] / rho[j] / rho[j] + PIij) * gWx;
        ayt -= mass[j] * (P[ii] / rho[ii] / rho[ii] + P[j] / rho[j] / rho[j] + PIij) * gWy;
        azt -= mass[j] * (P[ii] / rho[ii] / rho[ii] + P[j] / rho[j] / rho[j] + PIij) * gWz;
      }
    }
    ax[ii] = axt;
    ay[ii] = ayt;
    az[ii] = azt;
  }
}

//===========================================================
//====================== acc_sph ============================
//===========================================================
__global__ void acc_sph(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *h, float *c, float *rho, float *divV, float *curlV,
                        float *mass, float *P, float *ax, float *ay, float *az,
                        float visc_alpha, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float axt = 0.0f;
    float ayt = 0.0f;
    float azt = 0.0f;

    for (int j = 0; j < N; j++)
    {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];

      rr = sqrt(dx * dx + dy * dy + dz * dz);

      hij = 0.5f * (h[i] + h[j]);

      if (rr < 2.0f * hij)
      {

        nW = 0.0f;
        gWx = 0.0f;
        gWy = 0.0f;
        gWz = 0.0f;
        sig = 1.0f / M_PI;
        hij5 = hij * hij * hij * hij * hij;
        q = rr / hij;

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

        //-------- PIij ---------
        vxij = vx[i] - vx[j];
        vyij = vy[i] - vy[j];
        vzij = vz[i] - vz[j];

        vij_rij = vxij * dx + vyij * dy + vzij * dz;

        float cij = 0.5f * (c[i] + c[j]);

        wij = vij_rij / (rr + 1e-5);
        vij_sig = c[i] + c[j] - 3.0f * wij;
        rhoij = 0.5f * (rho[i] + rho[j]);

        PIij = 0.0f;
        if (vij_rij <= 0.0f)
        {

          PIij = -0.5f * visc_alpha * vij_sig * wij / rhoij;

          //------- Shear-viscosity correction -------
          fi = abs(divV[i]) / (abs(divV[i]) + curlV[i] + 0.0001 * c[i] / h[i]);
          fj = abs(divV[j]) / (abs(divV[j]) + curlV[j] + 0.0001 * c[j] / h[j]);
          fij = 0.5f * (fi + fj);
          PIij = fij * PIij;
          //------- End of Shear-visc. correction -----
        }

        axt -= mass[j] * (P[i] / rho[i] / rho[i] + P[j] / rho[j] / rho[j] + PIij) * gWx;
        ayt -= mass[j] * (P[i] / rho[i] / rho[i] + P[j] / rho[j] / rho[j] + PIij) * gWy;
        azt -= mass[j] * (P[i] / rho[i] / rho[i] + P[j] / rho[j] / rho[j] + PIij) * gWz;
      }
    }
    ax[i] = axt;
    ay[i] = ayt;
    az[i] = azt;
  }
}

//===========================================================
//====================== acc_tot ============================
//===========================================================
__global__ void acc_g_sph(float *acc_totx, float *acc_toty, float *acc_totz,
                          float *acc_gx, float *acc_gy, float *acc_gz,
                          float *acc_sphx, float *acc_sphy, float *acc_sphz,
                          int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    acc_totx[i] = acc_gx[i] + acc_sphx[i];
    acc_toty[i] = acc_gy[i] + acc_sphy[i];
    acc_totz[i] = acc_gz[i] + acc_sphz[i];
  }
}


//===========================================================
//====================== acc_tot (Gas + DM) ============================
//===========================================================
__global__ void acc_g_sph_gas_dm(float *acc_totx, float *acc_toty, float *acc_totz,
                          float *acc_gx, float *acc_gy, float *acc_gz,
                          float *acc_sphx, float *acc_sphy, float *acc_sphz,
                          int N, int NGas)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    acc_totx[i] = acc_gx[i];
    acc_toty[i] = acc_gy[i];
    acc_totz[i] = acc_gz[i];
  }
  
  if (i < NGas)
  {
    acc_totx[i] = acc_gx[i] + acc_sphx[i];
    acc_toty[i] = acc_gy[i] + acc_sphy[i];
    acc_totz[i] = acc_gz[i] + acc_sphz[i];
  }
  
}

//===========================================================
//================= velocity evolution ======================
//===========================================================
__global__ void v_evolve(float *vx, float *vy, float *vz,
                         float *accx, float *accy, float *accz,
                         float dt, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    vx[i] += accx[i] * dt / 2.0f;
    vy[i] += accy[i] * dt / 2.0f;
    vz[i] += accz[i] * dt / 2.0f;
  }
}

//===========================================================
//================= position evolution ======================
//===========================================================
__global__ void r_evolve(float *x, float *y, float *z,
                         float *vx, float *vy, float *vz,
                         float dt, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
}



//===========================================================
//================= position evolution (BH fixed at [0, 0, 0]) ======================
//===========================================================
__global__ void r_evolve_BH_fixed(float *x, float *y, float *z,
                         float *vx, float *vy, float *vz,
                         float dt, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
  }
  
  if (i == N-1) // Setting BH position fixed at [0, 0, 0]
  {
    x[i] = 0.0f;
    y[i] = 0.0f;
    z[i] = 0.0f;
  }
  
}



//===========================================================
//================= hprevious updater =======================
//===========================================================
__global__ void hprevious_updater(float *hprevious, float *h, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    hprevious[i] = h[i];
  }
}

//===========================================================
//=================== dt estimation =========================
//===========================================================
__global__ void dt_array_indiv_dt(float *x, float *y, float *z,
                                  float *vx, float *vy, float *vz,
                                  float *accx, float *accy, float *accz,
                                  float *accx_tot, float *accy_tot, float *accz_tot,
                                  float *h, float *c, float *dt_particles,
                                  float *abs_acc_g, float *abs_acc_tot,
                                  float *divV, float *dh_dt, float C_CFL,
                                  float visc_alpha, float *eps, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    abs_acc_g[i] = sqrt(accx[i] * accx[i] + accy[i] * accy[i] + accz[i] * accz[i]);
    abs_acc_tot[i] = sqrt(accx_tot[i] * accx_tot[i] + accy_tot[i] * accy_tot[i] + accz_tot[i] * accz_tot[i]);

    float dx, dy, dz, vxij, vyij, vzij, wij, vij_rij, vij_sig, rr;

    float max_vij_sig = 0.0f;
    float tmp = 0.0f;
    for (int j = 0; j < N; j++)
    {

      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];

      rr = sqrt(dx * dx + dy * dy + dz * dz);

      vxij = vx[i] - vx[j];
      vyij = vy[i] - vy[j];
      vzij = vz[i] - vz[j];

      vij_rij = vxij * dx + vyij * dy + vzij * dz;

      wij = vij_rij / (rr + 1e-5);

      vij_sig = c[i] + c[j] - 3.0f * wij;

      tmp = vij_sig;

      if (tmp > max_vij_sig)
      {
        max_vij_sig = tmp;
      }
    }

    float dt_hyd = C_CFL * h[i] / max_vij_sig; // eq. 16 in Springel et al - 2005.

    // dt_cour: ref: Gadget 2 paper, eq. 49.
    float abs_v_i = sqrt(vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    float dt_cour = C_CFL * h[i] / (h[i] * abs(divV[i]) + max(c[i], abs_v_i) * (1.0f + 0.6f * visc_alpha));

    // dt_G1, dt_G2 ref: Dave et al - 1997 (Parallel TreeSPH) sec. 3.7.
    float ettaa = 0.4;
    float dt_G1 = ettaa * sqrt(eps[i] / abs_acc_tot[i]);
    float dt_G2 = ettaa * eps[i] / abs_v_i;

    float dt_f = sqrt(h[i] / abs_acc_g[i]);
    float dt_kin = sqrt(h[i] / abs_acc_tot[i]);

    float dtZ[6] = {dt_hyd, dt_cour, dt_f, dt_kin, dt_G1, dt_G2}; // Note to modify the max range in for loop
    float dtxx = dtZ[0];
    // finding the minimum !!
    for (int k = 0; k < 6; k++) // modify if you changed dtZ[]!!! IMPORTANT!!
    {
      if (dtZ[k] <= dtxx)
      {
        dtxx = dtZ[k];
      }
    }
    dt_particles[i] = dtxx;

    dh_dt[i] = 1.0f / 3.0f * h[i] * divV[i]; // See the line below eq.31 in Gadget 2 paper.
  }
}

//===================================================
//============ getPressure (Kitsonias) ==============
//===================================================
__global__ void getPressure_Kitsonias(float *P, float *rho, float UnitDensity_in_cgs,
                                      float Unit_P_in_cgs, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float const c_0 = 110472.5;
    float const rho_0 = 6.66e-22;
    float const c_s = 0.2e5;
    float const rho_1 = 1.0e-14;

    float rhot = rho[i] * UnitDensity_in_cgs;

    if (rhot <= rho_0)
    {
      P[i] = rhot * c_0 * c_0 / Unit_P_in_cgs;
    }

    if (rhot > rho_0)
    {
      P[i] = rhot * ((c_0 * c_0 - c_s * c_s) * pow((rhot / rho_0), (-2.0f / 3.0f)) + c_s * c_s) *
             sqrt(1.0f + pow((rhot / rho_1), (4.0f / 3.0f))) / Unit_P_in_cgs;
    }
  }
}

//===============================================
//============ getCsound (Kitsonias) ============
//===============================================
__global__ void getCsound_Kitsonias(float *csnd, float *rho, float UnitDensity_in_cgs,
                                    float unitVelocity, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float const c_0 = 110472.5;
    float const rho_0 = 6.66e-22;
    float const c_s = 0.2e5;
    float const rho_1 = 1.0e-14;

    float rhot = rho[i] * UnitDensity_in_cgs;

    if (rhot <= rho_0)
    {
      csnd[i] = c_0 / unitVelocity;
    }

    if (rhot > rho_0)
    {
      csnd[i] = sqrt(((c_0 * c_0 - c_s * c_s) * pow((rhot / rho_0), (-2.0f / 3.0f)) + c_s * c_s) *
                     sqrt(1.0f + pow((rhot / rho_1), (4.0f / 3.0f)))) /
                unitVelocity;
    }
  }
}

//===================================================
//============= getPressure (Arreaga) ===============
//===================================================
__global__ void getPressure_Arreaga(float *P, float *rho, float UnitDensity_in_cgs,
                                    float Unit_P_in_cgs, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float c_iso = 1.66e4;
    float rho_crit = 5.0e-14;

    float rhot = rho[i] * UnitDensity_in_cgs;

    P[i] = rhot * c_iso * c_iso * (1.0f + pow(rhot / rho_crit, gammah - 1.0f)) / Unit_P_in_cgs;
  }
}

//===============================================
//============= getCsound (Arreaga) =============
//===============================================
__global__ void getCsound_Arreaga(float *csnd, float *rho, float UnitDensity_in_cgs,
                                  float unitVelocity, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float c_iso = 1.66e4;
    float rho_crit = 5.0e-14;

    float rhot = rho[i] * UnitDensity_in_cgs;

    csnd[i] = c_iso * sqrt(1.0f + pow(rhot / rho_crit, gammah - 1.0f)) / unitVelocity;
  }
}

//===============================================
//=================== get_dUX ===================
//===============================================
__global__ void get_dUX(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *h, float *c, float *rho, float *divV, float *curlV,
                        float *mass, float *P, float *dudt,
                        float visc_alpha, int N, int NN, int *NGroupx)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < NN)
  {

    int ii = NGroupx[i];

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW, vij_gWij;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float dut = 0.0f;

    for (int j = 0; j < N; j++)
    {
      dx = x[ii] - x[j];
      dy = y[ii] - y[j];
      dz = z[ii] - z[j];

      rr = sqrt(dx * dx + dy * dy + dz * dz);

      hij = 0.5f * (h[ii] + h[j]);

      if (rr < 2.0f * hij)
      {

        nW = 0.0f;
        gWx = 0.0f;
        gWy = 0.0f;
        gWz = 0.0f;
        sig = 1.0f / M_PI;
        hij5 = hij * hij * hij * hij * hij;
        q = rr / hij;

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

        //-------- PIij ---------
        vxij = vx[ii] - vx[j];
        vyij = vy[ii] - vy[j];
        vzij = vz[ii] - vz[j];

        vij_gWij = vxij * gWx + vyij * gWy + vzij * gWz;

        vij_rij = vxij * dx + vyij * dy + vzij * dz;

        float cij = 0.5f * (c[ii] + c[j]);

        wij = vij_rij / (rr + 1e-5);
        vij_sig = c[ii] + c[j] - 3.0f * wij;
        rhoij = 0.5f * (rho[ii] + rho[j]);

        PIij = 0.0f;
        if (vij_rij <= 0.0f)
        {

          PIij = -0.5f * visc_alpha * vij_sig * wij / rhoij;

          //------- Shear-viscosity correction -------
          fi = abs(divV[ii]) / (abs(divV[ii]) + curlV[ii] + 0.0001 * c[ii] / h[ii]);
          fj = abs(divV[j]) / (abs(divV[j]) + curlV[j] + 0.0001 * c[j] / h[j]);
          fij = 0.5f * (fi + fj);
          PIij = fij * PIij;
          //------- End of Shear-visc. correction -----
        }
        dut += mass[j] * (P[ii] / rho[ii] / rho[ii] + PIij / 2.0f) * vij_gWij;
      }
    }
    dudt[ii] = dut;
  }
}

//===============================================
//=================== get_dU ====================
//===============================================
__global__ void get_dU(float *x, float *y, float *z, float *vx, float *vy, float *vz,
                       float *h, float *c, float *rho, float *divV, float *curlV,
                       float *mass, float *P, float *dudt,
                       float visc_alpha, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW, vij_gWij;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float dut = 0.0f;

    for (int j = 0; j < N; j++)
    {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];

      rr = sqrt(dx * dx + dy * dy + dz * dz);

      hij = 0.5f * (h[i] + h[j]);

      if (rr < 2.0f * hij)
      {

        nW = 0.0f;
        gWx = 0.0f;
        gWy = 0.0f;
        gWz = 0.0f;
        sig = 1.0f / M_PI;
        hij5 = hij * hij * hij * hij * hij;
        q = rr / hij;

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

        //-------- PIij ---------
        vxij = vx[i] - vx[j];
        vyij = vy[i] - vy[j];
        vzij = vz[i] - vz[j];

        vij_gWij = vxij * gWx + vyij * gWy + vzij * gWz;

        vij_rij = vxij * dx + vyij * dy + vzij * dz;

        float cij = 0.5f * (c[i] + c[j]);

        wij = vij_rij / (rr + 1e-5);
        vij_sig = c[i] + c[j] - 3.0f * wij;
        rhoij = 0.5f * (rho[i] + rho[j]);

        PIij = 0.0f;
        if (vij_rij <= 0.0f)
        {

          PIij = -0.5f * visc_alpha * vij_sig * wij / rhoij;

          //------- Shear-viscosity correction -------
          fi = abs(divV[i]) / (abs(divV[i]) + curlV[i] + 0.0001 * c[i] / h[i]);
          fj = abs(divV[j]) / (abs(divV[j]) + curlV[j] + 0.0001 * c[j] / h[j]);
          fij = 0.5f * (fi + fj);
          PIij = fij * PIij;
          //------- End of Shear-visc. correction -----
        }
        dut += mass[j] * (P[i] / rho[i] / rho[i] + PIij / 2.0f) * vij_gWij;
      }
    }
    dudt[i] = dut;
  }
}

//===============================================
//========= update u (Used only once!) ==========
//===============================================
__global__ void u_updater1(float *u, float *dudt, float dt, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    u[i] += dudt[i] * dt;
  }
}

//===============================================
//============ update u_ut_previous =============
//===============================================
__global__ void u_ut_previous_updater(float *u, float *dudt, float *uprevious,
                                      float *utprevious, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    uprevious[i] = u[i];
    utprevious[i] = dudt[i];
  }
}

//===============================================
//============= update u (Main one) =============
//===============================================
__global__ void u_updater_main(float *u, float *dudt, float *uprevious,
                               float *utprevious, float dt, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    u[i] = uprevious[i] + 0.5f * dt * (dudt[i] + utprevious[i]);
  }
}



//===============================================
//============= update u (AGN energy injection) =============
//===============================================
__global__ void u_AGN_energy_injection(float *u, float *weightsBH, float eps_AGN,
                               float L_Edd, float dt, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    u[i] = u[i] + eps_AGN * L_Edd * dt * weightsBH[i] * 5.0f ; // Injecting AGN energy! BH; Note that eps_AGN is the radiative efficiency.
  }
}



//===============================================
//============= update u (AGN energy and velocity injection) =============
//===============================================
__global__ void u_AGN_energy_velocity_injection(float *u, float *weightsBH, float *x, float *y, float *z,
                                                float *vx, float *vy, float *vz, float eps_AGN,
                                                float L_Edd, float dt, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    
    float ux, uy, uz;
    float v_r = 12.3f; // corresponding to 2,000 km/s.
  
    u[i] = u[i] + eps_AGN * L_Edd * dt * weightsBH[i] ; // Injecting AGN energy! BH; Note that eps_AGN is the radiative efficiency.
    
    if (weightsBH[i] != 0.0)
    {
      float rr = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
      ux = x[i] / rr; // unit vector indication directions.
      uy = y[i] / rr;
      uz = z[i] / rr;
      
      vx[i] = v_r * ux;
      vy[i] = v_r * uy;
      vz[i] = v_r * uz;
      
    }
  }
}


//===================================================
//============= getPressure (Adiabatic) =============
//===================================================
__global__ void getPressure_Adiabatic(float *P, float *rho, float *u, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    P[i] = (gammah - 1.0f) * rho[i] * u[i];
  }
}

//===============================================
//============= getCsound (Adiabatic) ===========
//===============================================
__global__ void getCsound_Adiabatic(float *csnd, float *u, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    csnd[i] = sqrt(gammah * (gammah - 1.0f) * u[i]);
  }
}

//===============================================
//================== applyCooling ===============
//===============================================
__global__ void applyCooling(float *uadT, float *rhoT, float *delta_u,
                             float ref_dt_cgs, float *uGrid, float *rhoGrid,
                             float MIN_uad, float MAX_uad, float MIN_rho,
                             float MAX_rho, float Unit_u_in_cgs,
                             float UnitDensity_in_cgs, float *ut, float *rhot,
                             float current_dt_cgs, int N, int N_u, int N_rho,
                             int NGrid)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    // Note: ut is the u before cooling.
    float coeff = current_dt_cgs / ref_dt_cgs;

    float ut_cgs = ut[i] * Unit_u_in_cgs;
    float rhot_cgs = rhot[i] * UnitDensity_in_cgs;

    if (ut_cgs < MIN_uad)
    {
      ut_cgs = MIN_uad + 0.001 * MIN_uad;
    }

    if (ut_cgs > MAX_uad)
    {
      ut_cgs = MAX_uad - 0.001 * MAX_uad;
    }

    if (rhot_cgs < MIN_rho)
    {
      rhot_cgs = MIN_rho + 0.001 * MIN_rho;
    }

    if (rhot_cgs > MAX_rho)
    {
      rhot_cgs = MAX_rho - 0.001 * MAX_rho;
    }

    int nx_u, nx_rho;

    //--- Getting u index
    for (int j = 0; j < N_u - 1; j++)
    {
      if ((ut_cgs >= uGrid[j]) && (ut_cgs <= uGrid[j + 1]))
      {
        nx_u = j;
        break;
      }
    }

    //--- Getting rho index
    for (int j = 0; j < N_rho - 1; j++)
    {
      if ((rhot_cgs >= rhoGrid[j]) && (rhot_cgs <= rhoGrid[j + 1]))
      {
        nx_rho = j;
        break;
      }
    }

    int nxx = nx_u * 1000 + nx_rho;

    float x_1 = uGrid[nx_u];
    float x_2 = uGrid[nx_u + 1];

    float y_1 = delta_u[nxx];
    float y_2 = delta_u[nxx + 1];

    float slope = (y_2 - y_1) / (x_2 - x_1);

    float yy = slope * (ut_cgs - x_1) + y_1;

    ut[i] = (ut_cgs - coeff * yy) / Unit_u_in_cgs; // updating u ! We converted it back to code unit !!!
  }
}

//===============================================
//=============== applyCloudyCooling ============
//===============================================
__global__ void applyCloudyCooling(float *uZ, float *nHZ, float *heatZ, float *coolZ,
                                   float *uGrid, float *nHGrid, float XH,
                                   float MIN_u, float MAX_u, float MIN_nH,
                                   float MAX_nH, float Unit_u_in_cgs,
                                   float UnitDensity_in_cgs, float *ut, float *rhot,
                                   float gamma, float mH, float kB, float *dudt_ad,
                                   float current_dt_cgs, int N, int N_u, int N_nH,
                                   int NGrid)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < N)
  {
    // Note: ut is the u before cooling.

    float ut_cgs = ut[i] * Unit_u_in_cgs;
    float rhot_cgs = rhot[i] * UnitDensity_in_cgs;

    float nH_cgs = XH * rhot_cgs / mH;

    if (ut_cgs <= MIN_u)
    {
      ut_cgs = MIN_u + 0.001 * MIN_u;
    }

    if (ut_cgs >= MAX_u)
    {
      ut_cgs = MAX_u - 0.001 * MAX_u;
    }

    if (nH_cgs <= MIN_nH)
    {
      nH_cgs = MIN_nH + 0.001 * MIN_nH;
    }

    if (nH_cgs >= MAX_nH)
    {
      nH_cgs = MAX_nH - 0.001 * MAX_nH;
    }

    int nx_u, nx_nH;

    //--- Getting u index
    for (int j = 0; j < N_u - 1; j++)
    {
      if ((ut_cgs >= uGrid[j]) && (ut_cgs <= uGrid[j + 1]))
      {
        nx_u = j;
        break;
      }
    }

    //--- Getting nH index
    for (int j = 0; j < N_nH - 1; j++)
    {
      if ((nH_cgs >= nHGrid[j]) && (nH_cgs <= nHGrid[j + 1]))
      {
        nx_nH = j;
        break;
      }
    }

    int nxx = nx_u * N_nH + nx_nH;

    float x_1 = uGrid[nx_u];
    float x_2 = uGrid[nx_u + 1];

    float y_1 = heatZ[nxx] - coolZ[nxx];
    float y_2 = heatZ[nxx + 1] - coolZ[nxx + 1];

    float slope = (y_2 - y_1) / (x_2 - x_1);

    float dudt_rad = slope * (ut_cgs - x_1) + y_1;
    float rate_coeff = 1.0f / rhot_cgs;

    dudt_rad = dudt_rad * rate_coeff;

    // Katz & Gunn 1991.
    /*
    float aa = 0.5f * ut_cgs / current_dt_cgs + dudt_ad[i] * Unit_u_in_cgs;
    float dudt_rad_damped = aa * dudt_rad / sqrt(aa * aa + dudt_rad * dudt_rad);
    ut[i] = (ut_cgs + dudt_rad_damped * current_dt_cgs) / Unit_u_in_cgs; // updating u ! We converted it back to code unit !!!
    */

    // updating u and converting it back to the code unit !!!
    float utmpx = (ut_cgs + dudt_rad * current_dt_cgs) / Unit_u_in_cgs;

    // a particle is allowed to maximally lose half of its internal energy in a single time step.
    // Gadget 2 paper - section 5.5, last paragraph.
    if (utmpx < 0.5f * ut[i])
    {
      ut[i] = 0.5f * ut[i];
    }
    else
    {
      ut[i] = utmpx;
    }
  }
}

#endif
