#ifndef HFVCPPLIBS_H
#define HFVCPPLIBS_H

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

//*************************************************************************
//*************** Function to save the OUTPUT Snap-Shots!! ****************
//*************************************************************************
void saveArraysToBinary(const std::string &filename, float *x, float *y, float *z, float *vx,
                        float *vy, float *vz, float *rho, float *h, float *u, float *mass,
                        int *Typ, int N)
{
  // Open the file in binary mode
  std::ofstream file(filename, std::ios::binary);

  // Check if the file was opened successfully
  if (!file)
  {
    std::cerr << "Failed to open file for writing: " << filename << std::endl;
    return;
  }

  // Write N and NG to the file
  file.write(reinterpret_cast<const char *>(&N), sizeof(int));

  // Write the arrays to the file
  file.write(reinterpret_cast<const char *>(Typ), N * sizeof(int));
  file.write(reinterpret_cast<const char *>(x), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(y), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(z), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(vx), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(vy), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(vz), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(rho), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(h), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(u), N * sizeof(float));
  file.write(reinterpret_cast<const char *>(mass), N * sizeof(float));

  // Close the file
  file.close();
}

//*******************************
//********* max_finder **********
//*******************************
float max_finder(int *Typ, float *arr, int N)
{

  float max_val = 0.0;
  for (int i = 0; i < N; i++)
  {
    if (Typ[i] == 0)
    {
      if (arr[i] >= max_val)
      {
        max_val = arr[i];
      }
    }
  }
  return max_val;
}

//*******************************
//********* min_finder **********
//*******************************
float min_finder(int *Typ, float *arr, int N)
{

  float min_val = 1e22; // used large value for the start! I did not want to use arr[0] as it could be 0.0 itself!
  for (int i = 0; i < N; i++)
  {
    if (Typ[i] == 0)
    {
      if (arr[i] <= min_val)
      {
        min_val = arr[i];
      }
    }
  }
  return min_val;
}

//*******************************
//********* dt_min_finder **********
//*******************************
float dt_min_finder(int *Typ, int *activeId, float *arr, float t, int N)
{

  float min_val = 1e3; // used large value for the start! I did not want to use arr[0] as it could be 0.0 itself!
  float tmp;

  for (int i = 0; i < N; i++)
  {
    if ((Typ[i] == 0) && (activeId[i] == 1))
    {
      tmp = arr[i] - t;
      if (tmp <= min_val)
      {
        min_val = tmp;
      }
    }
  }
  return min_val;
}

//========================================
//========== Smoothing Length ============ Updated: 28 Jan 2023. h_new adopted from eq.31 in Gadget2 Paper
//========================================
__global__ void smoothing_h(int *Typ, float *x, float *y, float *z, float *h,
                            int N, int Ndown, int Nup, float coeff,
                            float Nngb_f, float *Nngb_previous, float *divV, float dt)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float h_new = 2.0f * (0.5f * h[i] * (1.0f + pow((Nngb_f / Nngb_previous[i]), 1.0f / 3.0f)) +
                          1.0f / 3.0f * h[i] * divV[i] * dt);
    float h_tmp = h_new;
    int N_iter = 0;
    int k = 0;

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int j = 0; j < N; j++)
      {
        if (Typ[j] == 0)
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
      }

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * h[i];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * h[i];
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
    h[i] = 0.5 * h_new;
  }
}

//==========================================
//========== Set eps of Gas to h ===========
//==========================================
__global__ void set_eps_of_gas_to_h(int *Typ, float *eps, float *h, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    eps[i] = h[i];
  }
}

//==========================================
//============ getDensity ==================
//==========================================
__global__ void getDensity(int *Typ, float *x, float *y, float *z, float *mass,
                           float *rho, float *h, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float dx, dy, dz, rr, hij, sig, q, hij3;
    float WIij;
    float ss = 0.0f;

    for (int j = 0; j < N; j++)
    {
      if (Typ[j] == 0)
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
    }
    rho[i] = ss;
  }
}

//==============================================
//================= acc_g_block ======================
//==============================================
__global__ void acc_g_block(int *Typ, float *x, float *y, float *z, float *eps, float *accx,
                            float *accy, float *accz, float *mass,
                            int *activeId, float G, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)) && (activeId[i] == 1))
  {

    float dx, dy, dz, rr, inv_r3, epsij, q, q2, q3, q4, q5, q6, fk;
    float accxt = 0.0f, accyt = 0.0f, acczt = 0.0f;
    for (int j = 0; j < N; j++)
    {
      if ((Typ[i] == 0) || (Typ[i] == 1))
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
    }
    accx[i] = accxt;
    accy[i] = accyt;
    accz[i] = acczt;
  }
}

//===================================================
//============= getPressure (Adiabatic) =============
//===================================================
__global__ void getPressure_Adiabatic(int *Typ, float *P, float *rho, float *u, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    P[i] = (gammah - 1.0f) * rho[i] * u[i];
  }
}

//===============================================
//============= getCsound (Adiabatic) ===========
//===============================================
__global__ void getCsound_Adiabatic(int *Typ, float *csnd, float *u, float gammah, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    csnd[i] = sqrt(gammah * (gammah - 1.0f) * u[i]);
  }
}

//=====================================================
//================== div_curlVel ======================
//=====================================================
__global__ void div_curlVel(int *Typ, float *divV, float *curlV, float *x, float *y, float *z,
                            float *vx, float *vy, float *vz, float *rho, float *mass,
                            float *h, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
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
      if (Typ[j] == 0)
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
    }
    divV[i] = ss; // abs(ss);
    curlV[i] = sqrt(curlVx * curlVx + curlVy * curlVy + curlVz * curlVz);
  }
}

//===========================================================
//====================== acc_sph ============================
//===========================================================
__global__ void acc_sph(int *Typ, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *h, float *c, float *rho, float *divV, float *curlV,
                        float *mass, float *P, float *ax, float *ay, float *az,
                        float visc_alpha, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float axt = 0.0f;
    float ayt = 0.0f;
    float azt = 0.0f;

    for (int j = 0; j < N; j++)
    {
      if (Typ[j] == 0)
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
    }
    ax[i] = axt;
    ay[i] = ayt;
    az[i] = azt;
  }
}

//===========================================================
//====================== acc_tot ============================
//===========================================================
__global__ void acc_g_sphB(int *Typ, float *acc_totx, float *acc_toty, float *acc_totz,
                           float *acc_gx, float *acc_gy, float *acc_gz,
                           float *acc_sphx, float *acc_sphy, float *acc_sphz,
                           float *accx_prev, float *accy_prev, float *accz_prev,
                           int *activeId, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)) && (activeId[i] == 1))
  {
    accx_prev[i] = acc_totx[i];
    accy_prev[i] = acc_toty[i];
    accz_prev[i] = acc_totz[i];

    acc_totx[i] = acc_gx[i] + acc_sphx[i];
    acc_toty[i] = acc_gy[i] + acc_sphy[i];
    acc_totz[i] = acc_gz[i] + acc_sphz[i];
  }
}

//===============================================
//=================== get_dU ====================
//===============================================
__global__ void get_dU(int *Typ, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                       float *h, float *c, float *rho, float *divV, float *curlV,
                       float *mass, float *P, float *dudt,
                       float visc_alpha, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW, vij_gWij;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float dut = 0.0f;

    for (int j = 0; j < N; j++)
    {
      if (Typ[j] == 0)
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
    }
    dudt[i] = dut;
  }
}

//==================================================
//============== update u & utprevious =============
//==================================================
__global__ void u_updater(int *Typ, float *u, float *dudt,
                          float *utprevious, float dt, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    u[i] = u[i] + 0.5f * dt * (dudt[i] + utprevious[i]);
    utprevious[i] = dudt[i];
  }
}

//===========================================================
//================= velocity evolution ======================
//===========================================================
__global__ void v_evolveB(int *Typ, float *vx, float *vy, float *vz,
                          float *accx, float *accy, float *accz,
                          float *accx_prev, float *accy_prev, float *accz_prev,
                          int *activeId, float *t_last, float t, float dt, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {
    float delta_t_j;

    vx[i] += accx[i] * dt;
    vy[i] += accy[i] * dt;
    vz[i] += accz[i] * dt;

    if (activeId[i] == 1)
    {
      delta_t_j = t + dt - t_last[i];
      vx[i] += 0.5f * (accx[i] - accx_prev[i]) * delta_t_j;
      vy[i] += 0.5f * (accy[i] - accy_prev[i]) * delta_t_j;
      vz[i] += 0.5f * (accz[i] - accz_prev[i]) * delta_t_j;
    }
  }
}

//===========================================================
//================= position evolution ======================
//===========================================================
__global__ void r_evolveB(int *Typ, float *x, float *y, float *z,
                          float *vx, float *vy, float *vz,
                          float *accx, float *accy, float *accz,
                          float *accx_prev, float *accy_prev, float *accz_prev,
                          int *activeId, float *t_last, float t, float dt, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {
    float delta_t_j;

    x[i] += (vx[i] * dt + 0.5f * accx[i] * dt * dt);
    y[i] += (vy[i] * dt + 0.5f * accy[i] * dt * dt);
    z[i] += (vz[i] * dt + 0.5f * accz[i] * dt * dt);

    if (activeId[i] == 1)
    {
      delta_t_j = t + dt - t_last[i];
      x[i] += (1.0f / 6.0f) * (accx[i] - accx_prev[i]) * delta_t_j * delta_t_j;
      y[i] += (1.0f / 6.0f) * (accy[i] - accy_prev[i]) * delta_t_j * delta_t_j;
      z[i] += (1.0f / 6.0f) * (accz[i] - accz_prev[i]) * delta_t_j * delta_t_j;
    }
  }
}

//===========================================================
//=================== dt estimation =========================
//===========================================================
__global__ void dt_array_indiv_dt(int *Typ, float *x, float *y, float *z,
                                  float *vx, float *vy, float *vz,
                                  float *accx, float *accy, float *accz,
                                  float *accx_tot, float *accy_tot, float *accz_tot,
                                  float *h, float *c, float *dt_particles,
                                  float *abs_acc_g, float *abs_acc_tot,
                                  float *divV, float *dh_dt, float C_CFL,
                                  float visc_alpha, float *eps, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    abs_acc_g[i] = sqrt(accx[i] * accx[i] + accy[i] * accy[i] + accz[i] * accz[i]);
    abs_acc_tot[i] = sqrt(accx_tot[i] * accx_tot[i] + accy_tot[i] * accy_tot[i] + accz_tot[i] * accz_tot[i]);

    float dx, dy, dz, vxij, vyij, vzij, wij, vij_rij, vij_sig, rr;

    float max_vij_sig = 0.0f;
    float tmp = 0.0f;
    for (int j = 0; j < N; j++)
    {
      if (Typ[j] == 0)
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

//===========================================================
//================== create_dt_blocks =======================
//===========================================================
__global__ void create_dt_blocks(int *Typ, float *dt_i, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {

    float dt_0 = 1e-6;
    float dt_1 = 5e-6;
    float dt_2 = 1e-5;
    float dt_3 = 5e-5;
    float dt_4 = 1e-4;

    if (dt_i[i] < dt_0)
      dt_i[i] = dt_0;

    if ((dt_i[i] >= dt_0) && (dt_i[i] < dt_1))
      dt_i[i] = dt_0;

    if ((dt_i[i] >= dt_1) && (dt_i[i] < dt_2))
      dt_i[i] = dt_1;

    if ((dt_i[i] >= dt_2) && (dt_i[i] < dt_3))
      dt_i[i] = dt_2;

    if ((dt_i[i] >= dt_3) && (dt_i[i] < dt_4))
      dt_i[i] = dt_3;

    if (dt_i[i] >= dt_4)
      dt_i[i] = dt_4;
  }
}

// t_last[nxActive] = t;
// t_next[nxActive] = t_last[nxActive] + dt_i[nxActive];
// Only update for active particles!
// update_t_last_next<<<gridSize, blockSize>>>(d_Typ, d_t_last, d_t_next, d_activeIndx, dt_particles, t, N);
//===========================================================
//==================== update_t_last_next ========================
//===========================================================
__global__ void update_t_last_next(int *Typ, float *t_last, float *t_next, int *activeId,
                                   float *dt_i, float t, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0) && (activeId[i] == 1))
  {
    t_last[i] = t;
    t_next[i] = t_last[i] + dt_i[i];
  }
}

// active particles are those with t_next - t = min(t_next - t)
// who_is_active<<<gridSize, blockSize>>>(d_Typ, d_activeIndx, dt_min_j, N);
//===========================================================
//==================== who_is_active ========================
//===========================================================
__global__ void who_is_active(int *Typ, int *activeIndx, float *t_next, float dt_min_j,
                              float t, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    float tmp = t_next[i] - t;
    // if ((tmp >= 0.99f * dt_min_j) && (tmp <= 1.01f * dt_min_j)) // instead of "if ((t_next[i] - t) == dt_min_j)"
    if ((tmp >= 0.99f * dt_min_j) && (tmp <= 1.01f * dt_min_j))
      activeIndx[i] = 1;
    else
      activeIndx[i] = 0;
  }
}

#endif
