#ifndef HFVCPPLIBS_H
#define HFVCPPLIBS_H

#include <curand.h>
#include <curand_kernel.h>

// smoothing_h function updated (16 Oct 2023).
// Updating the "h" assignment for outflow particles to improve the smoothin_h calculation speed (13 Oct 2023).
// We Re-set the BH position back to (0, 0, 0). This was forgotten to be implemented !! (13 Oct 2023).
// We set max limit for uEv corresponding to 1e10K (9 Oct 2023).
// phi updated in outflow injection function (4 Oct 2023).
// Heating, Cooling using CHIMES added. Also UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s added to params.txt (11 Aug 2023).
// Added the isothermal gravitational field acceleration. (24 May 2023).
// Added the reading of the params.txt file and updated the IC reading file section and function. (22 May 2023).

//******************************************
//********** Reading params.txt ************
//******************************************

void readParams(std::string &filename, int &N_tot, int &ndx_BH, float &G, float &L_AGN_code_unit,
                float &M_dot_in_code_unit, float &vin_in_code_unit,
                float &u_for_10K_Temp, float &m_sph_high_res, float &sigma,
                float &UnitDensity_in_cgs, float &Unit_u_in_cgs, float &unitTime_in_s,
                float &unitLength_in_cm)
{
  std::ifstream file("params.txt");
  if (file.is_open())
  {
    std::getline(file, filename); // Read filename string
    file >> N_tot;                // Read N_tot
    file >> ndx_BH;               // Read ndx_BH
    file >> G;                    // Read G
    file >> L_AGN_code_unit;      // Read L_AGN_code_unit
    file >> M_dot_in_code_unit;   // Read M_dot_in_code_unit
    file >> vin_in_code_unit;     // Read vin_in_code_unit
    file >> u_for_10K_Temp;       // Read u_for_10K_Temp
    file >> m_sph_high_res;       // Read m_sph_high_res
    file >> sigma;                // Read sigma
    file >> UnitDensity_in_cgs;                // Read UnitDensity_in_cgs
    file >> Unit_u_in_cgs;                // Read Unit_u_in_cgs
    file >> unitTime_in_s;                // Read unitTime_in_s
    file >> unitLength_in_cm;                // Read unitLength_in_cm
  }
  else
  {
    std::cout << "Unable to open params.txt file";
  }
  file.close();
}

//***************************************
//********** Reading IC file ************
//***************************************

std::tuple<std::vector<int>, 
           std::vector<float>, std::vector<float>, std::vector<float>, 
           std::vector<float>, std::vector<float>, std::vector<float>, 
           std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
readVectorsFromFile(const std::string &filename) 
{
    int N;
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Could not open the IC file for reading." << std::endl;
        exit(1); // or other error handling
    }

    // Read N_tot from the start of the file
    file.read(reinterpret_cast<char*>(&N), sizeof(int));

    // Create and resize vectors based on N
    std::vector<int> Typvec(N);
    std::vector<float> xvec(N);
    std::vector<float> yvec(N);
    std::vector<float> zvec(N);
    std::vector<float> vxvec(N);
    std::vector<float> vyvec(N);
    std::vector<float> vzvec(N);
    std::vector<float> uvec(N);
    std::vector<float> hvec(N);
    std::vector<float> epsvec(N);
    std::vector<float> massvec(N);

    file.read(reinterpret_cast<char*>(Typvec.data()), sizeof(int) * N);
    file.read(reinterpret_cast<char*>(xvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(yvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(zvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(vxvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(vyvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(vzvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(uvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(hvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(epsvec.data()), sizeof(float) * N);
    file.read(reinterpret_cast<char*>(massvec.data()), sizeof(float) * N);

    file.close();

    return std::make_tuple(Typvec, xvec, yvec, zvec, vxvec, vyvec, vzvec, uvec, hvec, epsvec, massvec);
}



//*************************************************************************
//*************** Function to save the OUTPUT Snap-Shots!! ****************
//*************************************************************************
void saveArraysToBinary(const std::string &filename, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *rho, float *h, float *u, float *mass, float *ionFrac, int *Typ, int N, int N_ionFrac)
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
  file.write(reinterpret_cast<const char *>(&N_ionFrac), sizeof(int));

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
  
  file.write(reinterpret_cast<const char *>(ionFrac), N_ionFrac * sizeof(float));

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


//========================================
//================ ngbDB =================
//========================================
__global__ void ngbDB(int *Typ, float *x, float *y, float *z, float *h, int *ngb, int MAX_ngb, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float h_new = 2.05f * h[i];

    int k = 0;
    int j = 0;

    float dx, dy, dz;
    float rr;

    while (j < N && k < MAX_ngb)
    {
      if (Typ[j] == 0)
      {
        dx = x[j] - x[i];
        dy = y[j] - y[i];
        dz = z[j] - z[i];
        rr = sqrt(dx * dx + dy * dy + dz * dz);

        if (rr <= h_new)
        {
          ngb[i * MAX_ngb + k] = j;
          k++;
        }
      }
      j++;
    }
  }
}


//========================================
//========== Smoothing Length_ngb ============ Updated: 28 Jan 2023. h_new adopted from eq.31 in Gadget2 Paper
//========================================
__global__ void smoothing_h_ngb(int *Typ, float *x, float *y, float *z, float *h,
                            int N, int Ndown, int Nup, float coeff,
                            float Nngb_f, float *Nngb_previous, float *divV, int *ngb, int MAX_ngb, float dt)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float h_new = 2.0f * (0.5f * h[i] * (1.0f + pow((Nngb_f / Nngb_previous[i]), 1.0f / 3.0f)) +
                          1.0f / 3.0f * h[i] * divV[i] * dt);
    int N_iter = 0;
    int k = 0;
    int j = 0;
    
    int k_pre = Nup; // Just a choice so that it does not activate that if condition in the first run!

    float dx, dy, dz;
    while ((k < Ndown) || (k > Nup))
    {

      k = 0;

      for (int jj = 0; jj < MAX_ngb; jj++)
      {

        j = ngb[i * MAX_ngb + jj];
        
        if (j != -1) // Note that the Typ[j] == 0 condition was already checked during ngb array construction!
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

        //-----------
        if (k > Nup) // To stop unnecessary search after it reaches k > Nup!
        {
          break;
        }
        //-----------
      }

      //-----------
      if (((k < Ndown) && (k_pre > Nup)) || ((k > Nup) && (k_pre < Ndown))) // To prevent oscillation outside Nup and Ndown values!!
      {
        coeff = coeff / 2.0f;
      }
      //-----------

      if (k < Ndown)
      {
        h_new = h_new + coeff * 2.0f * h[i];
      }

      if (k > Nup)
      {
        h_new = h_new - coeff * 2.0f * h[i];
      }

      k_pre = k;

      N_iter++;
      if (N_iter > 500)
      {
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
//============ getDensity_ngb ==================
//==========================================
__global__ void getDensity_ngb(int *Typ, float *x, float *y, float *z, float *mass,
                           float *rho, float *h, int *ngb, int MAX_ngb, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float dx, dy, dz, rr, hij, sig, q, hij3;
    float WIij;
    float ss = 0.0f;
    
    int j = 0;

    for (int jj = 0; jj < MAX_ngb; jj++)
    {
    
      j = ngb[i * MAX_ngb + jj];
    
      if (j != -1) // Note that the Typ[j] == 0 condition was already checked during ngb array construction!
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
//================= acc_g ======================
//==============================================
__global__ void acc_g(int *Typ, float *x, float *y, float *z, float *eps, float *accx,
                      float *accy, float *accz, float *mass, float G, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {

    float dx, dy, dz, rr, inv_r3, epsij, q, q2, q3, q4, q5, q6, fk;
    float accxt = 0.0f, accyt = 0.0f, acczt = 0.0f;
    for (int j = 0; j < N; j++)
    {
      if ((Typ[j] == 0) || (Typ[j] == 1))
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
//================== div_curlVel_ngb ======================
//=====================================================
__global__ void div_curlVel_ngb(int *Typ, float *divV, float *curlV, float *x, float *y, float *z,
                            float *vx, float *vy, float *vz, float *rho, float *mass,
                            int *ngb, int MAX_ngb, float *h, int N)
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
    
    int j = 0;

    for (int jj = 0; jj < MAX_ngb; jj++)
    {
    
      j = ngb[i * MAX_ngb + jj];
    
      if (j != -1) // Note that the Typ[j] == 0 condition was already checked during ngb array construction!
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
//====================== acc_sph_ngb ============================
//===========================================================
__global__ void acc_sph_ngb(int *Typ, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                        float *h, float *c, float *rho, float *divV, float *curlV,
                        float *mass, float *P, float *ax, float *ay, float *az,
                        int *ngb, int MAX_ngb, float visc_alpha, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float axt = 0.0f;
    float ayt = 0.0f;
    float azt = 0.0f;
    
    int j = 0;

    for (int jj = 0; jj < MAX_ngb; jj++)
    {
    
      j = ngb[i * MAX_ngb + jj];
    
      if (j != -1) // Note that the Typ[j] == 0 condition was already checked during ngb array construction!
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
__global__ void acc_g_sph(int *Typ, float *acc_totx, float *acc_toty, float *acc_totz,
                          float *acc_gx, float *acc_gy, float *acc_gz,
                          float *acc_sphx, float *acc_sphy, float *acc_sphz,
                          int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {
    acc_totx[i] = acc_gx[i] + acc_sphx[i];
    acc_toty[i] = acc_gy[i] + acc_sphy[i];
    acc_totz[i] = acc_gz[i] + acc_sphz[i];
  }
}

//===============================================
//=================== get_dU_ngb ====================
//===============================================
__global__ void get_dU_ngb(int *Typ, float *x, float *y, float *z, float *vx, float *vy, float *vz,
                       float *h, float *c, float *rho, float *divV, float *curlV,
                       float *mass, float *P, float *dudt,
                       int *ngb, int MAX_ngb, float visc_alpha, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW, vij_gWij;
    float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
    float dut = 0.0f;
    
    int j = 0;

    for (int jj = 0; jj < MAX_ngb; jj++)
    {
    
      j = ngb[i * MAX_ngb + jj];
    
      if (j != -1) // Note that the Typ[j] == 0 condition was already checked during ngb array construction!
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
__global__ void v_evolve(int *Typ, float *vx, float *vy, float *vz,
                         float *accx, float *accy, float *accz,
                         float dt, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {
    vx[i] += accx[i] * dt / 2.0f;
    vy[i] += accy[i] * dt / 2.0f;
    vz[i] += accz[i] * dt / 2.0f;
  }
}

//===========================================================
//================= position evolution ======================
//===========================================================
__global__ void r_evolve(int *Typ, float *x, float *y, float *z,
                         float *vx, float *vy, float *vz,
                         float dt, int ndx_BH, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;
    
    if (i == ndx_BH)
    {
      // Re-setting the BH position back to (0, 0, 0)!
      x[ndx_BH] = 0.0f;
      y[ndx_BH] = 0.0f;
      z[ndx_BH] = 0.0f;
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



//==============================================================
//====================== hash function =========================
//==============================================================
__device__ unsigned long long hash(unsigned long long x, unsigned long long y) {
    return x * 6364136223846793005ULL + y;
}



//==============================================================
//========= Outflow injection 2 (Richings et al - 2018)===========
//==============================================================
__global__ void outflow_injector2(int *Typ, float *x, float *y, float *z,
                                 float *vx, float *vy, float *vz,
                                 float *h, float *eps, float *mass,
                                 float Nngb_f, float *Nngb_previous,
                                 float *u, float M_dot_in, float v_in,
                                 float mass_sph_high_res, float u_for_10K_Temp,
                                 float *leftover_mass, float dt, int ndx_BH, int N,
                                 unsigned long long seed)
{

  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx == 0)
  {
    // Calculate the mass to inject in each time step
    float mass_to_inject = M_dot_in * dt;
    mass_to_inject += *leftover_mass;

    // Calculate number of particle pairs to inject
    int num_pairs = static_cast<int>(mass_to_inject / (2.0f * mass_sph_high_res));

    float one_pc_in_code_unit = 0.001f; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    float h_outflow = 0.001; // ONLY place holder!

    // Finding the last index for which Typ is not equal to -1.
    // Note that with the way I do in this for loop, jj will be the number of times Typ is not
    // equal to -1. So to convert it to index, we need to subtract 1 from it !!!
    int jj = 0;
    for (int i = 0; i < N; i++)
    {
      if ((Typ[i] == 0) || (Typ[i] == 1))
      {
        jj += 1;
      }
    }
    jj -= 1; // See the above explanation!!
    
    //--- Estimation of h_outflow --
    if (jj > ndx_BH)
    {
      h_outflow = h[jj]; // We take the h of the last injected outflow particle!
    }
    else
    {
      h_outflow = h[ndx_BH - 1]; // This happens when we had not yet injected any outflow particle!
    }


    // Spawn particle pairs
    for (int j = 0; j < num_pairs; j++)
    {

      // Initialize cuRAND generator with the provided seed
      curandState state;
      curand_init(hash(seed, j), idx, 0, &state);

      // Calculate random distance from 0 to 1 pc (in cm)
      float rt = curand_uniform(&state) * one_pc_in_code_unit;
      // Calculate random orientation (assuming spherical coordinates)
      float theta = curand_uniform(&state) * 2.0f * M_PI;
      //float phi = curand_uniform(&state) * M_PI; // This was non symmetric!!
      float phi = acos(2.0f * curand_uniform(&state) - 1.0f);
      // Calculate Cartesian coordinates of the pair
      float xt = rt * sin(phi) * cos(theta);
      float yt = rt * sin(phi) * sin(theta);
      float zt = rt * cos(phi);

      float rr = sqrt(xt * xt + yt * yt + zt * zt);

      float vxt = xt / rr * v_in;
      float vyt = yt / rr * v_in;
      float vzt = zt / rr * v_in;

      //---- Injecting the first particle of the pair ----
      Typ[jj + 1] = 0;

      x[jj + 1] = xt;
      y[jj + 1] = yt;
      z[jj + 1] = zt;

      vx[jj + 1] = vxt;
      vy[jj + 1] = vyt;
      vz[jj + 1] = vzt;

      mass[jj + 1] = mass_sph_high_res;
      h[jj + 1] = h_outflow;
      eps[jj + 1] = h_outflow;
      u[jj + 1] = u_for_10K_Temp;

      Nngb_previous[jj + 1] = Nngb_f;

      //---- Injecting the second particle of the pair ----
      Typ[jj + 2] = 0;

      x[jj + 2] = -xt;
      y[jj + 2] = -yt;
      z[jj + 2] = -zt;

      vx[jj + 2] = -vxt;
      vy[jj + 2] = -vyt;
      vz[jj + 2] = -vzt;

      mass[jj + 2] = mass_sph_high_res;
      h[jj + 2] = h_outflow;
      eps[jj + 2] = h_outflow;
      u[jj + 2] = u_for_10K_Temp;

      Nngb_previous[jj + 2] = Nngb_f;

      jj += 2;

      mass_to_inject -= 2.0f * mass_sph_high_res;
    }
    *leftover_mass = mass_to_inject;
  }
}


//==================================================================================
//========= Isothermal gravitational acceleration (Richings et al - 2018)===========
//==================================================================================
__global__ void galaxy_isothermal_potential(int *Typ, float *x, float *y, float *z, float *accx,
                                            float *accy, float *accz, float sigma, float G, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && ((Typ[i] == 0) || (Typ[i] == 1)))
  {

    float rr = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);

    // Compute the enclosed mass at radius rr (eq.1 in Richings et al - 2018).
    float enclosed_mass = 2.0 * sigma * sigma * rr / G;

    // Compute the gravitational acceleration
    accx[i] += -G * enclosed_mass * x[i] / rr / rr / rr; // ===> -G * M * x / r^3
    accy[i] += -G * enclosed_mass * y[i] / rr / rr / rr;
    accz[i] += -G * enclosed_mass * z[i] / rr / rr / rr;
  }
}


//=====================================================================================
//======================= Heating, Cooling using CHIMES =============================== (11 Aug 2023)
//=====================================================================================
__global__ void hcoolingx(int *Typ, float *u, float *U, float *rho, float *metalz, float Zmetal, float dt, // Zmetal is the gass metallicity assumed.
                         float *nHGrid, float *ZGrid, float *uEvol, float *ionFrac, float *tArr_sec, float *x, float *y, float *z,
                         float *muA, float *logT, float *kpcGrid, float unit_density_in_cgs, float unit_time_in_sec, 
                         float unit_u_in_cgs, float unitLength_in_cm, float kpc_in_cm, float GAMMA_MINUS1,
                         int N_kpc, int N_nH, int N_Z, int N_T, int N_M, int N_Time, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {

    int ndx_kpc = -1;
    int ndx_nH = -1;
    int ndx_u = -1;
    int ndx_t = -1;
    
    float delta1 = 0.0f;
    float delta2 = 0.0f;
    
    float uEv = 0.0f;
    
    //float kB = 1.3807e-16;
    
    //======== kpc =========
    float r_i = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    r_i = r_i * unitLength_in_cm / kpc_in_cm; // converting r_i from code unit to kpc!
    
    for (int j = 0; j < N_kpc; j++)
    {
      if (ndx_kpc == -1 && r_i <= kpcGrid[j])
      {
        ndx_kpc = j;
      }
    }
    
    if (ndx_kpc > 0) 
    {
      delta1 = abs(kpcGrid[ndx_kpc] - r_i);
      delta2 = abs(kpcGrid[ndx_kpc - 1] - r_i);
      
      if (delta2 < delta1)
      {
        ndx_kpc -= 1;
      }
    }
    
    if (ndx_kpc == -1) // This happens when r_i is greater than max(kpcGrid) !
    {
      ndx_kpc = N_kpc - 1;
    }
    
    
    //======== nH ==========
    //---- convert rho to nH
    float mH = 1.6726e-24;
    float XH = 0.70;
    float rho_cgs = rho[i] * unit_density_in_cgs;
    float nH = rho_cgs * XH / mH;
    
    for (int j = 0; j < N_nH; j++)
    {
      if (ndx_nH == -1 && nH <= nHGrid[j])
      {
        ndx_nH = j;
      }
    }

    if (ndx_nH > 0) 
    {
      delta1 = abs(nHGrid[ndx_nH] - nH);
      delta2 = abs(nHGrid[ndx_nH - 1] - nH);
      
      if (delta2 < delta1)
      {
        ndx_nH -= 1;
      }
    }
    
    if (ndx_nH == -1) // This happens when nH is greater than max(nHGrid) !
    {
      ndx_nH = N_nH - 1;
    }

    //======== Z =========
    int ndx_Z = 1; // Assuming [Z/H] -1. Recall that Z = [-2, -1, 0].

    //========= u =========
    // Calculate the strides
    int stride_kpc = N_T * N_nH * N_Z * N_Time;
    int stride_T = N_nH * N_Z * N_Time;
    int stride_nH = N_Z * N_Time;
    int stride_Z = N_Time;
    
    // Copy data ---> uEvol[:, N_nH, N_Z, 0]. We loop over N_T!
    float u_i = u[i] * unit_u_in_cgs;
    
    for (int t = 0; t < N_T; t++)
    {
      int index = (ndx_kpc * stride_kpc) + (t * stride_T) + (ndx_nH * stride_nH) + (ndx_Z * stride_Z) + 0;
      U[t] = uEvol[index];
    }
    
    for (int j = 0; j < N_T; j++)
    {
      if (ndx_u == -1 && u_i <= U[j])
      {
        ndx_u = j;
      }
    }
    
    if (ndx_u == -1) // This happens when u_1 is greater than max(U) !
    {
      ndx_u = N_T - 1;
    }

    //========= time =========
    float dt_sec = dt * unit_time_in_sec;
    
    for (int j = 0; j < N_Time; j++)
    {
      if (ndx_t == -1 && dt_sec <= tArr_sec[j])
      {
        ndx_t = j;
      }
    }
    
    if (ndx_t == -1) // This happens when dt is greater than max(tArr_sec) !
    {
      ndx_t = N_Time - 1;
    }
    

    int ndx = (ndx_kpc * stride_kpc) + (ndx_u * stride_T) + (ndx_nH * stride_nH) + (ndx_Z * stride_Z) + ndx_t;
    
    //---------------------
    // Calculate the strides for Metalz
    int stride_kpcx = N_T * N_nH * N_Z * N_M * N_Time;
    int stride_Tx = N_nH * N_Z * N_M * N_Time;
    int stride_nHx = N_Z * N_M * N_Time;
    int stride_Zx = N_M * N_Time;
    int stride_time = N_Time;
    
    if (ndx_u > 0)
    {
      int ndx_minus1 = (ndx_kpc * stride_kpc) + ((ndx_u-1) * stride_T) + (ndx_nH * stride_nH) + (ndx_Z * stride_Z) + ndx_t;
      float uEv_1 = uEvol[ndx_minus1];
      float uEv_2 = uEvol[ndx];
      
      float diff = U[ndx_u] - U[ndx_u - 1];
      float fhi = (u_i - U[ndx_u - 1]) / diff;
      float flow = 1.0 - fhi;
      
      uEv = flow * uEv_1 + fhi * uEv_2;
      
    }
    else
    {
      uEv = uEvol[ndx];
    }
    
    if (uEv > 2.0287E+18) // This corresponds to T = 1e10 K. So we set a max Temp.
    {
      uEv = 2.0287E+18;
    }
    
    u[i] = uEv / unit_u_in_cgs; // uEv is the u after heating and/or cooling!
    
    for (int k = 0; k < N_M; k++)
    {
      int ndx_M = (ndx_kpc * stride_kpcx) + (ndx_u * stride_Tx) + (ndx_nH * stride_nHx) + (ndx_Z * stride_Zx) + (k * stride_time) + ndx_t;
      
      ionFrac[i * N_M + k] = metalz[ndx_M]; // k is species index! Here we use flattened version of a 2D array of (N, N_M) dimension
    }
    
  }
}


//=====================================================================================
//========================== For debugging Cooling ====================================
//=====================================================================================
__global__ void u_before_adiabatic(int *Typ, float *u, float *uB, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    uB[i] = u[i];
  }

}


//=====================================================================================
//========================== For debugging Cooling ====================================
//=====================================================================================
__global__ void u_After_cooling(int *Typ, float *u, float *uB, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    uB[i] = u[i];
  }

}



//=====================================================================================
//========================== For debugging Cooling ====================================
//=====================================================================================
__global__ void dudt_previous(int *Typ, float *u, float *uB, int N)
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if ((i < N) && (Typ[i] == 0))
  {
    uB[i] = u[i];
  }

}


#endif
