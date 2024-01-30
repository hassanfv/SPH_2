%%writefile test.cu

#include <iostream>
#include <cmath>
#include <fstream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <chrono>


using namespace std;


__global__ void getSmoothing(float *x, float *y, float *z, float *h, float *Nngb_previous, float coeff, int Nup, int Ndown, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N)
  {

    float h_new = 2.0f * h[i];
    int N_iter = 0;
    int k = 0;
    
    int k_pre = Nup; // Just a choice so that it does not activate in the first run!

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


//----- random_uniform
float random_uniform(float a, float b)
{
  return a + (b - a) * (rand() / static_cast<float>(RAND_MAX));
}


int main()
{

  auto T_beg = std::chrono::high_resolution_clock::now();

  // Set the seed for the random number generator
  srand(42);

  // Constants
  const double Msun = 1.989e33;  // Solar mass in grams
  const double grav_const_in_cgs = 6.67430e-8;  // Gravitational constant in cm^3 g^-1 s^-2
  //const double G = 6.67430e-8;
  const double kB = 1.380649e-16;  // Boltzmann constant in erg K^-1
  const double mH = 1.6726219e-24;  // Proton mass in grams
  const double clight = 29979245800.0;  // cm/s
  const double cm_to_kpc = 3.086e21;
  const double mu = 0.61f;
  const double sigma = 200.0f * 1000.0f * 100.0f; // cm/s =====> 200 km/s - See eq.1 in Richings et al - 2018

  const float L_box = 1.04f; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  const float mSPH_high = 10.0f; // Msun !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const int N_high = 5000000; // This is the number of particles for only the high res octant!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  const float mSPH_low = 80.0f; // Msun!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const int N_low = 625000; // This is the number of particles for each low-res octant. In total, it will be 7 * N_low for all 7 low-res octants!!!!!!!!!!!!!!!!!!

  const float stp_high = L_box / 2.0f / trunc(pow(N_high, 1.0f/3.0f));
  const float stp_low = L_box / 2.0f / trunc(pow(N_low, 1.0f/3.0f));

  const float L_box_in_kpc = L_box;

  //const int N_blank = 20000;

  int N = 7 * N_low + N_high + 1000; // added 1000 for the BH! Just to be save!! Note that N will be updated later!
  float *x = new float[N];
  float *y = new float[N];
  float *z = new float[N];
  float *m_in_Msun = new float[N];
  int currentIndex = 0;
  
  
  //---------------- For Hi-res --------------------------
  int Nbins = 0;
  for (float xx = 0; xx < L_box; xx += stp_high)
  {
    Nbins++;
  }
  
  float *valz = new float[Nbins];
  
  int i = 0;
  for (float xx = 0; xx < L_box; xx += stp_high)
  {
    valz[i] = xx;
    i++;
  }
  //-----------------------------------------------------
  
  cout << "last val = " << valz[Nbins-1] << endl;
  
  cout << "Nbins = " << Nbins << endl;
  cout << "(Nbins/2.) = " << (Nbins/2) << endl;
  cout << "trunc(Nbins/2) = " << trunc(Nbins/2) << endl;
  

  for (int i = 0; i < Nbins/2; i++)
  {
    for (int j = 0; j < Nbins/2; j++)
    {
      for (int k = 0; k < Nbins/2; k++)
      {
        float xx = random_uniform(valz[i], valz[i+1]) - L_box_in_kpc / 2.0f;
        float yy = random_uniform(valz[j], valz[j+1]) - L_box_in_kpc / 2.0f;
        float zz = random_uniform(valz[k], valz[k+1]) - L_box_in_kpc / 2.0f;

        if (sqrt(xx*xx + yy*yy + zz*zz) <= L_box/2.0f) {
        x[currentIndex] = xx;
        y[currentIndex] = yy;
        z[currentIndex] = zz;
        m_in_Msun[currentIndex] = mSPH_high;
        currentIndex++;
        }
      }
    }
  }
  
  
  //---------------- For Low-res -------------------------
  Nbins = 0;
  for (float xx = 0; xx < L_box; xx += stp_low)
  {
    Nbins++;
  }
  
  float *valzL = new float[Nbins];
  
  i = 0;
  for (float xx = 0; xx < L_box; xx += stp_low)
  {
    valzL[i] = xx;
    i++;
  }
  //-----------------------------------------------------
  
  int ends[] = {0, Nbins/2};
  
  // Fill the rest of the box with low-res particles
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        if (ends[i] == 0 && ends[j] == 0 && ends[k] == 0) continue; // Skip the high-res octant

        for (int ii = ends[i]; ii < ends[i] + Nbins/2; ii++)
        {
          for (int jj = ends[j]; jj < ends[j] + Nbins/2; jj++)
          {
            for (int kk = ends[k]; kk < ends[k] + Nbins/2; kk++)
            {
            
              if ((ii+1 < Nbins-1) && (jj+1 < Nbins-1) && (kk+1 < Nbins-1))
              {
                float xx = random_uniform(valzL[ii], valzL[ii+1]) - L_box_in_kpc / 2.0f;
                float yy = random_uniform(valzL[jj], valzL[jj+1]) - L_box_in_kpc / 2.0f;
                float zz = random_uniform(valzL[kk], valzL[kk+1]) - L_box_in_kpc / 2.0f;

              if (sqrt(xx*xx + yy*yy + zz*zz) <= L_box/2.0f) {
                  if (currentIndex >= N) {
                      cout << "currentIndex = " << currentIndex << endl;
                      cerr << "Error: Exceeded allocated memory." << endl;
                      exit(1);
                  }
                  x[currentIndex] = xx;
                  y[currentIndex] = yy;
                  z[currentIndex] = zz;
                  m_in_Msun[currentIndex] = mSPH_low;
                  currentIndex++;
              }
              }
            }
          }
        }
      }
    }
  }

  // IMPORTANT: N should change to currentIndex (since it represents the number of particles, there is no need to subtract 1 from it!!!)

  cout << "N = " << N << endl;
  cout << "currentIndex = " << currentIndex << endl;
  
  N = currentIndex;
  
  cout << "updated N = " << N << endl;
  
  cout << "First mass = " << m_in_Msun[0] << endl;
  cout << "Last mass = " << m_in_Msun[N-1] << endl;

  float M_tot_in_Msun = 0.0f;
  for (int i = 0; i < N; i++)
  {
    M_tot_in_Msun += m_in_Msun[i];
  }
  
  cout << "M_tot_in_Msun = " << M_tot_in_Msun << endl;

  //===============================================================
  //============ Setting the units of the simulation ==============
  //===============================================================
  double unitMass_in_g_tmp = M_tot_in_Msun * Msun;
  double unitLength_in_cm_tmp = cm_to_kpc;  // This is 1 kpc in cm so our unit length is 1 kpc!
  float Unit_u_in_cgs = grav_const_in_cgs * unitMass_in_g_tmp / unitLength_in_cm_tmp;
  
  float UnitDensity_in_cgs = unitMass_in_g_tmp / unitLength_in_cm_tmp;
  UnitDensity_in_cgs = UnitDensity_in_cgs / unitLength_in_cm_tmp / unitLength_in_cm_tmp;
  
  float unitTime_in_s = unitLength_in_cm_tmp / unitMass_in_g_tmp;
  unitTime_in_s = sqrt(unitTime_in_s * unitLength_in_cm_tmp * unitLength_in_cm_tmp / grav_const_in_cgs);
  
  float unitVelocity_in_cm_per_s = unitLength_in_cm_tmp / unitTime_in_s;

  float unitLength_in_cm = unitLength_in_cm_tmp;

  cout << "unitLength_in_cm = " << unitLength_in_cm << endl;
  cout << "unitTime_in_s = " << unitTime_in_s << endl;
  cout << "unitVelocity_in_cm_per_s = " << unitVelocity_in_cm_per_s << endl;
  cout << "Unit_u_in_cgs = " << Unit_u_in_cgs << endl;
  cout << "UnitDensity_in_cgs = " << UnitDensity_in_cgs << endl;


  //---- setting up velocity arrays ------
  float *vx = new float[N];
  float *vy = new float[N];
  float *vz = new float[N];
  
  for (int i = 0; i < N; i++)
  {
    vx[i] = 0.0f;
    vy[i] = 0.0f;
    vz[i] = 0.0f;
  }

  //---- setting up the mass array -------
  float *mass = new float[N];
  
  for (int i = 0; i < N; i++)
  {
    mass[i] = m_in_Msun[i] / M_tot_in_Msun;
  }
  
  cout << "First mass in code unit = " << mass[0] << endl;
  cout << "Last mass in code unit = " << mass[N-1] << endl;


  //---- setting up u --------------------
  float T_gas = 10000.0f; // K
  float u_tmp = (3.0f/2.0f) * kB * T_gas / mu / mH / Unit_u_in_cgs;
  
  cout << "u_tmp in code unit = " << u_tmp << endl;

  float *u = new float[N];
  
  for (int i = 0; i < N; i++)
  {
    u[i] = u_tmp;
  }
  
  cout << "First u = " << u[0] << endl;
  cout << "Last u = " << u[N-1] << endl;
  
  
  //===== Used for Outflow injection =====
  float G = 1.0f;

  float Tou_in = 1.0f;
  double L_AGN = 1e46;  // erg/s //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Note that we multiply u by mass because u is in per unit mass!!
  float L_AGN_code_unit = L_AGN / (Unit_u_in_cgs * unitMass_in_g_tmp / unitTime_in_s);
  // So L_AGN is now in energy per unit mass per unit time.

  float clight_code_unit = clight / unitVelocity_in_cm_per_s;

  float vin = 30000.0f * 1000.0f * 100.0f; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  float vin_in_code_unit = vin / unitVelocity_in_cm_per_s;

  float M_dot_in_code_unit = Tou_in * L_AGN_code_unit / clight_code_unit / vin_in_code_unit;

  float u_for_10K_Temp = (3.0f/2.0f) * kB * T_gas / mu / mH / Unit_u_in_cgs;
  //=======================================

  cout << endl;
  cout << "L_AGN_code_unit = " << L_AGN_code_unit << endl;
  cout << "vin_in_code_unit = " << vin_in_code_unit << endl;
  cout << "M_dot_in_code_unit = " << M_dot_in_code_unit << endl;
  cout << "u_for_10K_Temp = " << u_for_10K_Temp << endl;

  //====== estimating the multiplier =======

  float dt = 1e-7; //----> used to estimate the multiplier!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  float multiplier = 1.0f;
  float m_sph_outflow = mSPH_high / M_tot_in_Msun * multiplier;  //!!!!!!! divided by M_tot_in_Msun to convert to code unit !!!!!!!!!
  
  cout << endl;
  cout << "M_dot_in_code_unit * dt / m_sph_outflow = " << M_dot_in_code_unit * dt / m_sph_outflow << endl;
  cout << endl;
  

  //======= determining the smoothing length h ======
  float *h, *d_h, *d_x, *d_y, *d_z, *Nngb, *d_Nngb;

  h = new float[N];
  cudaMalloc(&d_h, N * sizeof(float));
  
  Nngb = new float[N];
  cudaMalloc(&d_Nngb, N * sizeof(float));
  
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_z, N * sizeof(float));
  
  for (int i = 0; i < N; i++)
  {
    h[i] = 0.5f * (stp_high + stp_low); // initial estimate!
    Nngb[i] = 0.0;
  }
  
  cudaMemcpy(d_h, h, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Nngb, Nngb, N * sizeof(float), cudaMemcpyHostToDevice);
  
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, z, N * sizeof(float), cudaMemcpyHostToDevice);
  
  int Nngb_ref = 64;
  int Nup = Nngb_ref + 5;
  int Ndown = Nngb_ref - 5;
  
  float coeff = 0.02;

  int blockSize = 256;                            // number of threads in a block
  int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks in a grid

  getSmoothing<<<gridSize, blockSize>>>(d_x, d_y, d_z, d_h, d_Nngb, coeff, Nup, Ndown, N);
  cudaDeviceSynchronize();
  
  cudaMemcpy(h, d_h, N * sizeof(float), cudaMemcpyDeviceToHost);


  //===== defining grav. epsilon ======
  float *epsilon = new float[N];
  
  for (int i = 0; i < N; i++)
  {
    epsilon[i] = h[i];
  }
  
  //===== defining Typ ========
  int *Typ = new int[N];
  
  for (int i = 0; i < N; i++)
  {
    Typ[i] = 0; // gas particle!
  }

  //===== Seting up BH particle at the center ====> index N is set at BH. Note that the last gas particle is at N-1 !
  Typ[N] = 1;  // Since BH is a collisionless particle, its Type is set to 1 !

  int ndx_BH = N;

  x[N] = 0.0;
  y[N] = 0.0;
  z[N] = 0.0;

  vx[N] = 0.0;
  vy[N] = 0.0;
  vz[N] = 0.0;

  double M_BH_in_g_tmp = 1e8 * Msun;
  float M_BH_in_g = M_BH_in_g_tmp / unitMass_in_g_tmp;
  mass[N] = M_BH_in_g;
  u[N] = 0.0; // non-gas particle!
  h[N] = 0.0; // non-gas particle!
  // equivalent to 1 pc in code unit. Recall that unitLength is 1 kpc!
  epsilon[N] = 0.001;

  //***********************************************
  //************* adding N_blank ******************
  //***********************************************
  int N_blank = 100000; //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  int N_tot = N + N_blank;
  
  int *Typnew = new int[N_tot];

  float *xnew = new float[N_tot];
  float *ynew = new float[N_tot];
  float *znew = new float[N_tot];
  
  float *vxnew = new float[N_tot];
  float *vynew = new float[N_tot];
  float *vznew = new float[N_tot];
  
  float *unew = new float[N_tot];
  float *hnew = new float[N_tot];
  float *epsnew = new float[N_tot];
  float *massnew = new float[N_tot];
  
  for (int i = 0; i <= N; i++) // Note that i<=N! We inlcude N itself as it is the BH position!!
  {
    Typnew[i] = Typ[i];
    
    xnew[i] = x[i];
    ynew[i] = y[i];
    znew[i] = z[i];
    
    vxnew[i] = vx[i];
    vynew[i] = vy[i];
    vznew[i] = vz[i];
    
    unew[i] = u[i];
    hnew[i] = h[i];
    epsnew[i] = epsilon[i];
    massnew[i] = mass[i];
  }
  
  // Setting values for the N_blank all to zero!!
  for (int i = N+1; i < N_tot; i++) 
  {
    Typnew[i] = -1; // Typ = -1 indicates this place is free and not yet taken by a particle!!
    
    xnew[i] = 0.0f;
    ynew[i] = 0.0f;
    znew[i] = 0.0f;
    
    vxnew[i] = 0.0f;
    vynew[i] = 0.0f;
    vznew[i] = 0.0f;
    
    unew[i] = 0.0f;
    hnew[i] = 0.0f;
    epsnew[i] = 0.0f;
    massnew[i] = 0.0f;
  }

  //====== Output to a binary file ===========
  int numValue = static_cast<int>(std::floor(N_tot / 1000.0));
  std::string filename = "IC_R_" + std::to_string(numValue) + "k.bin";
  
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  if(!out)
  {
    std::cerr << "Cannot open the file." << std::endl;
    return 1;  // or any other error handling
  }
  
  // Save N_tot at the start of the file
  out.write((char*)&N_tot, sizeof(int));

  out.write((char*)Typnew, N_tot * sizeof(int));

  out.write((char*)xnew, N_tot * sizeof(float));
  out.write((char*)ynew, N_tot * sizeof(float));
  out.write((char*)znew, N_tot * sizeof(float));

  out.write((char*)vxnew, N_tot * sizeof(float));
  out.write((char*)vynew, N_tot * sizeof(float));
  out.write((char*)vznew, N_tot * sizeof(float));

  out.write((char*)unew, N_tot * sizeof(float));
  out.write((char*)hnew, N_tot * sizeof(float));
  out.write((char*)epsnew, N_tot * sizeof(float));
  out.write((char*)massnew, N_tot * sizeof(float));

  out.close();
  
  
  //===== Saving the parameters and constants! ========
  float sigma_in_code_unit = sigma / unitVelocity_in_cm_per_s;
  
  std::ofstream outfile("params.txt");
  if (outfile.is_open()) 
  {
    outfile << filename << "\n";
    outfile << N_tot << "\n";
    outfile << ndx_BH << "\n";
    outfile << G << "\n";
    outfile << L_AGN_code_unit << "\n"; // Note will be multiplied by dt in the code.
    outfile << M_dot_in_code_unit << "\n"; // Note will be multiplied by dt in the code.
    outfile << vin_in_code_unit << "\n";
    outfile << u_for_10K_Temp << "\n";
    outfile << m_sph_outflow << "\n";
    outfile << sigma_in_code_unit << "\n";
    outfile << UnitDensity_in_cgs << "\n";
    outfile << Unit_u_in_cgs << "\n";
    outfile << unitTime_in_s << "\n";
    outfile << unitLength_in_cm << "\n";

    outfile.close();
  } else
  {
    std::cerr << "Unable to open file for writing!";
  }
  

  delete[] x;
  delete[] y;
  delete[] z;
  delete[] vx;
  delete[] vy;
  delete[] vz;
  delete[] u;
  delete[] h;
  delete[] epsilon;
  delete[] mass;
  
  delete[] Typnew;
  delete[] xnew;
  delete[] ynew;
  delete[] znew;
  delete[] vxnew;
  delete[] vynew;
  delete[] vznew;
  delete[] unew;
  delete[] hnew;
  delete[] epsnew;
  delete[] massnew;

  auto T_end = std::chrono::high_resolution_clock::now();
  auto elapsed_T = std::chrono::duration_cast<std::chrono::nanoseconds>(T_end - T_beg);
  cout << "Elapsed T = " << elapsed_T.count() * 1e-9 << endl;

}


