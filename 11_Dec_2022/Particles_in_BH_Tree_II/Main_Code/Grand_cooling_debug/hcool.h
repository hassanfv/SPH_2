
#ifndef HCOOL_H
#define HCOOL_H

using namespace std;

//=====================================================================================
//======================= Heating, Cooling using CHIMES =============================== (11 Aug 2023)
//=====================================================================================
float hcoolingZ(int i, int *Typ, float *u, float *U, float *rho, float *metalz, float Zmetal, float dt, // Zmetal is the gass metallicity assumed.
                float *nHGrid, float *ZGrid, float *uEvol, float *ionFrac, float *tArr_sec, float *x, float *y, float *z,
                float *muA, float *logT, float *kpcGrid, float unit_density_in_cgs, float unit_time_in_sec, 
                float unit_u_in_cgs, float unitLength_in_cm, float kpc_in_cm, float GAMMA_MINUS1,
                int N_kpc, int N_nH, int N_Z, int N_T, int N_M, int N_Time, int N)
{
  if ((i < N) && (Typ[i] == 0))
  {

    int ndx_kpc = -1;
    int ndx_nH = -1;
    int ndx_u = -1;
    int ndx_t = -1;
    
    float delta1 = 0.0f;
    float delta2 = 0.0f;
    
    float uEv = 0.0f;
    
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
    
    cout << "ndx_kpc = " << ndx_kpc << endl;
    
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
    
    cout << "ndx_nH = " << ndx_nH << endl;
    cout << "nH[ndx_nH] = " << nHGrid[ndx_nH] << endl;
    cout << "rho[i] = " << rho[i] << endl;
    cout << "nH = " << nH << endl;
    cout << "rho_cgs = " << rho_cgs << endl;
    cout << "unit_density_in_cgs = " << unit_density_in_cgs << endl;
    cout << endl;

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

    cout << "ndx_u = " << ndx_u << endl;
    cout << "U[ndx_u-1]/unit_u_in_cgs = " << U[ndx_u-1]/unit_u_in_cgs << endl;
    cout << "U[ndx_u]/unit_u_in_cgs = " << U[ndx_u]/unit_u_in_cgs << endl;
    cout << "U[ndx_u+1]/unit_u_in_cgs = " << U[ndx_u+1]/unit_u_in_cgs << endl;
    cout << endl;

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
    
    cout << "dt = " << dt << endl;
    cout << "ndx_t = " << ndx_t << endl;
    cout << "tArr_sec[ndx_t] = " << tArr_sec[ndx_t]/unit_time_in_sec << endl;
    cout << endl;

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
      
      cout << "uEvol[ndx_minus1]/unit_u_in_cgs = " << uEvol[ndx_minus1]/unit_u_in_cgs << endl;
      cout << "uEvol[ndx]/unit_u_in_cgs = " << uEvol[ndx]/unit_u_in_cgs << endl;
      cout << endl;
      
      float diff = U[ndx_u] - U[ndx_u - 1];
      float fhi = (u_i - U[ndx_u - 1]) / diff;
      float flow = 1.0 - fhi;
      
      uEv = flow * uEv_1 + fhi * uEv_2;
      
      cout << "uEv/unit_u_in_cgs = " << uEv/unit_u_in_cgs << endl;
      
    }
    else
    {
      uEv = uEvol[ndx];
    }
    
    /*
    if (uEv > 2.0287E+18) // This corresponds to T = 1e10 K. So we set a max Temp.
    {
      uEv = 2.0287E+18;
    }
    */
    
    return uEv / unit_u_in_cgs; // uEv is the u after heating and/or cooling!
    
  }
  
  return 0.0;
}



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




#endif
