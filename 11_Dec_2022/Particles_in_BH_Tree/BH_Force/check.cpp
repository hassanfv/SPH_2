#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <tuple>


using namespace std;


struct Vector
{
float x, y, z;
};


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



//==============================================
//================= acc_g ======================
//==============================================
Vector acc_g(int i, int *Typ, float *x, float *y, float *z, float *eps, float *mass, float G, int N)
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
  
  Vector acc = {accxt, accyt, acczt};

  return acc;
    
}


int main()
{


  //********************************************************************
  //**************** Reading the params.txt file ***********************
  //********************************************************************
  std::string filename;
  int N, ndx_BH;
  float G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
        unitLength_in_cm;

  readParams(filename, N, ndx_BH, G, L_AGN_code_unit, M_dot_in, v_in, u_for_10K_Temp, m_sph_high_res, sigma, UnitDensity_in_cgs, Unit_u_in_cgs, unitTime_in_s,
             unitLength_in_cm);


  auto data = readVectorsFromFile("IC_R_1334k.bin");
  std::vector<int> &Typ = std::get<0>(data);
  std::vector<float> &x = std::get<1>(data);
  std::vector<float> &y = std::get<2>(data);
  std::vector<float> &z = std::get<3>(data);
  std::vector<float> &vx = std::get<4>(data);
  std::vector<float> &vy = std::get<5>(data);
  std::vector<float> &vz = std::get<6>(data);
  std::vector<float> &u = std::get<7>(data);
  std::vector<float> &h = std::get<8>(data);
  std::vector<float> &eps = std::get<9>(data);
  std::vector<float> &mass = std::get<10>(data);
  
  

  int i = 213509;

  Vector acc = acc_g(i, &Typ[0], &x[0], &y[0], &z[0], &eps[0], &mass[0], G, N);
  
  cout << "x = " << x[i] << endl;
  cout << "x = " << y[i] << endl;
  cout << "x = " << z[i] << endl;
  cout << endl;
  
  cout << "accx = " << acc.x << endl;
  cout << "accy = " << acc.y << endl;
  cout << "accz = " << acc.z << endl;
  

}


