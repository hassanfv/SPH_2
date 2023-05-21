#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <random>
#include <mpi.h>

using namespace std;

struct Particle
{
    int Type = -1; // 0 ==> gas particle, 1 ==> collisionless particle.
    int Nngb_previous;
    float x, y, z, vx, vy, vz, mass, u, h, eps;
    float accx_g, accy_g, accz_g, accx_sph = 0.0f, accy_sph = 0.0f, accz_sph = 0.0f;
    float accx, accy, accz, rho = 0.0f, P = 0.0f, csnd = 0.0f, divV = 0.0f, curlV = 0.0f;
    float dudt = 0.0f, utprevious = 0.0f;
};

//========================================
//========== Smoothing Length ============ Updated: 28 Jan 2023. h_new adopted from eq.31 in Gadget2 Paper
//========================================

// void smoothing_h(Particle p[], int N, int Ndown, int Nup, float coeff,
//                  float Nngb_f, float *divV, float dt)
void smoothing_h(Particle p[], int N, int Ndown, int Nup, float coeff,
                 float Nngb_f, float dt)

{

    for (int i = 0; i < N; i++)
    {

        if (p[i].Type == 0)
        {

            // float h_new = 2.0f * (0.5f * p[i].h * (1.0f + pow((Nngb_f / p[i].Nngb_previous), 1.0f / 3.0f)) + 1.0f / 3.0f * p[i].h * divV[i] * dt);

            float h_new = 2.01f * p[i].h;

            float h_tmp = h_new;
            int N_iter = 0;
            int k = 0;

            float dx, dy, dz;
            while ((k < Ndown) || (k > Nup))
            {

                k = 0;

                for (int j = 0; j < N; j++)
                {
                    if (p[j].Type == 0)
                    {
                        dx = p[j].x - p[i].x;
                        dy = p[j].y - p[i].y;
                        dz = p[j].z - p[i].z;
                        float rr = sqrt(dx * dx + dy * dy + dz * dz);

                        if (rr <= h_new)
                        {
                            k++;
                        }
                    }
                }

                if (k < Ndown)
                {
                    h_new = h_new + coeff * 2.0f * p[i].h;
                }

                if (k > Nup)
                {
                    h_new = h_new - coeff * 2.0f * p[i].h;
                }

                if (h_new > h_tmp)
                {
                    h_tmp = h_new;
                }

                N_iter++;
                if (N_iter > 100)
                {
                    cout << "i, N_iter = " << i << ", " << N_iter << endl;
                    h_new = h_tmp;
                    break;
                }
            }
            p[i].Nngb_previous = k;
            p[i].h = 0.5 * h_new;
        }
    }
}

//==========================================
//============ getDensity ==================
//==========================================
void getDensity(Particle p[], int N)
{

    for (int i = 0; i < N; i++)
    {

        if (p[i].Type == 0)
        {
            float dx, dy, dz, rr, hij, sig, q, hij3;
            float WIij;
            float ss = 0.0f;

            for (int j = 0; j < N; j++)
            {
                if (p[j].Type == 0)
                {

                    dx = p[i].x - p[j].x;
                    dy = p[i].y - p[j].y;
                    dz = p[i].z - p[j].z;

                    rr = sqrt(dx * dx + dy * dy + dz * dz);
                    hij = 0.5f * (p[i].h + p[j].h);

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

                        ss += p[j].mass * WIij;
                    }
                }
            }
            p[i].rho = ss;
        }
    }
}

//==============================================
//================= acc_g ======================
//==============================================
void acc_g(Particle p[], float G, int N)
{

    for (int i = 0; i < N; i++)
    {
        if ((p[i].Type == 0) || p[i].Type == 1)
        {
            float dx, dy, dz, rr, inv_r3, epsij, q, q2, q3, q4, q5, q6, fk;
            float accxt = 0.0f, accyt = 0.0f, acczt = 0.0f;

            for (int j = 0; j < N; j++)
            {
                if ((p[j].Type == 0) || p[j].Type == 1)
                {
                    dx = p[j].x - p[i].x;
                    dy = p[j].y - p[i].y;
                    dz = p[j].z - p[i].z;

                    rr = sqrt(dx * dx + dy * dy + dz * dz);
                    inv_r3 = 1.0f / (rr * rr * rr + 1e-5);
                    epsij = 0.5f * (p[i].eps + p[j].eps);
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

                    accxt += G * fk * dx * p[j].mass;
                    accyt += G * fk * dy * p[j].mass;
                    acczt += G * fk * dz * p[j].mass;
                }
                p[i].accx_g = accxt;
                p[i].accy_g = accyt;
                p[i].accz_g = acczt;
            }
        }
    }
}

//===================================================
//============= getPressure (Adiabatic) =============
//===================================================
void getPressure_Adiabatic(Particle p[], float Gamma, int N)
{

    for (int i = 0; i < N; i++)
    {
        if (p[i].Type == 0)
        {
            p[i].P = (Gamma - 1.0f) * p[i].rho * p[i].u;
        }
    }
}

//===============================================
//============= getCsound (Adiabatic) ===========
//===============================================
void getCsound_Adiabatic(Particle p[], float Gamma, int N)
{

    for (int i = 0; i < N; i++)
    {
        if (p[i].Type == 0)
        {
            p[i].csnd = sqrt(Gamma * (Gamma - 1.0f) * p[i].u);
        }
    }
}

//=====================================================
//================== div_curlVel ======================
//=====================================================
void div_curlVel(Particle p[], int N)
{

    for (int i = 0; i < N; i++)
    {

        if (p[i].Type == 0)
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
                if (p[j].Type == 0)
                {
                    dx = p[j].x - p[i].x;
                    dy = p[j].y - p[i].y;
                    dz = p[j].z - p[i].z;

                    rr = sqrt(dx * dx + dy * dy + dz * dz);
                    hij = 0.5f * (p[i].h + p[j].h);
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

                        vxji = p[j].vx - p[i].vx;
                        vyji = p[j].vy - p[i].vy;
                        vzji = p[j].vz - p[i].vz;

                        ss += p[j].mass / p[i].rho * (vxji * gWx + vyji * gWy + vzji * gWz);

                        vxij = p[i].vx - p[j].vx; //-vxji;
                        vyij = p[i].vy - p[j].vy; //-vyji;
                        vzij = p[i].vz - p[j].vz; //-vzji;

                        curlVx += p[j].mass / p[i].rho * (vyij * gWz - vzij * gWy); // eq. 18 in Beck et al. 2016.
                        curlVy += p[j].mass / p[i].rho * (vzij * gWx - vxij * gWz);
                        curlVz += p[j].mass / p[i].rho * (vxij * gWy - vyij * gWx);
                    }
                }
            }
            p[i].divV = ss; // abs(ss);
            p[i].curlV = sqrt(curlVx * curlVx + curlVy * curlVy + curlVz * curlVz);
        }
    }
}

//===========================================================
//====================== acc_sph ============================
//===========================================================
void acc_sph(Particle p[], float visc_alpha, int N)
{

    for (int i = 0; i < N; i++)
    {

        if (p[i].Type == 0)
        {

            float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW;
            float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
            float axt = 0.0f;
            float ayt = 0.0f;
            float azt = 0.0f;

            for (int j = 0; j < N; j++)
            {
                if (p[j].Type == 0)
                {

                    dx = p[i].x - p[j].x;
                    dy = p[i].y - p[j].y;
                    dz = p[i].z - p[j].z;

                    rr = sqrt(dx * dx + dy * dy + dz * dz);

                    hij = 0.5f * (p[i].h + p[j].h);

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
                        vxij = p[i].vx - p[j].vx;
                        vyij = p[i].vy - p[j].vy;
                        vzij = p[i].vz - p[j].vz;

                        vij_rij = vxij * dx + vyij * dy + vzij * dz;

                        float cij = 0.5f * (p[i].csnd + p[j].csnd);

                        wij = vij_rij / (rr + 1e-5);
                        vij_sig = p[i].csnd + p[j].csnd - 3.0f * wij;
                        rhoij = 0.5f * (p[i].rho + p[j].rho);

                        PIij = 0.0f;
                        if (vij_rij <= 0.0f)
                        {

                            PIij = -0.5f * visc_alpha * vij_sig * wij / rhoij;

                            //------- Shear-viscosity correction -------
                            fi = abs(p[i].divV) / (abs(p[i].divV) + p[i].curlV + 0.0001 * p[i].csnd / p[i].h);
                            fj = abs(p[j].divV) / (abs(p[j].divV) + p[j].curlV + 0.0001 * p[j].csnd / p[j].h);
                            fij = 0.5f * (fi + fj);
                            PIij = fij * PIij;
                            //------- End of Shear-visc. correction -----
                        }

                        axt -= p[j].mass * (p[i].P / p[i].rho / p[i].rho + p[j].P / p[j].rho / p[j].rho + PIij) * gWx;
                        ayt -= p[j].mass * (p[i].P / p[i].rho / p[i].rho + p[j].P / p[j].rho / p[j].rho + PIij) * gWy;
                        azt -= p[j].mass * (p[i].P / p[i].rho / p[i].rho + p[j].P / p[j].rho / p[j].rho + PIij) * gWz;
                    }
                }
            }
            p[i].accx_sph = axt;
            p[i].accy_sph = ayt;
            p[i].accz_sph = azt;
        }
    }
}

//===========================================================
//====================== acc_tot ============================
//===========================================================
void acc_tot(Particle p[], int N)
{

    for (int i = 0; i < N; i++)
    {
        p[i].accx = p[i].accx_g + p[i].accx_sph;
        p[i].accy = p[i].accy_g + p[i].accy_sph;
        p[i].accz = p[i].accz_g + p[i].accz_sph;
    }
}

//===============================================
//=================== get_dU ====================
//===============================================
void get_dU(Particle p[], float visc_alpha, int N)
{

    for (int i = 0; i < N; i++)
    {

        if (p[i].Type == 0)
        {

            float dx, dy, dz, rr, hij, q, sig, hij5, gWx, gWy, gWz, nW, vij_gWij;
            float vxij, vyij, vzij, wij, vij_rij, vij_sig, rhoij, PIij, fi, fj, fij;
            float dut = 0.0f;

            for (int j = 0; j < N; j++)
            {

                if (p[j].Type == 0)
                {

                    dx = p[i].x - p[j].x;
                    dy = p[i].y - p[j].y;
                    dz = p[i].z - p[j].z;

                    rr = sqrt(dx * dx + dy * dy + dz * dz);

                    hij = 0.5f * (p[i].h + p[j].h);

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
                        vxij = p[i].vx - p[j].vx;
                        vyij = p[i].vy - p[j].vy;
                        vzij = p[i].vz - p[j].vz;

                        vij_gWij = vxij * gWx + vyij * gWy + vzij * gWz;

                        vij_rij = vxij * dx + vyij * dy + vzij * dz;

                        float cij = 0.5f * (p[i].csnd + p[j].csnd);

                        wij = vij_rij / (rr + 1e-5);
                        vij_sig = p[i].csnd + p[j].csnd - 3.0f * wij;
                        rhoij = 0.5f * (p[i].rho + p[j].rho);

                        PIij = 0.0f;
                        if (vij_rij <= 0.0f)
                        {

                            PIij = -0.5f * visc_alpha * vij_sig * wij / rhoij;

                            //------- Shear-viscosity correction -------
                            fi = abs(p[i].divV) / (abs(p[i].divV) + p[i].curlV + 0.0001 * p[i].csnd / p[i].h);
                            fj = abs(p[j].divV) / (abs(p[j].divV) + p[j].curlV + 0.0001 * p[j].csnd / p[j].h);
                            fij = 0.5f * (fi + fj);
                            PIij = fij * PIij;
                            //------- End of Shear-visc. correction -----
                        }
                        dut += p[j].mass * (p[i].P / p[i].rho / p[i].rho + PIij / 2.0f) * vij_gWij;
                    }
                }
            }
            p[i].dudt = dut;
        }
    }
}

//===============================================
//================== update u ===================
//===============================================
void u_updater(Particle p[], float dt, int N)
{

    for (int i = 0; i < N; i++)
    {
        if (p[i].Type == 0)
        {
            p[i].u = p[i].u + 0.5f * dt * (p[i].dudt + p[i].utprevious);
            p[i].utprevious = p[i].dudt;
        }
    }
}

//===========================================================
//================= velocity evolution ======================
//===========================================================
void v_evolve(Particle p[], float dt, int N)
{

    for (int i = 0; i < N; i++)
    {
        p[i].vx += p[i].accx * dt / 2.0f;
        p[i].vy += p[i].accy * dt / 2.0f;
        p[i].vz += p[i].accz * dt / 2.0f;
    }
}

//===========================================================
//================= position evolution ======================
//===========================================================
void r_evolve(Particle p[], float dt, int N)
{

    for (int i = 0; i < N; i++)
    {
        p[i].x += p[i].x * dt;
        p[i].y += p[i].y * dt;
        p[i].z += p[i].z * dt;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int nCPUs;
    MPI_Comm_size(MPI_COMM_WORLD, &nCPUs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const float G = 1.0f;
    const float Gamma = 5.0f / 3.0f;
    float visc_alpha = 1.0f;

    int Ndown = 64 - 5;
    int Nup = 64 + 5;
    float Nngb_f = 64.0f;
    float coeff = 0.001f;

    float dt = 1e-2;

    int N = 10000;

    int N_tot = 11000;

    //***************************************
    //********** Reading IC file ************
    //***************************************

    if (rank == 0)
    {
        std::vector<int> Typvec(N_tot);
        std::vector<float> xvec(N_tot);
        std::vector<float> yvec(N_tot);
        std::vector<float> zvec(N_tot);
        std::vector<float> vxvec(N_tot);
        std::vector<float> vyvec(N_tot);
        std::vector<float> vzvec(N_tot);
        std::vector<float> massvec(N_tot);
        std::vector<float> hvec(N_tot);
        std::vector<float> epsvec(N_tot);
        std::vector<float> uvec(N_tot);

        // Check if the binary file exists
        std::ifstream file("IC_000k.bin", std::ios::binary);
        if (!file)
        {
            std::cout << "Could not open the binary file." << std::endl;
        }
        file.close();

        // Open the binary file for reading
        file.open("IC_000k.bin", std::ios::binary);
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
            std::cout << "Failed to open the file." << std::endl;
        }

        Particle p[N_tot];

        for (int i = 0; i < N_tot; i++)
        {

            p[i].Type = Typvec[i];

            p[i].x = xvec[i];
            p[i].y = yvec[i];
            p[i].z = zvec[i];

            p[i].vx = vxvec[i];
            p[i].vy = vyvec[i];
            p[i].vz = vzvec[i];

            p[i].mass = massvec[i];
            p[i].h = hvec[i];
            p[i].eps = epsvec[i];
            p[i].u = uvec[i];
        }
    }

        smoothing_h(p, N, Ndown, Nup, coeff, Nngb_f, dt);

    // getDensity(p, N);

    return 0;
}
