#ifndef MYPHOTOLIBSGPU
#define MYPHOTOLIBSGPU

#include <iostream>
#include <cmath>
#include <string>
using namespace std;

struct hfv_type
{
    float v1;
    float v2;
    float v3;
    float v4;
    float v5;
    float v6;
    float v7;
};

/* lamb_to_neu */
__host__ __device__ float lamb_to_neu(float lamb)
{

    float clight = 29979245800.0f; /* cm/s */
    float lamb_in_cm = lamb * 1e-8;

    return clight / lamb_in_cm;
}

/* neu_to_lamb (lamb will be in Angstrom) */
__host__ __device__ float neu_to_lamb(float neu)
{

    float clight = 29979245800.0f; /* cm/s */

    return clight / neu * 1e8;
}

/* eV_to_neu */
__host__ __device__ float eV_to_neu(float eV)
{

    float h_in_eV = 4.135667696e-15; /* h in eV.Hz^-1 */
    return eV / h_in_eV;
}

/* neu_to_eV */
__host__ __device__ float neu_to_eV(float neu)
{

    float h_in_eV = 4.135667696e-15; /* h in eV.Hz^-1 */
    return h_in_eV * neu;
}

/* eV_to_lamb (lamb will be in Angstrom)*/
__host__ __device__ float eV_to_lamb(float eV)
{

    float neu = eV_to_neu(eV);
    return neu_to_lamb(neu);
}

/* lamb_to_eV */
__host__ __device__ float lamb_to_eV(float lamb)
{

    float neu = lamb_to_neu(lamb);
    return neu_to_eV(neu);
}

/* ryd_to_eV */
float ryd_to_eV(float ryd)
{

    return 13.60f * ryd;
}

/* eV_to_ryd */
__host__ __device__ float eV_to_ryd(float eV)
{

    return eV / 13.60f;
}

/* fneu_to_flamb */
__host__ __device__ double fneu_to_flamb(float fneu, float neu)
{

    float lamb = neu_to_lamb(neu);
    double flamb = 2.99792458e18 / lamb / lamb * fneu;

    return flamb;
}

/* flamb_to_fneu */
__host__ __device__ float flamb_to_fneu(double flamb, float lamb)
{

    float neu = lamb_to_neu(lamb);
    float fneu = 2.99792458e18 / neu / neu * flamb;

    return fneu;
}

/* pc_to_cm */
__host__ __device__ float pc_to_cm(float pc)
{

    return pc * 3.086e+18;
}

/* cm_to_pc */
__host__ __device__ float cm_to_pc(float cm)
{

    return cm / 3.086e+18;
}

/* phCrossSection */
__host__ __device__ void phCrossSection(float *neu, float *sig, float E0, float sig0, float ya,
                                        float P, float yw, float y0, float y1, int N)
{

    float E, x, y, Fy;

    for (int i = 0; i < N; i++)
    {

        E = neu_to_eV(neu[i]);
        x = E / E0 - y0;
        y = sqrt(x * x + y1 * y1);

        Fy = (((x - 1.0f) * (x - 1.0f) + yw * yw) * pow(y, (0.5f * P - 5.5f))) * pow(1.0f + sqrt(y / ya), -P);
        sig[i] = sig0 * Fy * 1e-18;
    }
}

/* J_neu_func */
__host__ __device__ void J_neu_func(float *neu, float *Jneu, float eV, float Jcoeff, float J912QSO, int N)
{

    /* J_neu from Vedel et al. 1994 */
    // float z_redshift = 4.0;
    float J_minus_21 = Jcoeff;
    float neu0 = lamb_to_neu(eV_to_lamb(eV));

    for (int i = 0; i < N; i++)
    {

        Jneu[i] = J_minus_21 * 1e-21 / (neu[i] / neu0);
    }
}

/* RadiationField */
__host__ __device__
    hfv_type
    RadiationField()
{
    hfv_type HeatResults;

    // string RFiD = "NotQSO"; /* if you want J912 of a quasar then set this to "QSO". */
    float Jcoeff = 0.00001f;

    /* A typical Eclisping DLA Quasar RF */
    double L_912_t = 3.25e42 / 1e20; // To avoid using long double! Below we multiply back 1e20!!
    float L_912 = flamb_to_fneu(L_912_t, 912.0f) * 1e20;
    float dist = 500.0f; // pc
    float dist_cm = pc_to_cm(dist);
    float J912QSO = 1.0f / (4.0f * M_PI) * L_912 / (4.0f * M_PI * dist_cm * dist_cm);
    /* J912QSO is J at lamb = 912 A, and we assume a flat spectrum from 1 to 20 Ryd (Fathivavsari et al. 2015)*/
    /*--------------------------------------*/

    /* E is the photon energy in eV */
    float E_H0 = 13.60;  // eV
    float E_He0 = 24.59; // eV
    float E_Hep = 54.42; // eV

    int NN = 200;
    float neu_H0 = eV_to_neu(E_H0) / 1e16; /* This is neu at photo-ionisation of H0. Note division by 1e16. */
    float stpH0 = (6.0f - neu_H0) / NN;
    float *neuH0 = new float[NN];

    float neu_He0 = eV_to_neu(E_He0) / 1e16; /* This is neu at photo-ionisation of He0 */
    float stpHe0 = (6.0f - neu_He0) / NN;
    float *neuHe0 = new float[NN];

    float neu_Hep = eV_to_neu(E_Hep) / 1e16; /* This is neu at photo-ionisation of Hep */
    float stpHep = (6.0f - neu_Hep) / NN;
    float *neuHep = new float[NN];

    for (int i = 0; i < NN; i++)
    {
        neuH0[i] = (neu_H0 + i * stpH0) * 1e16;
        neuHe0[i] = (neu_He0 + i * stpHe0) * 1e16;
        neuHep[i] = (neu_Hep + i * stpHep) * 1e16;
    }

    float *sigH0 = new float[NN];
    float *sigHe0 = new float[NN];
    float *sigHep = new float[NN];

    phCrossSection(neuH0, sigH0, 4.298e-1, 5.475e4, 3.288e1, 2.963, 0.0, 0.0, 0.0, NN);            // H0
    phCrossSection(neuHe0, sigHe0, 1.361E+1, 9.492E2, 1.469E0, 3.188, 2.039, 4.434E-1, 2.136, NN); // He0
    phCrossSection(neuHep, sigHep, 1.720E+0, 1.369E4, 3.288E1, 2.963, 0.0, 0.0, 0.0, NN);          // Hep

    /**********************************************************************/
    /************************ Photoionization Rate ************************/
    /**********************************************************************/
    float hplanck = 6.626196E-27; // h in erg.Hz^-1 or erg.s

    float *JneuH0 = new float[NN];
    float *JneuHe0 = new float[NN];
    float *JneuHep = new float[NN];
    J_neu_func(neuH0, JneuH0, E_H0, Jcoeff, J912QSO, NN);
    J_neu_func(neuHe0, JneuHe0, E_He0, Jcoeff, J912QSO, NN);
    J_neu_func(neuHep, JneuHep, E_Hep, Jcoeff, J912QSO, NN);

    float *fxH0 = new float[NN];
    float *fxHe0 = new float[NN];
    float *fxHep = new float[NN];

    for (int i = 0; i < NN; i++)
    {

        fxH0[i] = 4.0f * M_PI * JneuH0[i] * sigH0[i] / hplanck / neuH0[i];
        fxHe0[i] = 4.0f * M_PI * JneuHe0[i] * sigHe0[i] / hplanck / neuHe0[i];
        fxHep[i] = 4.0f * M_PI * JneuHep[i] * sigHep[i] / hplanck / neuHep[i];
    }

    /*** H0, He0, Hep Photoionization Rates ***/
    float delta_neu_H0 = neuH0[1] - neuH0[0];
    float delta_neu_He0 = neuHe0[1] - neuHe0[0];
    float delta_neu_Hep = neuHep[1] - neuHep[0];

    float gJH0 = 0.0f;
    float gJHe0 = 0.0f;
    float gJHep = 0.0f;

    for (int i = 0; i < NN - 1; i++)
    {

        gJH0 += delta_neu_H0 * (fxH0[i] + fxH0[i + 1]) / 2.0f;
        gJHe0 += delta_neu_He0 * (fxHe0[i] + fxHe0[i + 1]) / 2.0f;
        gJHep += delta_neu_Hep * (fxHep[i] + fxHep[i + 1]) / 2.0f;
    }

    /**********************************************************************/
    /**************************** Heating Rate ****************************/
    /**********************************************************************/

    for (int i = 0; i < NN; i++)
    {

        fxH0[i] = 4.0f * M_PI * JneuH0[i] * sigH0[i] * (neuH0[i] - neu_H0 * 1e16) / neuH0[i];
        fxHe0[i] = 4.0f * M_PI * JneuHe0[i] * sigHe0[i] * (neuHe0[i] - neu_He0 * 1e16) / neuHe0[i];
        fxHep[i] = 4.0f * M_PI * JneuHep[i] * sigHep[i] * (neuHep[i] - neu_Hep * 1e16) / neuHep[i];
    }

    float HRate_H0 = 0.0f;
    float HRate_He0 = 0.0f;
    float HRate_Hep = 0.0f;

    for (int i = 0; i < NN - 1; i++)
    {

        HRate_H0 += delta_neu_H0 * (fxH0[i] + fxH0[i + 1]) / 2.0f;
        HRate_He0 += delta_neu_He0 * (fxHe0[i] + fxHe0[i + 1]) / 2.0f;
        HRate_Hep += delta_neu_Hep * (fxHep[i] + fxHep[i + 1]) / 2.0f;
    }

    delete[] neuH0;
    delete[] neuHe0;
    delete[] neuHep;

    delete[] sigH0;
    delete[] sigHe0;
    delete[] sigHep;

    delete[] JneuH0;
    delete[] JneuHe0;
    delete[] JneuHep;

    delete[] fxH0;
    delete[] fxHe0;
    delete[] fxHep;

    HeatResults = {gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep};

    return HeatResults;
}

/* RandCIRates */
__host__ __device__
    hfv_type
    RandCIRates(float T)
{
    /****** Recombination and Collisional Ionization Rates ************/
    float Tfact = 1.0f / (1.0f + sqrt(T / 1e5));

    // Recombination (Cen 1992)
    // Hydrogen II:
    float AlphaHp = 8.41e-11 * pow(T / 1000.0f, -0.2) / (1. + pow(T / 1e6, 0.7)) / sqrt(T);
    // Helium II:
    float AlphaHep = 1.5e-10 * pow(T, -0.6353);
    // Helium III:
    float AlphaHepp = 4.0f * AlphaHp;

    // dielectric recombination
    float Alphad = 1.9e-3 * pow(T, -1.5) * exp(-470000.0 / T) * (1. + 0.3 * exp(-94000.0 / T));

    // collisional ionization (Cen 1992):
    // Hydrogen:
    float GammaeH0 = 5.85e-11 * sqrt(T) * exp(-157809.1 / T) * Tfact;
    // Helium:
    float GammaeHe0 = 2.38e-11 * sqrt(T) * exp(-285335.4 / T) * Tfact;
    // Helium II:
    float GammaeHep = 5.68e-12 * sqrt(T) * exp(-631515.0 / T) * Tfact;

    hfv_type R_CI_results;

    R_CI_results = {AlphaHp, AlphaHep, AlphaHepp, Alphad, GammaeH0, GammaeHe0, GammaeHep};

    return R_CI_results;
}

/* getAbundance */
__host__ __device__
    hfv_type
    getAbundance(float Temp, float nHcgs, float gJH0, float gJHe0, float gJHep)
{
    float Tfact = 1.0f / (1.0f + sqrt(Temp / 1e5));

    float aHp, aHep, aHepp, ad, geH0, geHe0, geHep;

    hfv_type R_CI_results = RandCIRates(Temp);

    aHp = R_CI_results.v1;
    aHep = R_CI_results.v2;
    aHepp = R_CI_results.v3;
    ad = R_CI_results.v4;
    geH0 = R_CI_results.v5;
    geHe0 = R_CI_results.v6;
    geHep = R_CI_results.v7;

    float Y = 0.24f; // Helium abundance by mass.
    float y = Y / (4.0f - 4.0f * Y);

    // NOTE: all number densities are relative to nH.

    float ne = 1.0f; // initial guess
    float ne_old = ne / 2.0f;

    int MAXITER = 100;
    int niter = 1;

    float nH0, nHp, nHe0, nHep, nHepp, ne_new;

    float dne = 1.0f; /* chosen to be larger than 1e-4 so that the while loop can start !*/

    while ((dne > 1e-4) && (niter < MAXITER))
    {
        ne_old = ne;

        nH0 = aHp / (aHp + geH0 + gJH0 / (ne * nHcgs));
        nHp = 1.0f - nH0;
        nHep = y / (1.0f + (aHep + ad) / (geHe0 + gJHe0 / (ne * nHcgs)) + (geHep + gJHep / (ne * nHcgs)) / aHepp);
        nHe0 = nHep * (aHep + ad) / (geHe0 + gJHe0 / (ne * nHcgs));
        nHepp = nHep * (geHep + gJHep / (ne * nHcgs)) / aHepp;
        ne = nHp + nHep + 2.0f * nHepp;

        ne_new = 0.5f * (ne + ne_old);
        ne = ne_new;
        dne = abs(ne - ne_old);

        niter++;
    }

    /*
    if (niter >= MAXITER)
    {
        cout << "Number of iterations reached maximum in the getAbundance function!!!";
        exit(1);
    }
    */

    hfv_type Abund_results;

    Abund_results = {nH0, nHe0, nHp, ne, nHep, nHepp};

    return Abund_results;
}

/* coolinHeatingRates (For your desired Radiation Field, please modify the 'RadiationField' function.) */
__host__ __device__ float coolingHeatingRates(float Temp, float nHcgs)
{

    hfv_type HeatResults = RadiationField();

    float gJH0, gJHe0, gJHep, HRate_H0, HRate_He0, HRate_Hep;
    float aHp, aHep, aHepp, ad, geH0, geHe0, geHep;
    float nH0, nHe0, nHp, ne, nHep, nHepp;

    /* Note: we must multiply these values by the nH0, nHe0, and nHep
       to convert them to heating Rate by photoionization.*/
    gJH0 = HeatResults.v1;
    gJHe0 = HeatResults.v2;
    gJHep = HeatResults.v3;
    HRate_H0 = HeatResults.v4;
    HRate_He0 = HeatResults.v5;
    HRate_Hep = HeatResults.v6;

    hfv_type R_CI_results = RandCIRates(Temp);

    aHp = R_CI_results.v1;
    aHep = R_CI_results.v2;
    aHepp = R_CI_results.v3;
    ad = R_CI_results.v4;
    geH0 = R_CI_results.v5;
    geHe0 = R_CI_results.v6;
    geHep = R_CI_results.v7;

    hfv_type Abund_results = getAbundance(Temp, nHcgs, gJH0, gJHe0, gJHep);

    nH0 = Abund_results.v1;
    nHe0 = Abund_results.v2;
    nHp = Abund_results.v3;
    ne = Abund_results.v4;
    nHep = Abund_results.v5;
    nHepp = Abund_results.v6;

    float HeatingRate_H0 = nH0 * HRate_H0 / nHcgs; // This is H/nH^2
    float HeatingRate_He0 = nHe0 * HRate_He0 / nHcgs;
    float HeatingRate_Hep = nHep * HRate_Hep / nHcgs;

    float Total_HeatingRate = HeatingRate_H0 + HeatingRate_He0 + HeatingRate_Hep;

    //************* COOLING SECTION **********************
    float Tfact = 1.0f / (1.0f + sqrt(Temp / 1e5));
    // collisional excitation (Cen 1992):
    float BetaH0 = 7.5e-19 * exp(-118348.0f / Temp) * Tfact;
    float BetaHep = 5.54e-17 * pow(Temp, -0.397f) * exp(-473638.0f / Temp) * Tfact;
    // free-free:
    float cte = 5.5f - log10(Temp);
    float Betaff = 1.43e-27 * sqrt(Temp) * (1.0f + 0.34f * exp(-cte * cte / 3.0f));

    float LambdaExcH0 = BetaH0 * ne * nH0;
    float LambdaExcHep = BetaHep * ne * nHep;
    float LambdaExc = LambdaExcH0 + LambdaExcHep; // excitation

    float LambdaIonH0 = 2.18e-11 * geH0 * ne * nH0;
    float LambdaIonHe0 = 3.94e-11 * geHe0 * ne * nHe0;
    float LambdaIonHep = 8.72e-11 * geHep * ne * nHep;
    float LambdaIon = LambdaIonH0 + LambdaIonHe0 + LambdaIonHep; // ionization

    float LambdaRecHp = 1.036e-16 * Temp * ne * (aHp * nHp);
    float LambdaRecHep = 1.036e-16 * Temp * ne * (aHep * nHep);
    float LambdaRecHepp = 1.036e-16 * Temp * ne * (aHepp * nHepp);
    float LambdaRecHepd = 6.526e-11 * ad * ne * nHep;
    float LambdaRec = LambdaRecHp + LambdaRecHep + LambdaRecHepp + LambdaRecHepd;

    float LambdaFF = Betaff * (nHp + nHep + 4 * nHepp) * ne;

    float Lambda = LambdaExc + LambdaIon + LambdaRec + LambdaFF;

    /* note that Total_HeatingRate is HeatingRate/nH^2 and Lambda is Lambda/nH^2. */
    return Total_HeatingRate - Lambda;
}

/* convert_u_to_Temp */
__host__ __device__ float convert_u_to_Temp(float u, float nHcgs, float XH)
{

    // u MUST be in physical units !!!!!!
    float kB = 1.3807e-16; // cm2 g s-2 K-1
    float mH = 1.6726e-24; // gram
    float gamma = 5.0f / 3.0f;

    float yHelium = (1.0f - XH) / (4.0f * XH);

    float ne_guess = 1.0f; // our initial guess is that elec_density = hydrogen density.
    float mu = (1.0f + 4.0f * yHelium) / (1.0f + yHelium + ne_guess);
    // yHelium = nHe/nH and ne = ne/nH
    float Temp = (gamma - 1.0f) * mH / kB * mu * u;

    int MAXITER = 100;
    float Temp_old = Temp / 2.0f;
    int niter = 1;

    float ne, new_Temp;
    float gJH0, gJHe0, gJHep;

    hfv_type HeatResults = RadiationField();

    gJH0 = HeatResults.v1;
    gJHe0 = HeatResults.v2;
    gJHep = HeatResults.v3;

    hfv_type Abund_results;

    float dTemp = Temp;

    while ((dTemp / Temp > 1e-4) && (niter < MAXITER))
    {

        Temp_old = Temp;

        // updating ne
        Abund_results = getAbundance(Temp_old, nHcgs, gJH0, gJHe0, gJHep);
        ne = Abund_results.v4;

        mu = (1.0f + 4.0f * yHelium) / (1.0f + yHelium + ne);
        Temp = (gamma - 1.0f) * mH / kB * mu * u;

        new_Temp = 0.5f * (Temp + Temp_old);
        Temp = new_Temp;
        dTemp = abs(Temp - Temp_old);

        niter++;
    }

    /*
    if (niter >= MAXITER)
    {
        cout << "Number of iterations reached maximum in the convert_u_to_Temp function!!!";
        exit(1);
    }
    */

    return Temp;
}

/* convert_Temp_to_u */
__host__ __device__ float convert_Temp_to_u(float Temp, float nHcgs, float XH)
{

    float kB = 1.3807e-16; // cm2 g s-2 K-1
    float mH = 1.6726e-24; // gram
    float gamma = 5.0f / 3.0f;

    float yHelium = (1.0f - XH) / (4.0f * XH);

    float ne_guess = 1.0f; /*our initial guess is that elec_density = hydrogen density.*/
    float mu = (1.0f + 4.0f * yHelium) / (1.0f + yHelium + ne_guess);
    // yHelium = nHe/nH and ne = ne/nH

    float u = kB / mH / (gamma - 1.0f) / mu * Temp;

    int MAXITER = 100;
    float u_old = u / 2.0f;
    int niter = 1;

    float ne;
    float gJH0, gJHe0, gJHep;

    hfv_type HeatResults = RadiationField();

    gJH0 = HeatResults.v1;
    gJHe0 = HeatResults.v2;
    gJHep = HeatResults.v3;

    hfv_type Abund_results;

    float Temp_old, new_u;

    float du = u;

    while ((du / u > 1e-4) && (niter < MAXITER))
    {

        u_old = u;

        // updating ne
        Temp_old = convert_u_to_Temp(u, nHcgs, XH);

        Abund_results = getAbundance(Temp_old, nHcgs, gJH0, gJHe0, gJHep);
        ne = Abund_results.v4;

        mu = (1.0f + 4.0f * yHelium) / (1.0f + yHelium + ne);
        u = kB / mH / (gamma - 1.0f) / mu * Temp;

        new_u = 0.5 * (u + u_old);
        u = new_u;
        du = abs(u - u_old);

        niter++;
    }

    return u;
}

/* coolingRateFromU */
__host__ __device__ float coolingRateFromU(float u, float nHcgs, float XH)
{

    // Note that u must be in physical units (i.e. cgs);

    float Tmin = log10(1e4);
    float Tmax = log10(1e8);

    float Temp = convert_u_to_Temp(u, nHcgs, XH);

    if (Temp < pow(10, Tmin))
    {
        Temp = pow(10, Tmin);
    }

    if (Temp > pow(10, Tmax))
    {
        Temp = pow(10, Tmax);
    }

    float Heating_minus_cooling = coolingHeatingRates(Temp, nHcgs);

    return Heating_minus_cooling;
}

/* DoCooling */
__host__ __device__ float DoCooling(float rho, float u_old, float dt, float XH)
{
    int MAXITER = 100;
    int niter = 1;
    float mH = 1.6726e-24;                // gram
    float nHcgs = XH * rho / mH;          // hydrogen number dens in cgs units
    float ratefact = nHcgs * nHcgs / rho; // The effect of the density on the cooling is here.

    float u = u_old;
    float u_upper = u;

    float GammaLambdaNet = coolingRateFromU(u, nHcgs, XH);

    float u_lower;

    if (u - u_old - ratefact * GammaLambdaNet * dt > 0.0f)
    {
        while (u_upper - u_old - ratefact * coolingRateFromU(u_upper, nHcgs, XH) * dt > 0.0f)
        {

            u_upper /= 1.1f;
        }

        u_lower = u_upper;
        u_upper = u_lower * 1.1f;
    }

    if (u - u_old - ratefact * GammaLambdaNet * dt < 0.0f)
    {
        while (u_upper - u_old - ratefact * coolingRateFromU(u_upper, nHcgs, XH) * dt < 0.0f)
        {

            u_upper *= 1.1f;
        }

        u_lower = u_upper / 1.1f;
        // u_upper = u_upper;
    }

    float du = u;

    while ((abs(du / u) > 0.0001f) && (niter < MAXITER))
    {

        u = 0.5f * (u_lower + u_upper);

        GammaLambdaNet = coolingRateFromU(u, nHcgs, XH);

        if (u - u_old - ratefact * GammaLambdaNet * dt > 0.0f)
        {
            u_upper = u;
        }
        else
        {
            u_lower = u;
        }

        du = abs(u_upper - u_lower);

        niter++;
    }

    /*
    if (niter >= MAXITER)
    {
        cout << "Number of iterations reached maximum in the DoCooling_h function!!!";
        exit(1);
    }
    */

    return u;
}

#endif
