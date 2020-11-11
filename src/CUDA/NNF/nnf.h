#ifndef __NNF_H
#define __NNF_H

#include <math.h>
#include "./../var.h"


/* ------------------------ NON NEWTONIAN FLUID TYPE ------------------------ */
constexpr dfloat ETA_0 = RHO_0 * (TAU - 0.5) / 3;
#ifdef BINGHAM
    constexpr dfloat S_Y = 1.49e-3;  //yield rate of strain
#endif

#ifdef POWERLAW
    constexpr dfloat N_INDEX = 0.55;
    constexpr dfloat GAMMA_0 = 0.00;
#endif

#ifdef HERSCHEL_BULKLEY
    constexpr dfloat N_INDEX = 0.8;
    constexpr dfloat S_Y = 2e-5;  // yield rate of strain
#endif
/* -------------------------------------------------------------------------- */


__device__ 
dfloat __forceinline__ calcOmega(
    dfloat omegaOld, dfloat const stressMag
){
    dfloat omega;

#ifdef POWERLAW
    // Apparent viscosity
    dfloat eta = ((1/omegaOld) - 0.5) / 3.0;

    // Rate of strain
    const dfloat gammaDot = (1 - 0.5 * (omega)) * stressMag / eta;

    if (gamma_dot <= GAMMA_0) {
        eta = ETA_0;
    }
    else {
        eta = ETA_0 * POW_FUNCTION((double)gammaDot, (double)(N_INDEX - 1));
    }

    omega = 1 / (0.5 + 3 * eta);
#endif // POWERLAW

#ifdef BINGHAM
    omega = 1 / (0.5 + 3 * ETA_0);
    omega *= (1 - S_Y / stressMag);
    if(omega < 0)
        omega = 0;
#endif // BINGHAM

    return omega;
}

#if defined(HERSCHEL_BULKLEY) || defined(POWERLAW) || defined(BINGHAM)
    #define NON_NEWTONIAN_FLUID
#endif

#endif // __NNF_H