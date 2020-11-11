#ifndef __NNF_H
#define __NNF_H

#include <math.h>
#include "./../var.h"


/* ------------------------ NON NEWTONIAN FLUID TYPE ------------------------ */
//#define POWERLAW
//#define BINGHAM
//#define HERSCHEL_BULKLEY

#ifdef BINGHAM
    constexpr dfloat S_Y = 1.49e-3;  //yield rate of strain
    constexpr dfloat ETA_0 = RHO_0 * (TAU - 0.5) / 3;
#endif

#ifdef POWERLAW
    constexpr dfloat N_INDEX = 0.55;
    constexpr dfloat ETA_0 = RHO_0*(TAU-0.5)/3;
    constexpr dfloat GAMMA_0 = 0.00;

#endif

#ifdef HERSCHEL_BULKLEY
    constexpr dfloat N_INDEX = 0.8;
    constexpr dfloat ETA_0 = RHO_0*(TAU-0.5)/3;
    constexpr dfloat S_Y = 2e-5;  // yield rate of strain
#endif //HERSCHEL_BULKLEY
/* -------------------------------------------------------------------------- */

__device__ 
void __forceinline__ calcOmega(
    dfloat* __restrict__ omegaOld, dfloat* __restrict__ gammaOld,
    const dfloat pXX, const dfloat pYY, const dfloat pZZ,
    const dfloat pXY, const dfloat pXZ, const dfloat pYZ,
    const dfloat uxVar, const dfloat uyVar, const dfloat uzVar, const dfloat rhoVar,
    const dfloat fxVar, const dfloat fyVar, const dfloat fzVar, const dfloat lambda
);

#ifdef HERSCHEL_BULKLEY || POWERLAW || BINGHAM
    #define NON_NEWTONIAN_FLUID
#endif

#endif // __NNF_H