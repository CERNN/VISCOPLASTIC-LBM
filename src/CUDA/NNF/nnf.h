#ifndef __NNF_H
#define __NNF_H

#include <math.h>
#include "./../var.h"


/* ------------------------ NON NEWTONIAN FLUID TYPE ------------------------ */
#ifdef POWERLAW
constexpr dfloat N_INDEX = 1.5;                         // Power index
constexpr dfloat K_CONSISTENCY = RHO_0*(TAU-0.5)/3;      // Consistency factor
constexpr dfloat GAMMA_0 = 0;       // Truncated Power-Law. 
                                    // Leave as 0 to no truncate
#define OMEGA_LAST_STEP // Needs omega from last step
#endif

#ifdef BINGHAM
// Inputs
constexpr dfloat ETA_P = 0.3979814;              // Plastic viscosity
constexpr dfloat S_Y = 3.4545e-4;                // Yield stress
// Calculated variables
constexpr dfloat OMEGA_P = 1 / (3*ETA_P+0.5);    // 1/tau_p = 1/(3*eta_p+0.5)
#endif
/* -------------------------------------------------------------------------- */

#if defined(HERSCHEL_BULKLEY) || defined(POWERLAW) || defined(BINGHAM)
    #define NON_NEWTONIAN_FLUID
#endif

#ifdef NON_NEWTONIAN_FLUID
__device__ 
dfloat __forceinline__ calcOmega(
    dfloat omegaOld, dfloat const auxStressMag
){
    dfloat omega;

#ifdef POWERLAW
    // Apparent viscosity
    dfloat eta = ((1/omegaOld) - 0.5) / 3.0;

    // Rate of strain
    const dfloat gammaDot = (1 - 0.5 * omegaOld) * auxStressMag / eta;

    if (gammaDot <= GAMMA_0) {
        eta = K_CONSISTENCY;
    }
    else {
        eta = K_CONSISTENCY * POW_FUNCTION(gammaDot, N_INDEX - 1);
    }

    omega = 1 / (0.5 + 3 * eta);
#endif // POWERLAW

#ifdef BINGHAM
    omega = OMEGA_P * myMax(0.0, (1 - S_Y / auxStressMag));
#endif // BINGHAM

    return omega;
}
#endif // NON_NEWTONIAN_FLUID


#endif // __NNF_H