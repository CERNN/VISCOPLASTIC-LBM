#include "nnf.h"


#ifdef NON_NEWTONIAN_FLUID
__device__
void calcOmega(dfloat* __restrict__ omegaOld, dfloat* __restrict__ gammaOld,
    dfloat pXX, dfloat pYY, dfloat pZZ, 
    dfloat pXY, dfloat pXZ, dfloat pYZ,
    dfloat uxVar, dfloat uyVar, dfloat uzVar, dfloat rhoVar,
    dfloat fxVar, dfloat fyVar, dfloat fzVar, dfloat lambda) {

    //calculate the magnitude of the stress tensor
    const dfloat stress_mag = sqrt(0.5 * (
        (pXX + 0.5 * (2.0 * uxVar * fxVar)) * (pXX + 0.5 * (2.0 * uxVar * fxVar)) +
        (pYY + 0.5 * (2.0 * uyVar * fyVar)) * (pYY + 0.5 * (2.0 * uyVar * fyVar)) +
        (pZZ + 0.5 * (2.0 * uzVar * fzVar)) * (pZZ + 0.5 * (2.0 * uzVar * fzVar)) +
        (pXY + 0.5 * (uxVar * fyVar + uyVar * fxVar)) * (pXY + 0.5 * (uxVar * fyVar + uyVar * fxVar)) +
        (pXZ + 0.5 * (uxVar * fzVar + uzVar * fxVar)) * (pXZ + 0.5 * (uxVar * fzVar + uzVar * fxVar)) + 
        (pYZ + 0.5 * (uyVar * fzVar + uzVar * fyVar)) * (pYZ + 0.5 * (uyVar * fzVar + uzVar * fyVar))));

    dfloat omega; 
    dfloat tau = 1 / omegaOld;

    // apparent viscosity
    dfloat eta = (tau - 0.5) / 3.0;

    //rate of strain
    const dfloat gamma_dot = (1 - 0.5 * (*omegaOld)) * stress_mag / eta;

#ifdef POWERLAW
    if (gamma_dot <= GAMMA_0) {
        eta = ETA_0;
    }
    else {
        eta = ETA_0 * POW_FUNCTION((double)gamma_dot, (double)(N_INDEX - 1));
    }
    tau = 0.5 + 3 * eta;
    omega = 1 / tau;
#endif // POWERLAW

#ifdef BINGHAM
    tau = 0.5 + 3 * ETA_0;
    omega = 1 / tau;
    omega *= (dfloat)(1 - S_Y / stress_mag);
    if(omega < 0)
        omega = 0;

#endif // BINGHAM

    // return std::make_pair(omega, gamma_dot);
}
#endif // NON_NEWTONIAN_FLUID
