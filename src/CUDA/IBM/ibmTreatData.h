#ifndef __IBM_TREAT_DATA_H
#define __IBM_TREAT_DATA_H

#include "structs/ibmProc.h"

/* Allocate necessary variables, if required dynamic allocation */
void allocateIBMProc(IBMProc* processingIBM);

/* Free allocated variables, if required dynamic allocation */
void freeIBMProc(IBMProc* processingIBM);

void treatDataIBM(IBMProc* processingIBM, ParticlesSoA particles);

bool stopSimIBM(IBMProc* processingIBM, ParticlesSoA particles);

void printTreatDataIBM(IBMProc* processingIBM);

void saveTreatDataIBM(IBMProc* processingIBM);

#endif //!__IBM_TREAT_DATA_H