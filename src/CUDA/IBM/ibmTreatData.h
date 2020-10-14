#ifndef __IBM_TREAT_DATA_H
#define __IBM_TREAT_DATA_H

#include "structs/particleProc.h"

void treatDataIBM(ParticleProc* processingIBM);

bool stopSimIBM(ParticleProc* processingIBM);

void printTreatDataIBM(ParticleProc* processingIBM);

void saveTreatDataIBM(ParticleProc* processingIBM);

#endif //!__IBM_TREAT_DATA_H