/*
*   @file ibmReport.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Functions for reporting informations about IBM particles
*   @version 0.3.0
*   @date 13/10/2020
*/


#ifndef __IBM_REPORT_H
#define __IBM_REPORT_H

#include <fstream>
#include <string>
#include <sstream>
#include "../lbmReport.h"
#include "structs/particle.h"

void saveParticlesInfo(ParticlesSoA particles, unsigned int step, bool saveNodes);

void printParticlesInfo(ParticlesSoA particles, unsigned int step);

#endif //!__IBM_REPORT_H