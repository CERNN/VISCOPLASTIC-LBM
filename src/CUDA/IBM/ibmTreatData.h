#ifndef __IBM_TREAT_DATA_H
#define __IBM_TREAT_DATA_H

#include "structs/ibmProc.h"

/**
*   @brief Allocate necessary variables, if required dynamic allocation 
*   
*   @param processingIBM: object to allocate to
*/
void allocateIBMProc(IBMProc* processingIBM);

/**
*   @brief Free necessary variables, if required dynamic allocation 
*   
*   @param processingIBM: object to free allocated variables
*/
void freeIBMProc(IBMProc* processingIBM);

/**
*   @brief Treat data for IBM
*
*   @param processingIBM: IBM treat data object
*   @param particles: particles to use
*/
void treatDataIBM(IBMProc* processingIBM, ParticlesSoA particles);

/**
*   @brief Stop simulation due to IBM values
*
*   @param processingIBM: IBM treat data object
*   @param particles: particles to use
*   @return true if simulation must stop, false otherwise
*/
bool stopSimIBM(IBMProc* processingIBM, ParticlesSoA particles);

/**
*   @brief Print IBM treat data information
*
*   @param processingIBM: IBM treat data object
*/
void printTreatDataIBM(IBMProc* processingIBM);

/**
*   @brief Save IBM treat data information
*
*   @param processingIBM: IBM treat data object
*/
void saveTreatDataIBM(IBMProc* processingIBM);

#endif //!__IBM_TREAT_DATA_H