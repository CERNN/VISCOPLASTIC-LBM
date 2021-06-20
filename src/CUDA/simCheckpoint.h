/*
*   @file lbmCheckpoint.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Loading and saving simulation checkpoints
*   @version 0.3.0
*   @date 19/06/2021
*/

#include <string>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "globalFunctions.h"
#include "structs/macroscopics.h"
#include "structs/populations.h"
#include "NNF/nnf.h"
#include "IBM/ibm.h"



/**
*   @brief Load simulation checkpoint
*
*   @param pop Populations array
*   @param macr Macroscopics array
*   @param particlesSoA Particles structure of arrays object
*   @param step Pointer to current step value in main
*/
__host__
void loadSimCheckpoint( 
    Populations pop[N_GPUS],
    Macroscopics macr[N_GPUS],
    ParticlesSoA particlesSoA,
    int *step
);

/**
*   @brief Save simulation checkpoint
*
*   @param pop Populations array
*   @param macr Macroscopics array
*   @param particlesSoA Particles structure of arrays object
*   @param step Pointer to current step value in main
*/
__host__
void saveSimCheckpoint( 
    Populations pop[N_GPUS],
    Macroscopics macr[N_GPUS],
    ParticlesSoA particlesSoA,
    int *step
);
