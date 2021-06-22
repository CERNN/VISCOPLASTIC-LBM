/*
*   @file lbmInitialization.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Initialization of LBM populations and macroscopics
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __LBM_INITIALIZATION_H
#define __LBM_INITIALIZATION_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include "globalFunctions.h"
#include "structs/macroscopics.h"
#include "structs/populations.h"
#include "NNF/nnf.h"


/*
*   @brief Initializes random numbers (useful to initialize turbulence)
*   @param randomNumbers: vector of random numbers (size is NX*NY*NZ)
*   @param seed: seed to use for generation
*/
__host__
void initializationRandomNumbers(
    float* randomNumbers, 
    int seed
);


/*
*   @brief Initializes populations with equilibrium population, with density 
*          and velocity defined by "gpuMacrInitValue"
*   @param pop: populations to be initialized in equilibrium
*   @param macr: macroscopics to be initialized by "gpuMacrInitValue"
*   @param randomNumbers: vector of random numbers (size is NX*NY*NZ)
*                         useful for turbulence 
*/
__global__
void gpuInitialization(
    Populations pop,
    Macroscopics macr,
    float* randomNumbers
);


/*
*   @brief Initializes macroscopics value in function of its location.
*          To be called in "gpuInitialization"
*   @param macr: macroscopics to initialize
*   @param randomNumbers: vector of random numbers (size is NX*NY*NZ)
*                         useful for turbulence
*   @param x, y, z: location
*/
__device__
void gpuMacrInitValue(
    Macroscopics* macr,
    float* randomNumbers,
    int x, int y, int z
);



#endif // !__LBM_INITIALIZATION_H