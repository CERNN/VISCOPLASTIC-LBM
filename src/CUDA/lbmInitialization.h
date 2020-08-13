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
#include <device_launch_parameters.h>
#include "globalFunctions.h"
#include "structs/macroscopics.h"
#include "structs/populations.h"

/*
*   @brief Initializes device populations with values in binary file
*   @param pop: populations to be initialized
*   @param filePop: file with population's content
*   @param filePopAux: file with auxiliary population's content
*/
__host__
void initializationPop( 
    Populations* pop,
    FILE* filePop,
    FILE* filePopAux
);


/*
*   @brief Initializes device macroscopics with values in binary file
*   @param macr: macroscopics to be initialized
*   @param fileRho: file with density content
*   @param fileUx: file with ux content
*   @param fileUy: file with uy content
*   @param fileUz: file with uz content
*/
__host__
void initializationMacr(
    Macroscopics* macr,
    FILE* fileRho,
    FILE* fileUx,
    FILE* fileUy,
    FILE* fileUz
);


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
*   @param isMacrInit: macroscopics are already initialized or not
*   @param randomNumbers: vector of random numbers (size is NX*NY*NZ)
*                         useful for turbulence 
*/
__global__
void gpuInitialization(
    Populations pop,
    Macroscopics macr,
    bool isMacrInit,
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