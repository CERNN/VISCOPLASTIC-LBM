/*
*   @file lbmInitialization.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Initialization of LBM populations and macroscopics
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __LBM_INITIALIZATION_H
#define __LBM_INITIALIZATION_H

#include <string>
#include <math.h>
#include <cuda.h>
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
*/
__host__
void initializationPop( 
    Populations* pop,
    FILE* filePop
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
*   @brief Initializes populations with equilibrium population, with density 
*          and velocity defined by "gpuMacrInitValue"
*   @param pop: populations to be initialized in equilibrium
*   @param macr: macroscopics to be initialized by "gpuMacrInitValue"
*   @param isMacrInit: macroscopics are already initialized or not
*/
__global__
void gpuInitialization(
    Populations* pop,
    Macroscopics* macr,
    bool isMacrInit
);


/*
*   @brief Initializes macroscopics value in function of its location.
*          To be called in "gpuInitialization"
*   @param macr: macroscopics to initialize
*   @param x, y, z: location
*/
__device__
void gpuMacrInitValue(
    Macroscopics* macr,
    int x, int y, int z
);



#endif // !__LBM_INITIALIZATION_H