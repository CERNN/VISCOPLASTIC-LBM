/*
*   @file gInitialization.h
*   @author Marco Ferrari. (marcoferrari@alunos.utfpr.edu.br)
*   @brief Initialization of Additional populations and macroscopics
*   @version 0.3.0
*   @date 16/12/2019
*/


#ifndef __G_INITIALIZATION_H
#define __G_INITIALIZATION_H

#include <string>
#include <math.h>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <builtin_types.h>
#include <device_launch_parameters.h>
#include "../globalFunctions.h"
#include "../structs/macroscopics.h"
#include "../NNF/nnf.h"
#include "gVar.h"
#include "gPopulations.h"
#include "gLBM.h"

/*
*   @brief Initializes device populations for scalar variable with macroscopics values
*   @param pop: populations to be initialized
*   @param macr: macroscopics values
*/
__global__
void gpuInitializationG(
    gPopulations* pop,
    Macroscopics* macr
);



#endif // !__G_INITIALIZATION_H

