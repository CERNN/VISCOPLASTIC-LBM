/*
*   @file boundaryConditionsBuilder.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Build boundary condition grid
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __BOUNDARY_CONDITIONS_BUILDER_H
#define __BOUNDARY_CONDITIONS_BUILDER_H

#include <cuda.h>
#include "globalFunctions.h"
#include "structs/populations.h"

/*
*   @brief Builds boundary conditions map
*   @param gpuMapBC: device pointer to the boundary conditions map
*/
__global__
void gpuBuildBoundaryConditions(NodeTypeMap* const gpuMapBC);


#endif // !__BOUNDARY_CONDITIONS_BUILDER_H