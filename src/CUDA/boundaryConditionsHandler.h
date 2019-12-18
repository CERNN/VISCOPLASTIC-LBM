/*
*   @file boundaryConditionsHandler.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Handle application of boundary conditions
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __BOUNDARY_CONDITIONS_HANDLER_H
#define __BOUNDARY_CONDITIONS_HANDLER_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "structs/populations.h"
#include "boundaryConditionsSchemes/bounceBack.h"
#include "boundaryConditionsSchemes/freeSlip.h"
#include "boundaryConditionsSchemes/interpolatedBounceBack.h"
#ifdef D3Q19
#include "boundaryConditionsSchemes/D3Q19_VelBounceBack.h"
#include "boundaryConditionsSchemes/D3Q19_VelZouHe.h"
#include "boundaryConditionsSchemes/D3Q19_PresZouHe.h"
#endif // !D3Q19

/*
*   @brief Applies boundary conditions given node type and its population
*   @param gpuNT: node's map
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__
void gpuBoundaryConditions(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z);


/*
*   @brief Applies specials boundaries conditions given node's population
*   @param gpuNT: node's map
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__
void gpuSchSpecial(NodeTypeMap* gpuNT,
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z);


/*
*   @brief Applies free slip boundary condition given node's type
*   @param gpuNT: node's map
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__
void gpuSchFreeSlip(NodeTypeMap* gpuNT,
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z);


/*
*   @brief Applies bounce back boundary condition given node's population
*   @param gpuNT: node's map
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__
void gpuSchBounceBack(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x,
    const short unsigned int y,
    const short unsigned int z);


/*
*   @brief Applies velocity bounce back boundary condition given node's 
*          population
*   @param gpuNT: node's map
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__
void gpuSchVelBounceBack(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z);


/*
*   @brief Applies pressure non-equilibrium bounce back boundary condition 
*          given node's population
*   @param gpuNT: node's map
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__
void gpuSchPresZouHe(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z);


/*
*   @brief Applies velocity Zou-He boundary condition given node's population
*   @param gpuNT: node's map
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*/
__device__
void gpuSchVelZouHe(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z);


#endif // !__BOUNDARY_CONDITIONS_HANDLER_H