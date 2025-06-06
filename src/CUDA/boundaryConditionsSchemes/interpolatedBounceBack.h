/*
*   @file interpolatedBounceBack.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Interpolated bounce back boundary condition. It is suited for 
*          cilinders and 2D curved surfaces. Some assumptions are made for 
*          faster processing, as the shape being a cilinder with center
*          in (NX/2, NY/2) and axial direction in Z.
*           https://doi.org/10.1063/1.1399290 
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __BC_INTERPOLATED_BOUNCE_BACK_H
#define __BC_INTERPOLATED_BOUNCE_BACK_H

#include "./../globalFunctions.h"
#include "./../structs/nodeTypeMap.h"
#include <cuda_runtime.h>
#include <cmath>


/*
*   @brief Applies interpolated bounce back boundary condition
*   @param unknownPops: unknown populations (using D2Q9 velocity set) 
*   @param is_inside: Node is an inside Interpolated BC Node (wall is to the center)
                or outside (wall is away from the center) (used for annular duct)
*   @param fPostStream[(NX, NY, NZ, Q)]: populations post streaming
*   @param fPostCol[(NX, NY, NZ, Q)]: post collision populations from last step 
*   @param x: node's x value
*   @param y: node's y value
*   @param z: node's z value
*   @param n_gpu: current gpu processing
*/
__device__ 
void gpuBCInterpolatedBounceBack(const unsigned char unknownPops,
    const bool is_inside,
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x,
    const short unsigned int y,
    const short unsigned int z,
    const int n_gpu);


/*
*   @brief Calculate result population of the interpolated bounce back 
*          boundary condition for q <= 0.5
*   @param fBound: population from boundary node
*   @param fAdj: population from adjacent node
*   @param q: distance between the boundary node and the wall
*   @param df: additional term for wall velocity
*/
__device__ 
dfloat __inline__ gpuInterpolatedBounceBackLowerQ( 
    dfloat fBound,
    dfloat fAdj,
    dfloat q,
    dfloat df)
{
    return (2*q*fBound + (1-2*q)*fAdj + 2*a_s_2*df);
}


/*
*   @brief Calculate result population of the interpolated bounce back 
*          boundary condition for q > 0.5 
*   @param fBoundI: population i from boundary node
*   @param fBoundJ: population j (opposite of i) from boundary node
*   @param q: distance between the boundary node and the wall
*   @param df: additional term for wall velocity
*/
__device__ 
dfloat __inline__ gpuInterpolatedBounceBackHigherQ( 
    dfloat fBoundI,
    dfloat fBoundJ,
    dfloat q,
    dfloat df)
{
    return ((0.5/q)*fBoundI + ((q-0.5)/(q))*fBoundJ + a_s_2*df/q);
}


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_a0(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R
);


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_ainf(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R
);


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_ap1(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R
);


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_am1(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R
);

#endif // !__BC_INTERPOLATED_BOUNCE_BACK_H
