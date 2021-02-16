#ifndef __gD3Q7_H
#define __gD3Q7_H

#include <builtin_types.h> // for device variables

/*
------ POPULATIONS -------
    [ i]: (cx,cy,cz)
    [ 0]: ( 0, 0, 0)
    [ 1]: ( 1, 0, 0)
    [ 2]: (-1, 0, 0)
    [ 3]: ( 0, 1, 0)
    [ 4]: ( 0,-1, 0)
    [ 5]: ( 0, 0, 1)
    [ 6]: ( 0, 0,-1)
--------------------------
*/

constexpr unsigned char GQ = 7;  
const size_t memSizeGPop = sizeof(dfloat) * numberNodes * GQ;

// number of velocities
constexpr dfloat gW0 = 1.0 / 4.0;         // population 0 weight (0, 0, 0)
constexpr dfloat gW1 = 1.0 / 8.0;        // adjacent populations (1, 0, 0)
constexpr dfloat c_s_2 = 1.0 / 4.0;

// velocities weight vector
__device__ const dfloat gw[GQ] = { 
    gW0,
    gW1, gW1, gW1, gW1, gW1, gW1
};

// populations velocities vector
__device__ const char gcx[GQ] = { 0, 1,-1, 0, 0, 0, 0 };
__device__ const char gcy[GQ] = { 0, 0, 0, 1,-1, 0, 0 };
__device__ const char gcz[GQ] = { 0, 0, 0, 0, 0, 1,-1 };

#endif // !__gD3Q7_H


