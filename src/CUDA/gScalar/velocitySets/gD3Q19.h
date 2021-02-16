#ifndef __gD3Q19_H
#define __gD3Q19_H

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
    [ 7]: ( 1, 1, 0)
    [ 8]: (-1,-1, 0)
    [ 9]: ( 1, 0, 1)
    [10]: (-1, 0,-1)
    [11]: ( 0, 1, 1)
    [12]: ( 0,-1,-1)
    [13]: ( 1,-1, 0)
    [14]: (-1, 1, 0)
    [15]: ( 1, 0,-1)
    [16]: (-1, 0, 1)
    [17]: ( 0, 1,-1)
    [18]: ( 0,-1, 1)
--------------------------
*/

constexpr unsigned char GQ = 19;  
const size_t memSizeGPop = sizeof(dfloat) * NX*NY*NZ * GQ;

// number of velocities
constexpr dfloat gW0 = 1.0 / 3.0;         // population 0 weight (0, 0, 0)
constexpr dfloat gW1 = 1.0 / 18.0;        // adjacent populations (1, 0, 0)
constexpr dfloat gW2 = 1.0 / 36.0;        // adjacent populations (1, 1, 0)

constexpr dfloat inv_gc_s_2 = 3.0;


// velocities weight vector
__device__ const dfloat gw[GQ] = { 
    gW0,
    gW1, gW1, gW1, gW1, gW1, gW1,
    gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2, gW2
};

// populations velocities vector
__device__ const char gcx[GQ] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
__device__ const char gcy[GQ] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
__device__ const char gcz[GQ] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };

#endif // !__gD3Q19_H


