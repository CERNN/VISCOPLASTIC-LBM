#ifndef __D3Q19_H
#define __D3Q19_H

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

constexpr unsigned char Q = 19;         // number of velocities
constexpr dfloat W0 = 1.0 / 3;         // population 0 weight (0, 0, 0)
constexpr dfloat W1 = 1.0 / 18;        // adjacent populations (1, 0, 0)
constexpr dfloat W2 = 1.0 / 36;        // diagonal populations (1, 1, 0)

// velocities weight vector
__device__ const dfloat w[Q] = { W0,
    W1, W1, W1, W1, W1, W1,
    W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2, W2
};

// populations velocities vector
__device__ const char cx[Q] = { 0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0 };
__device__ const char cy[Q] = { 0, 0, 0, 1,-1, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 1,-1 };
__device__ const char cz[Q] = { 0, 0, 0, 0, 0, 1,-1, 0, 0, 1,-1, 1,-1, 0, 0,-1, 1,-1, 1 };

#endif // !__D3Q19_H