#include "interpolatedBounceBack.h"

__device__ 
void gpuBCInterpolatedBounceBack(const unsigned char unknownPops,
    const bool is_inside,
    dfloat* fPostStream, 
    dfloat* fPostCol, 
    const short unsigned int x, 
    const short unsigned int y,
    const short unsigned int z)
{
    dfloat q, r, R, xNode, yNode, radius;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    // THIS RADIUS MUST BE THE SAME AS IN THE BOUNDARY CONDITION BUILDER
    R = NY/2.0-0.5;
    r = R/4.0;
    // q = R - distPoints2D(x+0.5, y+0.5, NX/2.0, NY/2.0);

    // Dislocate coordinates to get x^2+y^2=R^2
    xNode = x - NX/2.0 + 0.5;
    yNode = y - NY/2.0 + 0.5;

    if(is_inside)
        radius = r;
    else
        radius = R;

    if(unknownPops & UNKNOWN_POP_1)
    {
        // Populations with cx=1, cy=0
        q = gpuDistNormalizedFromNodePopulationToWall_a0(
            xNode, yNode, xNode-1, yNode, radius);
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 1)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 2)], fPostCol[idxPop(x, y, z, 1)], q);

            fPostStream[idxPop(x, y, z, 9)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 10)], fPostCol[idxPop(x, y, z, 9)], q);

            fPostStream[idxPop(x, y, z, 15)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 16)], fPostCol[idxPop(x, y, z, 15)], q);
        }
        else
        {
            fPostStream[idxPop(x, y, z, 1)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 2)], fPostCol[idxPop(x+1, y, z, 2)], q);

            fPostStream[idxPop(x, y, z, 9)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 10)], fPostCol[idxPop(x+1, y, zp1, 10)], q);

            fPostStream[idxPop(x, y, z, 15)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 16)], fPostCol[idxPop(x+1, y, zm1, 16)], q);
        }
    }
    if(unknownPops & UNKNOWN_POP_2)
    {
        q = gpuDistNormalizedFromNodePopulationToWall_ainf(
            xNode, yNode, xNode, yNode-1, radius);
        // Populations with cx=0, cy=1
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 3)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 4)], fPostCol[idxPop(x, y, z, 3)], q);

            fPostStream[idxPop(x, y, z, 11)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 12)], fPostCol[idxPop(x, y, z, 11)], q);

            fPostStream[idxPop(x, y, z, 17)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 18)], fPostCol[idxPop(x, y, z, 17)], q);
        }
        else
        {
            fPostStream[idxPop(x, y, z, 3)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 4)], fPostCol[idxPop(x, y+1, z, 4)], q);

            fPostStream[idxPop(x, y, z, 11)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 12)], fPostCol[idxPop(x, y+1, zp1, 12)], q);

            fPostStream[idxPop(x, y, z, 17)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 18)], fPostCol[idxPop(x, y+1, zm1, 18)], q);
        }
    }
    if(unknownPops & UNKNOWN_POP_3)
    {
        q = gpuDistNormalizedFromNodePopulationToWall_a0(
            xNode, yNode, xNode+1, yNode, radius);
        // Populations with cx=-1, cy=0
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 2)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 1)], fPostCol[idxPop(x, y, z, 2)], q);

            fPostStream[idxPop(x, y, z, 10)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 9)], fPostCol[idxPop(x, y, z, 10)], q);

            fPostStream[idxPop(x, y, z, 16)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 15)], fPostCol[idxPop(x, y, z, 16)], q);
        }
        else
        {
            fPostStream[idxPop(x, y, z, 2)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 1)], fPostCol[idxPop(x-1, y, z, 1)], q);

            fPostStream[idxPop(x, y, z, 10)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 9)], fPostCol[idxPop(x-1, y, zm1, 9)], q);

            fPostStream[idxPop(x, y, z, 16)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 15)], fPostCol[idxPop(x-1, y, zp1, 15)], q);
        }
    }
    if(unknownPops & UNKNOWN_POP_4)
    {
        q = gpuDistNormalizedFromNodePopulationToWall_ainf(
            xNode, yNode, xNode, yNode+1, radius);
        // Populations with cx=0, cy=-1
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 4)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 3)], fPostCol[idxPop(x, y, z, 4)], q);

            fPostStream[idxPop(x, y, z, 12)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 11)], fPostCol[idxPop(x, y, z, 12)], q);

            fPostStream[idxPop(x, y, z, 18)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 17)], fPostCol[idxPop(x, y, z, 18)], q);
        }
        else
        {
            fPostStream[idxPop(x, y, z, 4)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 3)], fPostCol[idxPop(x, y-1, z, 3)], q);

            fPostStream[idxPop(x, y, z, 12)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 11)], fPostCol[idxPop(x, y-1, zm1, 11)], q);

            fPostStream[idxPop(x, y, z, 18)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 17)], fPostCol[idxPop(x, y-1, zp1, 17)], q);
        }
    }
    if(unknownPops & UNKNOWN_POP_5)
    {
        q = gpuDistNormalizedFromNodePopulationToWall_ap1(
            xNode, yNode, xNode-1, yNode-1, radius);
        // Populations with cx=1, cy=1
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 7)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 8)], fPostCol[idxPop(x, y, z, 7)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 19)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 20)], fPostCol[idxPop(x, y, z, 19)], q);

            fPostStream[idxPop(x, y, z, 21)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 22)], fPostCol[idxPop(x, y, z, 21)], q);
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 7)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 8)], fPostCol[idxPop(x+1, y+1, z, 8)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 19)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 20)], fPostCol[idxPop(x+1, y+1, zp1, 20)], q);

            fPostStream[idxPop(x, y, z, 21)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 22)], fPostCol[idxPop(x+1, y+1, zm1, 22)], q);
            #endif
        }
    }
    if(unknownPops & UNKNOWN_POP_6)
    {
        q = gpuDistNormalizedFromNodePopulationToWall_am1(
            xNode, yNode, xNode+1, yNode-1, radius);
        // Populations with cx=-1, cy=1
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 14)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 13)], fPostCol[idxPop(x, y, z, 14)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 24)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 23)], fPostCol[idxPop(x, y, z, 24)], q);

            fPostStream[idxPop(x, y, z, 25)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 26)], fPostCol[idxPop(x, y, z, 25)], q);
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 14)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 13)], fPostCol[idxPop(x-1, y+1, z, 13)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 24)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 23)], fPostCol[idxPop(x-1, y+1, zm1, 23)], q);

            fPostStream[idxPop(x, y, z, 25)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 26)], fPostCol[idxPop(x-1, y+1, zp1, 26)], q);
            #endif
        }
    }
    if(unknownPops & UNKNOWN_POP_7)
    {
        q = gpuDistNormalizedFromNodePopulationToWall_ap1(
            xNode, yNode, xNode+1, yNode+1, radius);
        // Populations with cx=-1, cy=-1
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 8)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 7)], fPostCol[idxPop(x, y, z, 8)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 20)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 19)], fPostCol[idxPop(x, y, z, 20)], q);

            fPostStream[idxPop(x, y, z, 22)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 21)], fPostCol[idxPop(x, y, z, 22)], q);
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 8)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 7)], fPostCol[idxPop(x-1, y-1, z, 7)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 20)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 19)], fPostCol[idxPop(x-1, y-1, zm1, 19)], q);

            fPostStream[idxPop(x, y, z, 22)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 21)], fPostCol[idxPop(x-1, y-1, zp1, 21)], q);
            #endif
        }
    }
    if(unknownPops & UNKNOWN_POP_8)
    {
        q = gpuDistNormalizedFromNodePopulationToWall_am1(
            xNode, yNode, xNode-1, yNode+1, radius);
        // Populations with cx=1, cy=-1
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 13)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 14)], fPostCol[idxPop(x, y, z, 13)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 23)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 24)], fPostCol[idxPop(x, y, z, 23)], q);

            fPostStream[idxPop(x, y, z, 26)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 25)], fPostCol[idxPop(x, y, z, 26)], q);
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 13)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 14)], fPostCol[idxPop(x+1, y-1, z, 14)], q);
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 23)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 24)], fPostCol[idxPop(x+1, y-1, zp1, 24)], q);

            fPostStream[idxPop(x, y, z, 26)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 25)], fPostCol[idxPop(x+1, y-1, zm1, 25)], q);
            #endif
        }
    }
}


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_a0(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R)
{
    dfloat x, y, dist;
    y = y2;
    x = sqrt(R*R-y*y);
    if(x2 < 0)
        x = -x;
    dist = distPoints2D(x, y, x1, y1);
    return dist;
}


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_ainf(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R)
{
    dfloat x, y, dist;
    x = x2;
    y = sqrt(R*R-x*x);
    if(y2 < 0)
        y = -y;
    dist = distPoints2D(x, y, x1, y1);
    return dist;
}


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_ap1(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R)
{
    dfloat x, y, b, delta_r, dist;
    b = y1-x1;
    delta_r = sqrt(2*R*R-b*b);
    x = (-b+delta_r)*0.5;
    if((x1 > x2 && (x > x1 || x < x2)) ||
        (x1 < x2 && (x < x1 || x > x2)))
        x = (-b-delta_r)*0.5;
    y = x+b;
    dist = distPoints2D(x, y, x1, y1);
    return dist/SQRT_2;
}


__device__
dfloat gpuDistNormalizedFromNodePopulationToWall_am1(
    dfloat x1,
    dfloat y1,
    dfloat x2,
    dfloat y2,
    dfloat R)
{
    dfloat x, y, b, delta_r, dist;
    b = y1+x1;
    delta_r = sqrt(2*R*R-b*b);
    x = (b+delta_r)*0.5;
    if((x1 > x2 && (x > x1 || x < x2)) ||
        (x1 < x2 && (x < x1 || x > x2)))
        x = (b-delta_r)*0.5;
    y = -x+b;
    dist = distPoints2D(x, y, x1, y1);
    return dist/SQRT_2;
}