#include "interpolatedBounceBack.h"


#ifdef BC_SCHEME_INTERP_BOUNCE_BACK

__device__ 
void gpuBCInterpolatedBounceBack(const unsigned char unknownPops,
    const bool is_inside,
    dfloat* fPostStream, 
    dfloat* fPostCol, 
    const short unsigned int x, 
    const short unsigned int y,
    const short unsigned int z,
    const int n_gpu)
{
    dfloat q, r, R, xNode, yNode, radius;
    const unsigned short int zp1 = (z + 1) % NZ;
    const unsigned short int zm1 = (NZ + z - 1) % NZ;
    const unsigned int zDomain = z + NZ*n_gpu;
    // THIS RADIUS MUST BE THE SAME AS IN THE BOUNDARY CONDITION BUILDER
    R = OUTER_RADIUS;
    // q = R - distPoints2D(x+0.5, y+0.5, NX/2.0, NY/2.0);

    //wall velocity
    dfloat w_i, uz_i;
    dfloat w_o, uz_o;

    w_o = OUTER_ROTATION*((dfloat)zDomain/(dfloat)NZ_TOTAL);
    uz_o = OUTER_VELOCITY;

    #ifdef INTERNAL_DUCT_BC 
        r = INNER_RADIUS;
        w_i = INNER_ROTATION;//0.0;
        uz_i = INNER_VELOCITY;
    #endif


    // Dislocate coordinates to get x^2+y^2=R^2
    xNode = x - (NX-1)/2.0;
    yNode = y - (NY-1)/2.0;
    
    dfloat rr =  sqrt(xNode*xNode+yNode*yNode);
    dfloat c = xNode / (rr);
    dfloat s = yNode / (rr);

    dfloat ux,uy,uz;

    if(is_inside){
        #ifdef INTERNAL_DUCT_BC
            radius = r;
            ux = - w_i * r * s;
            uy =   w_i * r * c;
            uz = uz_i;
        #endif
    }else{
        radius = R;
        ux = - w_o * R * s;
        uy =   w_o * R * c;
        uz = uz_o;
    }


    dfloat rho_w=RHO_0;
    if (zDomain ==  0 || z == (NZ_TOTAL-1)){
        rho_w = fPostCol[idxPop(x, y, z, 0)] + fPostCol[idxPop(x, y, z, 1)] + fPostCol[idxPop(x, y, z, 2)] +
            fPostCol[idxPop(x, y, z, 3)] + fPostCol[idxPop(x, y, z, 4)] + fPostCol[idxPop(x, y, z, 5)] + fPostCol[idxPop(x, y, z, 6)] +
            fPostCol[idxPop(x, y, z, 7)] + fPostCol[idxPop(x, y, z, 8)] + fPostCol[idxPop(x, y, z, 9)] + fPostCol[idxPop(x, y, z, 10)] +
            fPostCol[idxPop(x, y, z, 11)] + fPostCol[idxPop(x, y, z, 12)] + fPostCol[idxPop(x, y, z, 13)] + fPostCol[idxPop(x, y, z, 14)] +
            fPostCol[idxPop(x, y, z, 15)] + fPostCol[idxPop(x, y, z, 16)] + fPostCol[idxPop(x, y, z, 17)] + fPostCol[idxPop(x, y, z, 18)];
    }

    if(unknownPops & UNKNOWN_POP_1)
    {
        // Populations with cx=1, cy=0
        q = gpuDistNormalizedFromNodePopulationToWall_a0(
            xNode, yNode, xNode-1, yNode, radius);
        if(q > 0.5)
        {
            fPostStream[idxPop(x, y, z, 1)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 2)], fPostCol[idxPop(x, y, z, 1)], q, W1*ux);

            fPostStream[idxPop(x, y, z, 9)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 10)], fPostCol[idxPop(x, y, z, 9)], q, W2*(ux+uz));

            fPostStream[idxPop(x, y, z, 15)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 16)], fPostCol[idxPop(x, y, z, 15)], q, W2*(ux-uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)] - 6 * rho_w*W2*(-0.0 - ux);
            }
            if (zDomain == (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)] - 6 * rho_w*W2*(0.0 - ux);
            }
        }
        else
        {
            fPostStream[idxPop(x, y, z, 1)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 2)], fPostCol[idxPop(x+1, y, z, 2)], q, W1*ux);

            fPostStream[idxPop(x, y, z, 9)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 10)], fPostCol[idxPop(x+1, y, zp1, 10)], q, W2*(ux+uz));

            fPostStream[idxPop(x, y, z, 15)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 16)], fPostCol[idxPop(x+1, y, zm1, 16)], q, W2*(ux-uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 9)] = fPostCol[idxPop(x, y, z, 10)] - 6 * rho_w*W2*(-0.0 - ux);
            }
            if (zDomain == (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 15)] = fPostCol[idxPop(x, y, z, 16)] - 6 * rho_w*W2*(0.0 - ux);
            }                
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
                fPostCol[idxPop(x, y, z, 4)], fPostCol[idxPop(x, y, z, 3)], q, W1*uy);

            fPostStream[idxPop(x, y, z, 11)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 12)], fPostCol[idxPop(x, y, z, 11)], q, W2*(uy+uz));

            fPostStream[idxPop(x, y, z, 17)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 18)], fPostCol[idxPop(x, y, z, 17)], q, W2*(uy-uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)] - 6 * rho_w*W2*(-0.0 - uy);
            }
            if (zDomain == (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)] - 6 * rho_w*W2*(0.0 - uy);
            }
        }
        else
        {
            fPostStream[idxPop(x, y, z, 3)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 4)], fPostCol[idxPop(x, y+1, z, 4)], q, W1*uy);

            fPostStream[idxPop(x, y, z, 11)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 12)], fPostCol[idxPop(x, y+1, zp1, 12)], q, W2*(uy+uz));

            fPostStream[idxPop(x, y, z, 17)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 18)], fPostCol[idxPop(x, y+1, zm1, 18)], q, W2*(uy-uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 11)] = fPostCol[idxPop(x, y, z, 12)] - 6 * rho_w*W2*(-0.0 - uy);
            }
            if (zDomain == (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 17)] = fPostCol[idxPop(x, y, z, 18)] - 6 * rho_w*W2*(0.0 - uy);
            }
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
                fPostCol[idxPop(x, y, z, 1)], fPostCol[idxPop(x, y, z, 2)], q, -W1*ux);

            fPostStream[idxPop(x, y, z, 10)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 9)], fPostCol[idxPop(x, y, z, 10)], q, W2*(-ux-uz));

            fPostStream[idxPop(x, y, z, 16)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 15)], fPostCol[idxPop(x, y, z, 16)], q, W2*(-ux+uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)] - 6 * rho_w*W2*(-0.0 + ux);
            }
            if (zDomain == (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)] - 6 * rho_w*W2*(0.0 + ux);
            }
        }
        else
        {
            fPostStream[idxPop(x, y, z, 2)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 1)], fPostCol[idxPop(x-1, y, z, 1)], q, -W1*ux);

            fPostStream[idxPop(x, y, z, 10)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 9)], fPostCol[idxPop(x-1, y, zm1, 9)], q, W2*(-ux-uz));

            fPostStream[idxPop(x, y, z, 16)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 15)], fPostCol[idxPop(x-1, y, zp1, 15)], q, W2*(-ux+uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 16)] = fPostCol[idxPop(x, y, z, 15)] - 6 * rho_w*W2*(-0.0 + ux);
            }
            if (zDomain== (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 10)] = fPostCol[idxPop(x, y, z, 9)] - 6 * rho_w*W2*(0.0 + ux);
            }
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
                fPostCol[idxPop(x, y, z, 3)], fPostCol[idxPop(x, y, z, 4)], q, -W1*uy);

            fPostStream[idxPop(x, y, z, 12)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 11)], fPostCol[idxPop(x, y, z, 12)], q, W2*(-uy-uz));

            fPostStream[idxPop(x, y, z, 18)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 17)], fPostCol[idxPop(x, y, z, 18)], q, W2*(-uy+uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)] - 6 * rho_w*W2*(-0.0 + uy);
            }
            if (zDomain == (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)] - 6 * rho_w*W2*(0.0 + uy);
            }
        }
        else
        {
            fPostStream[idxPop(x, y, z, 4)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 3)], fPostCol[idxPop(x, y-1, z, 3)], q, -W1*uy);

            fPostStream[idxPop(x, y, z, 12)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 11)], fPostCol[idxPop(x, y-1, zm1, 11)], q, W2*(-uy-uz));

            fPostStream[idxPop(x, y, z, 18)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 17)], fPostCol[idxPop(x, y-1, zp1, 17)], q, W2*(-uy+uz));
            if (zDomain ==  0 && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 18)] = fPostCol[idxPop(x, y, z, 17)] - 6 * rho_w*W2*(-0.0 + uy);
            }
            if (zDomain == (NZ_TOTAL-1) && BC_RHEOMETER){
                fPostStream[idxPop(x, y, z, 12)] = fPostCol[idxPop(x, y, z, 11)] - 6 * rho_w*W2*(0.0 + uy);
            }
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
                fPostCol[idxPop(x, y, z, 8)], fPostCol[idxPop(x, y, z, 7)], q, W2*(ux+uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 19)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 20)], fPostCol[idxPop(x, y, z, 19)], q, W3*(ux+uy+uz));

            fPostStream[idxPop(x, y, z, 21)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 22)], fPostCol[idxPop(x, y, z, 21)], q, W3*(ux+uy-uz));
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 7)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 8)], fPostCol[idxPop(x+1, y+1, z, 8)], q, W2*(ux+uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 19)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 20)], fPostCol[idxPop(x+1, y+1, zp1, 20)], q, W3*(ux+uy+uz));

            fPostStream[idxPop(x, y, z, 21)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 22)], fPostCol[idxPop(x+1, y+1, zm1, 22)], q, W3*(ux+uy-uz));
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
                fPostCol[idxPop(x, y, z, 13)], fPostCol[idxPop(x, y, z, 14)], q, W2*(-ux+uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 24)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 23)], fPostCol[idxPop(x, y, z, 24)], q, W3*(-ux+uy-uz));

            fPostStream[idxPop(x, y, z, 25)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 26)], fPostCol[idxPop(x, y, z, 25)], q, W3*(-ux+uy+uz));
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 14)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 13)], fPostCol[idxPop(x-1, y+1, z, 13)], q, W2*(-ux+uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 24)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 23)], fPostCol[idxPop(x-1, y+1, zm1, 23)], q, W3*(-ux+uy-uz));

            fPostStream[idxPop(x, y, z, 25)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 26)], fPostCol[idxPop(x-1, y+1, zp1, 26)], q, W3*(-ux+uy+uz));
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
                fPostCol[idxPop(x, y, z, 7)], fPostCol[idxPop(x, y, z, 8)], q, W2*(-ux-uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 20)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 19)], fPostCol[idxPop(x, y, z, 20)], q, W3*(-ux-uy-uz));

            fPostStream[idxPop(x, y, z, 22)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 21)], fPostCol[idxPop(x, y, z, 22)], q,  W3*(-ux-uy+uz));
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 8)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 7)], fPostCol[idxPop(x-1, y-1, z, 7)], q, W2*(-ux-uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 20)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 19)], fPostCol[idxPop(x-1, y-1, zm1, 19)], q,  W3*(-ux-uy-uz));

            fPostStream[idxPop(x, y, z, 22)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 21)], fPostCol[idxPop(x-1, y-1, zp1, 21)], q,  W3*(-ux-uy+uz));
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
                fPostCol[idxPop(x, y, z, 14)], fPostCol[idxPop(x, y, z, 13)], q, W2*(ux-uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 23)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 24)], fPostCol[idxPop(x, y, z, 23)], q,  W3*(ux-uy+uz));

            fPostStream[idxPop(x, y, z, 26)] = gpuInterpolatedBounceBackHigherQ(
                fPostCol[idxPop(x, y, z, 25)], fPostCol[idxPop(x, y, z, 26)], q, W3*(ux-uy-uz));
            #endif
        }
        else
        {
            fPostStream[idxPop(x, y, z, 13)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 14)], fPostCol[idxPop(x+1, y-1, z, 14)], q, W2*(ux-uy));
            #ifdef D3Q27
            fPostStream[idxPop(x, y, z, 23)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 24)], fPostCol[idxPop(x+1, y-1, zp1, 24)], q, W3*(ux-uy+uz));

            fPostStream[idxPop(x, y, z, 26)] = gpuInterpolatedBounceBackLowerQ(
                fPostCol[idxPop(x, y, z, 25)], fPostCol[idxPop(x+1, y-1, zm1, 25)], q, W3*(ux-uy-uz));
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

#endif