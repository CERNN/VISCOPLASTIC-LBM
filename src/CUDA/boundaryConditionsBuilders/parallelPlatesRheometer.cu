/*
*   @file parallelPlatesRheometer.cu
*   @author Marco Ferrari Jr. (marcoferrari@alunos.utfpr.edu.br)
*   @brief Models a parallel plates rheometer geometry
*          uses interpoleted bounce back for the N/S/W/E walls
*          for front/back (z direciton) uses velocity bounce back with velocity based position
*   @version 0.3.0
*   @date 30/12/2022
*/

/*
*   LBM-CERNN
*   Copyright (C) 2018-2019 Waine Barbosa de Oliveira Junior
*
*   This program is free software; you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation; either version 2 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License along
*   with this program; if not, write to the Free Software Foundation, Inc.,
*   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*
*   Contact: cernn-ct@utfpr.edu.br and waine@alunos.utfpr.edu.br
*/

#include "boundaryConditionsBuilder.h"


__global__
void gpuBuildBoundaryConditions(NodeTypeMap* const gpuMapBC, int gpuNumber)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int z = threadIdx.z + blockDim.z * blockIdx.z;
    const unsigned int zDomain = z + NZ*gpuNumber;

    if(x >= NX || y >= NY || z >= NZ)
        return;

    gpuMapBC[idxScalar(x, y, z)].setIsUsed(true); //set all nodes fluid inicially and no bc
    gpuMapBC[idxScalar(x, y, z)].setSavePostCol(false); // set all nodes to not save post 
                                                    // collision population (just stream)
    gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_NULL);
    gpuMapBC[idxScalar(x, y, z)].setGeometry(CONCAVE);
    gpuMapBC[idxScalar(x, y, z)].setUxIdx(0); // manually assigned (index of ux=0)
    gpuMapBC[idxScalar(x, y, z)].setUyIdx(0); // manually assigned (index of uy=0)
    gpuMapBC[idxScalar(x, y, z)].setUzIdx(0); // manually assigned (index of uz=0)
    gpuMapBC[idxScalar(x, y, z)].setRhoIdx(0); // manually assigned (index of rho=RHO_0)

    // Cilinder values
    // THIS RADIUS MUST BE THE SAME AS IN 
    // "boundaryConditionsSchemes/interpolatedBounceBack.cu"
    dfloat R = OUTER_RADIUS;
    dfloat xCenter = DUCT_CENTER_X;
    dfloat yCenter = DUCT_CENTER_Y;

    // Node values
    dfloat xNode = x+0.5;
    dfloat yNode = y+0.5;

    dfloat distNode = distPoints2D(xNode, yNode, xCenter, yCenter);

    // if the point is out of the cilinder
    if(distNode > R ||zDomain == 0 || zDomain == NZ_TOTAL-1)
    {
        gpuMapBC[idxScalar(x, y, z)].setIsUsed(false);
        return;
    }
    else
    {   
        if(BC_RHEOMETER){
            if(zDomain == 1) // B
            {
                gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_SPECIAL);
                gpuMapBC[idxScalar(x, y, z)].setDirection(BACK);
            }
            //front side z= (NZ_TOTAL-1)
            else if(zDomain == (NZ_TOTAL-2)) // F
            {
                gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_SPECIAL);
                gpuMapBC[idxScalar(x, y, z)].setDirection(FRONT);
            }
        }
        // if the point is a boundary node (outside of the cylinder)
        if(distNode > (R-sqrt((float)2)))
        {
            gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_INTERP_BOUNCE_BACK);
            gpuMapBC[idxScalar(x, y, z)].setIsInsideNodeInterpoBB(false);
        }
        // if the point is next to a boundary node
        else if(distNode > (R-2*sqrt((float)2)))
        {
            gpuMapBC[idxScalar(x, y, z)].setSavePostCol(true);
        }
        else
            return;
    }

    // Process the adjacent coordinates (fluid or non fluid)
    
    // Directions in [x, y] (same as D2Q9, without the population 0)
    char dirs[8][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}, 
                       {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

    // char popUnknown = 0b0;
    for(char i = 0; i < 8; i++)
    {
        // Adjacent node coordinates (where the population comes from)
        dfloat xAdj = xNode - dirs[i][0];
        dfloat yAdj = yNode - dirs[i][1];
        // if the adjancent node is in boundaries
        if(xAdj < NX && xAdj > 0 && yAdj < NY && yAdj > 0)
        {
            dfloat distAdj = distPoints2D(xAdj, yAdj, xCenter, yCenter);
            if(distAdj > R)
            {
                // set population as unknown
                // popUnknown |= (0b1 << i);
                gpuMapBC[idxScalar(x, y, z)].setUnknowPopInterpBB(i);
            }
        }
        else
        {
            // set population as unknown
            gpuMapBC[idxScalar(x, y, z)].setUnknowPopInterpBB(i);
        }
    }

}


__device__
void gpuSchSpecial(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z)
{
   dfloat R = OUTER_RADIUS;

    dfloat w_f = OUTER_ROTATION;
    dfloat w_b = 0.0;



    // Dislocate coordinates to get x^2+y^2=R^2
    dfloat xNode = x - (NX-1)/2.0;
    dfloat yNode = y - (NY-1)/2.0;
    
    dfloat rr =  sqrt(xNode*xNode+yNode*yNode);
    dfloat c = xNode / (rr);
    dfloat s = yNode / (rr);

    dfloat ux_w,uy_w,uz_w;
    dfloat* f = fPostStream;
    dfloat rho_w;
    switch(gpuNT->getDirection())
    {
    case FRONT:
        ux_w = - w_f * rr * s;
        uy_w =   w_f * rr * c;
        uz_w = 0;
        // uses node's rho as the wall's rho
        rho_w = f[idxPop(x, y, z, 0)] + f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 2)] +
            f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 4)] + f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 6)] +
            f[idxPop(x, y, z, 7)] + f[idxPop(x, y, z, 8)] + f[idxPop(x, y, z, 9)] + f[idxPop(x, y, z, 10)] +
            f[idxPop(x, y, z, 11)] + f[idxPop(x, y, z, 12)] + f[idxPop(x, y, z, 13)] + f[idxPop(x, y, z, 14)] +
            f[idxPop(x, y, z, 15)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 17)] + f[idxPop(x, y, z, 18)];

            f[idxPop(x, y, z, 6)] = f[idxPop(x, y, z, 5)] - 6 * rho_w*W1*(uz_w);
            f[idxPop(x, y, z, 10)] = f[idxPop(x, y, z, 9)] - 6 * rho_w*W2*(uz_w + ux_w);
            f[idxPop(x, y, z, 12)] = f[idxPop(x, y, z, 11)] - 6 * rho_w*W2*(uz_w + uy_w);
            f[idxPop(x, y, z, 15)] = f[idxPop(x, y, z, 16)] - 6 * rho_w*W2*(uz_w - ux_w);
            f[idxPop(x, y, z, 17)] = f[idxPop(x, y, z, 18)] - 6 * rho_w*W2*(uz_w - uy_w);
            break;
    case BACK:
        ux_w = - w_b * rr * s;
        uy_w =   w_b * rr * c;
        uz_w = 0;
        // uses node's rho as the wall's rho
        rho_w = f[idxPop(x, y, z, 0)] + f[idxPop(x, y, z, 1)] + f[idxPop(x, y, z, 2)] +
            f[idxPop(x, y, z, 3)] + f[idxPop(x, y, z, 4)] + f[idxPop(x, y, z, 5)] + f[idxPop(x, y, z, 6)] +
            f[idxPop(x, y, z, 7)] + f[idxPop(x, y, z, 8)] + f[idxPop(x, y, z, 9)] + f[idxPop(x, y, z, 10)] +
            f[idxPop(x, y, z, 11)] + f[idxPop(x, y, z, 12)] + f[idxPop(x, y, z, 13)] + f[idxPop(x, y, z, 14)] +
            f[idxPop(x, y, z, 15)] + f[idxPop(x, y, z, 16)] + f[idxPop(x, y, z, 17)] + f[idxPop(x, y, z, 18)];

            f[idxPop(x, y, z, 5)] = f[idxPop(x, y, z, 6)] - 6 * rho_w*W1*(-uz_w);
            f[idxPop(x, y, z, 9)] = f[idxPop(x, y, z, 10)] - 6 * rho_w*W2*(-uz_w - ux_w);
            f[idxPop(x, y, z, 11)] = f[idxPop(x, y, z, 12)] - 6 * rho_w*W2*(-uz_w - uy_w);
            f[idxPop(x, y, z, 16)] = f[idxPop(x, y, z, 15)] - 6 * rho_w*W2*(-uz_w + ux_w);
            f[idxPop(x, y, z, 18)] = f[idxPop(x, y, z, 17)] - 6 * rho_w*W2*(-uz_w + uy_w);
            break;
    case NORTH_WEST:
        // SPECIAL TREATMENT FOR NW
        break;

    case NORTH_EAST:
        // SPECIAL TREATMENT FOR NE
        break;

    case NORTH_FRONT:
        // SPECIAL TREATMENT FOR NF
        break;

    case NORTH_BACK:
        // SPECIAL TREATMENT FOR NB
        break;

    case SOUTH_WEST:
        // SPECIAL TREATMENT FOR SW
        break;

    case SOUTH_EAST:
        // SPECIAL TREATMENT FOR SE
        break;

    case SOUTH_FRONT:
        // SPECIAL TREATMENT FOR SF
        break;

    case SOUTH_BACK:
        // SPECIAL TREATMENT FOR SB
        break;

    case WEST_FRONT:
        // SPECIAL TREATMENT FOR WF
        break;

    case WEST_BACK:
        // SPECIAL TREATMENT FOR WB
        break;

    case EAST_FRONT:
        // SPECIAL TREATMENT FOR EF
        break;

    case EAST_BACK:
        // SPECIAL TREATMENT FOR EB
        break;

    case NORTH_WEST_FRONT:
        // SPECIAL TREATMENT FOR NWF
        break;

    case NORTH_WEST_BACK:
        // SPECIAL TREATMENT FOR NWB
        break;

    case NORTH_EAST_FRONT:
        // SPECIAL TREATMENT FOR NEF
        break;

    case NORTH_EAST_BACK:
        // SPECIAL TREATMENT FOR NEB
        break;

    case SOUTH_WEST_FRONT:
        // SPECIAL TREATMENT FOR SWF
        break;

    case SOUTH_WEST_BACK:
        // SPECIAL TREATMENT FOR SWB
        break;

    case SOUTH_EAST_FRONT:
        // SPECIAL TREATMENT FOR SEF
        break;

    case SOUTH_EAST_BACK:
        // SPECIAL TREATMENT FOR SEB
        break;
    
    default:
        break;
    }
}