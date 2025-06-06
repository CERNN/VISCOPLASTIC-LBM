/*
*   @file inflowFreeSlip.cu
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Inflow and outflow in Z (back and front) and 
        free slip in Y (north and south)
*       N, S: free slip; B: inflow; F: outflow; W, E: periodic
*   @version 0.3.0
*   @date 19/01/2020
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

    if (y == 0 && zDomain == 0) // SB
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_FREE_SLIP);
        gpuMapBC[idxScalar(x, y, z)].setDirection(SOUTH);
    }
    else if (y == 0 && zDomain == (NZ_TOTAL-1)) // SF
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_FREE_SLIP);
        gpuMapBC[idxScalar(x, y, z)].setDirection(SOUTH);
    }
    else if (y == (NY - 1) && zDomain == 0) // NB
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_FREE_SLIP);
        gpuMapBC[idxScalar(x, y, z)].setDirection(NORTH);
    }
    else if (y == (NY - 1) && zDomain == (NZ_TOTAL-1)) // NF
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_FREE_SLIP);
        gpuMapBC[idxScalar(x, y, z)].setDirection(NORTH);
    }
    else if (y == 0) // S
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_FREE_SLIP);
        gpuMapBC[idxScalar(x, y, z)].setDirection(SOUTH);
    }
    else if (y == (NY - 1)) // N
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_FREE_SLIP);
        gpuMapBC[idxScalar(x, y, z)].setDirection(NORTH);
    }
    else if (zDomain == 0) // B
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_VEL_BOUNCE_BACK);
        gpuMapBC[idxScalar(x, y, z)].setDirection(BACK);
        gpuMapBC[idxScalar(x, y, z)].setUzIdx(1);
    }
    else if (zDomain == (NZ_TOTAL-1)) // F
    {
        gpuMapBC[idxScalar(x, y, z)].setSchemeBC(BC_SCHEME_VEL_BOUNCE_BACK);
        gpuMapBC[idxScalar(x, y, z)].setDirection(FRONT);
        gpuMapBC[idxScalar(x, y, z)].setUzIdx(1);
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
    switch(gpuNT->getDirection())
    {
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