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

#include "boundaryConditionsHandler.h"


__device__
void gpuBoundaryConditions(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z)
{
    /*
    -> BC_SCHEME
        -> DIRECTION
            -> GEOMETRY
    */
    switch(gpuNT->getSchemeBC())
    {
    case BC_SCHEME_BOUNCE_BACK:
        gpuSchBounceBack(gpuNT, fPostStream, fPostCol, x, y, z);
        break;
    case BC_SCHEME_VEL_BOUNCE_BACK:
        gpuSchVelBounceBack(gpuNT, fPostStream, fPostCol, x, y, z);
        break;  
    case BC_SCHEME_VEL_ZOUHE:
        gpuSchVelZouHe(gpuNT, fPostStream, fPostCol, x, y, z);
        break;
    case BC_SCHEME_PRES_ZOUHE:
        gpuSchPresZouHe(gpuNT, fPostStream, fPostCol, x, y, z);
        break;
    case BC_SCHEME_FREE_SLIP:
        gpuSchFreeSlip(gpuNT, fPostStream, fPostCol, x, y, z);
        break;
    case BC_SCHEME_SPECIAL:
        gpuSchSpecial(gpuNT, fPostStream, fPostCol, x, y, z);
        break;
    default:
        break;
    }
}


__device__
void gpuSchFreeSlip(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z)
{
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCFreeSlipN(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH:
        gpuBCFreeSlipS(fPostStream, fPostCol, x, y, z);
        break;

    case WEST:
        gpuBCFreeSlipW(fPostStream, fPostCol, x, y, z);
        break;

    case EAST:
        gpuBCFreeSlipE(fPostStream, fPostCol, x, y, z);
        break;

    case FRONT:
        gpuBCFreeSlipF(fPostStream, fPostCol, x, y, z);
        break;

    case BACK:
        gpuBCFreeSlipB(fPostStream, fPostCol, x, y, z);
        break;
    default:
        break;
    }
}


__device__
void gpuSchBounceBack(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z)
{
    switch(gpuNT->getDirection())
    {
    case NORTH:
        gpuBCBounceBackN(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH:
        gpuBCBounceBackS(fPostStream, fPostCol, x, y, z);
        break;

    case WEST:
        gpuBCBounceBackW(fPostStream, fPostCol, x, y, z);
        break;

    case EAST:
        gpuBCBounceBackE(fPostStream, fPostCol, x, y, z);
        break;

    case FRONT:
        gpuBCBounceBackF(fPostStream, fPostCol, x, y, z);
        break;

    case BACK:
        gpuBCBounceBackB(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_WEST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNW(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_EAST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNE(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNF(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNB(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_WEST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSW(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_EAST:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSE(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSF(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSB(fPostStream, fPostCol, x, y, z);
        break;

    case WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackWF(fPostStream, fPostCol, x, y, z);
        break;

    case WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackWB(fPostStream, fPostCol, x, y, z);
        break;

    case EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackEF(fPostStream, fPostCol, x, y, z);
        break;

    case EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackEB(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNWF(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNWB(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNEF(fPostStream, fPostCol, x, y, z);
        break;

    case NORTH_EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackNEB(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_WEST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSWF(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_WEST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSWB(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_EAST_FRONT:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSEF(fPostStream, fPostCol, x, y, z);
        break;

    case SOUTH_EAST_BACK:
        if(gpuNT->getGeometry() == CONCAVE)
            gpuBCBounceBackSEB(fPostStream, fPostCol, x, y, z);
        break;

    default:
        break;
    }
}


__device__
void gpuSchVelBounceBack(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCVelBounceBackN(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case SOUTH:
        gpuBCVelBounceBackS(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case WEST:
        gpuBCVelBounceBackW(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case EAST:
        gpuBCVelBounceBackE(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case FRONT:
        gpuBCVelBounceBackF(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case BACK:
        gpuBCVelBounceBackB(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()], uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    default:
        break;
    }
#endif
}


__device__
void gpuSchPresZouHe(NodeTypeMap* gpuNT, 
    dfloat * f, 
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCPresZouHeN(fPostStream, fPostCol, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case SOUTH:
        gpuBCPresZouHeS(fPostStream, fPostCol, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case WEST:
        gpuBCPresZouHeW(fPostStream, fPostCol, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case EAST:
        gpuBCPresZouHeE(fPostStream, fPostCol, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case FRONT:
        gpuBCPresZouHeF(fPostStream, fPostCol, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;

    case BACK:
        gpuBCPresZouHeB(fPostStream, fPostCol, x, y, z, rhoBC[gpuNT->getRhoIdx()]);
        break;
    default:
        break;
    }
#endif
}


__device__
void gpuSchVelZouHe(NodeTypeMap* gpuNT, 
    dfloat* fPostStream,
    dfloat* fPostCol,
    const short unsigned int x, 
    const short unsigned int y, 
    const short unsigned int z)
{
#ifdef D3Q19 // support only for D3Q19
    switch (gpuNT->getDirection())
    {
    case NORTH:
        gpuBCVelZouHeN(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()], 
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case SOUTH:
        gpuBCVelZouHeS(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case WEST:
        gpuBCVelZouHeW(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case EAST:
        gpuBCVelZouHeE(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case FRONT:
        gpuBCVelZouHeF(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;

    case BACK:
        gpuBCVelZouHeB(fPostStream, fPostCol, x, y, z, uxBC[gpuNT->getUxIdx()],
            uyBC[gpuNT->getUyIdx()], uzBC[gpuNT->getUzIdx()]);
        break;
    default:
        break;
    }
#endif
}
