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

#include "treatData.h"


void treatData(MacrProc* processing, int step)
{
    /* DATA TREATMENT EXAMPLE */
    dfloat denRes = 0.0, numRes = 0.0; // denominator and numerator for residual
    Macroscopics* macrCurr = processing->macrCurr; 
    Macroscopics* macrOld = processing->macrOld; 

    processing->avgRho = 0;

    for(int z = 0; z < NZ_TOTAL; z++)
    {
        for(int y = 0; y < NY; y++)
        {
            for(int x = 0; x < NX; x++)
            {
                size_t idx = idxScalar(x, y, z);
                /* ------- Residual calculation ------- */
                const dfloat diff_ux = macrCurr->u.x[idx] - macrOld->u.x[idx];
                const dfloat diff_uy = macrCurr->u.y[idx] - macrOld->u.y[idx];
                const dfloat diff_uz = macrCurr->u.z[idx] - macrOld->u.z[idx];

                numRes += std::sqrt(diff_ux * diff_ux + diff_uy * diff_uy + diff_uz * diff_uz);
                denRes += std::sqrt(macrCurr->u.x[idx] * macrCurr->u.x[idx]
                    + macrCurr->u.y[idx] * macrCurr->u.y[idx]
                    + macrCurr->u.z[idx] * macrCurr->u.z[idx]);
                /* ------------------------------------ */

                /* ------- Avg. rho calculation ------- */
                processing->avgRho += macrCurr->rho[idx];
                // printf("%d %d %d %f\n", x, y, z, macrCurr->rho[idx]);
                /* ------------------------------------ */
                
                /* ----- Avg. Uz plan calculation ----- */
                processing->avgUzPlanXZ[y] += macrCurr->u.z[idx];
                /* ------------------------------------ */
            }
        }
    }

    /* ------- Residual calculation ------- */
    if(denRes != 0)
        processing->residual = numRes/denRes;
    else
        processing->residual = 1;
    /* ------------------------------------ */

    /* ------- Avg. rho calculation ------- */
    processing->avgRho /= TOTAL_NUMBER_LBM_NODES;
    /* ------------------------------------ */

    /* ----- Avg. Uz plan calculation ----- */
    for(int y = 0; y < NY; y++)
        processing->avgUzPlanXZ[y] /= (NX*NZ_TOTAL);
    /* ------------------------------------ */


    
    /*
    Macroscopics* macrCurr = processing->macrCurr; 
    dfloat eta_co = visc*(1.0 + del*Bn);
    dfloat omg_co = 1.0/(0.5 + 3.0*eta_co);

    // rho avg
    dfloat rho_avg_inst = 0.0;
    for(int i = 0; i < idxScalar(NX-1, NY-1, NZ-1) + 1; i++)
        rho_avg_inst += macrCurr->rho[i];
    rho_avg_inst /= NX*NY*NZ;

    int r = 0;

    const dfloat R = NY/2.0-1.5;
    dfloat xCenter = (NX/2.0)-0.5;
    dfloat yCenter = (NY/2.0)-0.5;
    
    int theta_step = 3;
    dfloat theta = 0;


    dfloat  ux[del][120],uy[del][120],uz[del][120];
    dfloat ur[del][120], ut[del][120];
    dfloat p_m[del],ux_m[del],uy_m[del],uz_m[del];
    dfloat ur_m[del], ut_m[del];
    dfloat p[del][120],loc_p;

    int cell_count[del][120];
    int cell_count_m[del];

    for (int rr = 0; rr < del;rr++){
        for(int tt = 0; tt< 120;tt++){
             p[rr][tt] = 0;
            ux[rr][tt] = 0;
            uy[rr][tt] = 0;
            uz[rr][tt] = 0;
            cell_count[rr][tt] = 0;

            ur[rr][tt] = 0;
            ut[rr][tt] = 0;
        }
        p_m[rr] = 0;
        ux_m[rr] = 0;
        uy_m[rr] = 0;
        uz_m[rr] = 0;
        cell_count_m[rr] = 0;

        ur_m[rr] = 0;
        ut_m[rr] = 0;
    }


    for(int x = 0; x < NX; x++){
        for(int y = 0; y < NY; y++){
            for(int z = 0; z < NZ; z++){
                size_t idx = idxScalar(x, y, z);
                r = (int)ceil(sqrt((dfloat)((x-xCenter)*(x-xCenter)+(y-yCenter)*(y-yCenter))));
                theta = atan2((y-yCenter),(x-xCenter))*180/M_PI + 180;

                //check if is outside of the duct
                if (r <= 0 || r >= R)
                    continue;

                loc_p = (1.0/3.0)*macrCurr->rho[idx] - (1.0/3.0)*rho_avg_inst;

                 ux[r][(int)(floor(theta/3.0))] += macrCurr->u.x[idx];
                 uy[r][(int)(floor(theta/3.0))] += macrCurr->u.y[idx];
                 uz[r][(int)(floor(theta/3.0))] += macrCurr->u.z[idx];

                  p[r][(int)(floor(theta/3.0))] += loc_p;

                cell_count[r][(int)(floor(theta/3.0))]++;


                ur[r][(int)(floor(theta/3.0))] +=  macrCurr->u.x[idx]*cos(M_PI*theta/180) + macrCurr->u.y[idx]*sin(M_PI*theta/180);
                ut[r][(int)(floor(theta/3.0))] += -macrCurr->u.x[idx]*sin(M_PI*theta/180) + macrCurr->u.y[idx]*cos(M_PI*theta/180);

            }
        }
    }

    //calculate the mean value as function of r in xyz coordinates
    for(int rr = 0; rr < del; rr++){
        for (int tt = 0; tt < 120;tt++){
              p_m[rr] +=   p[rr][tt];
             ux_m[rr] +=  ux[rr][tt];
             uy_m[rr] +=  uy[rr][tt];
             uz_m[rr] +=  uz[rr][tt];
             cell_count_m[rr] += cell_count[rr][tt];

             ur_m[rr] += ur[rr][tt];
             ut_m[rr] += ut[rr][tt];

        }
        if (cell_count_m[rr] == 0)
            continue;

         p_m[rr] /= (dfloat)cell_count_m[rr];
        ux_m[rr] /= (dfloat)cell_count_m[rr];
        uy_m[rr] /= (dfloat)cell_count_m[rr];
        uz_m[rr] /= (dfloat)cell_count_m[rr];

        ur_m[rr] /= (dfloat)cell_count_m[rr];
        ut_m[rr] /= (dfloat)cell_count_m[rr];
    }


    // rho
    printf("\n%d 00 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,p_m[y]);

    // ux
    printf("\n%d 01 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,ux_m[y]);

    // uy
    printf("\n%d 02 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,uy_m[y]);

    // uz
    printf("\n%d 03 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,uz_m[y]);
    fflush(stdout);


    // ur
    printf("\n%d 04 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,ut_m[y]);
    fflush(stdout);

    // ut
    printf("\n%d 05 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,ut_m[y]);
    fflush(stdout);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    dfloat uxx_f[del][120],uyy_f[del][120],uxy_f[del][120],uzz_f[del][120];
    dfloat ur_rms[del][120],ut_rms[del][120],uz_rms[del][120];
    dfloat ur_rms_m[del],ut_rms_m[del],uz_rms_m[del];
    dfloat local_ux,local_uy,local_uz;

    //reset local
    for (int rr = 0; rr < del;rr++){
        for(int tt = 0; tt< 120;tt++){
            uxx_f[rr][tt] = 0;
            uyy_f[rr][tt] = 0;
            uxy_f[rr][tt] = 0;
            uzz_f[rr][tt] = 0;

            ur_rms[rr][tt]  = 0;
            ut_rms[rr][tt]  = 0;
            uz_rms[rr][tt]  = 0;
        }
        ur_rms_m[rr] = 0;
        ut_rms_m[rr] = 0;
        uz_rms_m[rr] = 0;
    }



    for(int x = 0; x < NX; x++){
        for(int y = 0; y < NY; y++){
            for(int z = 0; z < NZ; z++){
                size_t idx = idxScalar(x, y, z);
                r = (int)ceil(sqrt((dfloat)((x-xCenter)*(x-xCenter)+(y-yCenter)*(y-yCenter))));
                theta = atan2((y-yCenter),(x-xCenter))*180/M_PI + 180;

                //check if is outside of the duct
                if (r <= 0 || r >= R)
                    continue;

                local_ux =  macrCurr->u.x[idx] - ux_m[r];
                local_uy =  macrCurr->u.y[idx] - uy_m[r];
                local_uz =  macrCurr->u.z[idx] - uz_m[r];

                uxx_f[r][(int)(floor(theta/3.0))] += local_ux*local_ux;
                uyy_f[r][(int)(floor(theta/3.0))] += local_uy*local_uy;
                uxy_f[r][(int)(floor(theta/3.0))] += local_ux*local_uy;
                uzz_f[r][(int)(floor(theta/3.0))] += local_uz*local_uz;

            }
        }
    }


    //calculate the mean value of the fluctuation for each r and theta
    for (int rr = 0; rr < del;rr++){
        for(int tt = 0; tt< 120;tt++){

            if (cell_count[rr][tt] == 0)
                continue;

            uxx_f[rr][tt] /= (dfloat)cell_count[rr][tt];
            uyy_f[rr][tt] /= (dfloat)cell_count[rr][tt];
            uxy_f[rr][tt] /= (dfloat)cell_count[rr][tt];
            uzz_f[rr][tt] /= (dfloat)cell_count[rr][tt];
        }
    }


    //transform to cyclindric and sum in theta
    for(int rr = 0; rr < del; rr++){
        for (int tt = 0; tt < 120;tt++){
            ur_rms[rr][tt] += sqrt(uxx_f[rr][tt] * cos(M_PI*tt/180)*cos(M_PI*tt/180) 
                                 + uyy_f[rr][tt] * sin(M_PI*tt/180)*sin(M_PI*tt/180)
                                 + uxy_f[rr][tt] *  sin(2.0*M_PI*tt/180));
            ut_rms[rr][tt] += sqrt(uxx_f[rr][tt] * sin(M_PI*tt/180)*sin(M_PI*tt/180) 
                                 + uyy_f[rr][tt] * cos(M_PI*tt/180)*cos(M_PI*tt/180) 
                                 - uxy_f[rr][tt] * sin(2.0*M_PI*tt/180));
            uz_rms[rr][tt] +=sqrt(uzz_f[rr][tt]);
        }
    }

    //average over theta
    for(int rr = 0; rr < del; rr++){
        for (int tt = 0; tt < 120;tt++){
            ur_rms_m[rr] += ur_rms[rr][tt] * cell_count[rr][tt];
            ut_rms_m[rr] += ut_rms[rr][tt] * cell_count[rr][tt];
            uz_rms_m[rr] += uz_rms[rr][tt] * cell_count[rr][tt];
        }
        if (cell_count_m[rr] == 0)
            continue;

        ur_rms_m[rr] /= (dfloat)cell_count_m[rr];
        ut_rms_m[rr] /= (dfloat)cell_count_m[rr];
        uz_rms_m[rr] /= (dfloat)cell_count_m[rr];
    }

    // ur_rms
    printf("\n%d 11 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,ur_rms_m[y]);

    // ut_rms
    printf("\n%d 22 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,ut_rms_m[y]);

    // uz_rms
    printf("\n%d 33 " ,step);
    for(int y = 0; y < del; y++)
        printf("%e " ,uz_rms_m[y]);

    fflush(stdout);*/

    /*
    dfloat ux[NY],uy[NY],uz[NY],p[NY];
    dfloat uxx[NY],uxy[NY],uxz[NY],uyy[NY],uyz[NY],uzz[NY];
    //volume averages
    for(int y = 0; y < NY; y++)
    {
        ux[y] = 0;
        uy[y] = 0;
        uz[y] = 0;
        p[y] = 0;


        uxx[y] = 0;
        uxy[y] = 0;
        uxz[y] = 0;
        uyy[y] = 0;
        uyz[y] = 0;
        uzz[y] = 0;
        
        

        for(int z = 0; z < NZ; z++)
        {
            for(int x = 0; x < NX; x++)
            {
                size_t idx = idxScalar(x, y, z);
                dfloat loc_p = (1.0/3.0)*macrCurr->rho[idx] - (1.0/3.0)*rho_avg_inst;

                ux[y] += macrCurr->u.x[idx];
                uy[y] += macrCurr->u.y[idx];
                uz[y] += macrCurr->u.z[idx];
                p[y] += loc_p;


                uxx[y] += macrCurr->u.x[idx]*macrCurr->u.x[idx];
                uxy[y] += macrCurr->u.x[idx]*macrCurr->u.y[idx];
                uxz[y] += macrCurr->u.x[idx]*macrCurr->u.z[idx];
                uyy[y] += macrCurr->u.y[idx]*macrCurr->u.y[idx];
                uyz[y] += macrCurr->u.y[idx]*macrCurr->u.z[idx];
                uzz[y] += macrCurr->u.z[idx]*macrCurr->u.z[idx];
                
            }
        }
        ux[y] /= NX*NZ;
        uy[y] /= NX*NZ;
        uz[y] /= NX*NZ;
        p[y] /= NX*NZ;


        uxx[y] /= NX*NZ;
        uxy[y] /= NX*NZ;
        uxz[y] /= NX*NZ;
        uyy[y] /= NX*NZ;
        uyz[y] /= NX*NZ;
        uzz[y] /= NX*NZ;

    }

    // rho
    printf("%d 00 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,p[y]);

    // ux
    printf("\n%d 01 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,ux[y]);

    // uy
    printf("\n%d 02 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uy[y]);

    // uz
    printf("\n%d 03 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uz[y]);






    // uxx
    printf("\n%d 11 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uxx[y]);
    // uxy
    printf("\n%d 12 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uxy[y]);
    // uxz
    printf("\n%d 13 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uxz[y]);

    // uyy
    printf("\n%d 22 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uyy[y]);
    // uyz
    printf("\n%d 23 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uyz[y]);

    // uzz
    printf("\n%d 33 " ,step);
    for(int y = 0; y < NY; y++)
        printf("%e " ,uzz[y]);

fflush(stdout);
*/


}


bool stopSim(MacrProc* processing)
{
    /* SIMULATIONS STOP CONDITIONS EXAMPLE */
    if(processing->residual < RESID_MAX)
        return true;
    if(processing->avgRho < 0)
        return true;
    return false;
}


void printTreatData(MacrProc* processing)
{
    /* PRINT TREATED DATA EXAMPLE */
    printf("\n--------------------------------- TREATED DATA ---------------------------------\n");
    printf("                   Step: %d\n", *(processing->step));
    printf("               Residual: %.4e\n", processing->residual);
    printf("           Avg. density: %.4e\n", processing->avgRho);
    printf("       Avg. Uz (y=NY/2): %.4e\n", processing->avgUzPlanXZ[NY/2]);
    // +MACR_BORDER_NODES because of the ghost nodes
    printf("ux(x=0.5, y=0.5, z=0.5): %.4e\n", processing->macrCurr->u.x[idxScalar(NX/2, NY/2, NZ/2)]);
    printf("--------------------------------------------------------------------------------\n");
}


void saveTreatData(MacrProc* processing)
{
    /* SAVE TO CSV EXAMPLE */
    std::string strFileAvgUz;
    strFileAvgUz = getVarFilename("avgUz", *(processing->step), ".csv");
    
    FILE* fileAvgUz = nullptr;
    fileAvgUz = fopen(strFileAvgUz.c_str(), "w");
    if(fileAvgUz != nullptr)
    {
        // write header
        fprintf(fileAvgUz, "y\tavg uz\n");
        for(int y = 0; y < NY; y++)
        {
            fprintf(fileAvgUz, "%d\t%.6e\n", y, processing->avgUzPlanXZ[y]);
        }
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strFileAvgUz.c_str());
    }
}