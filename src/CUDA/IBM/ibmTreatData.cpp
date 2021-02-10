#include "ibmTreatData.h"

#ifdef IBM

void allocateIBMProc(IBMProc* procIBM)
{
    // printf("allocate ibm data\n");
}

void freeIBMProc(IBMProc* procIBM)
{
    // printf("free ibm data\n");
}

void treatDataIBM(IBMProc* procIBM, ParticlesSoA particles)
{
    ParticleCenter* pc = &(particles.pCenterArray[0]);
    // Kinematic viscosity
    dfloat nu = RHO_0*(TAU - 0.5)/3;

    // Cross section area
    dfloat tArea = M_PI*pc->radius*pc->radius;

    // Fixed sphere
    // procIBM->reynolds = U_MAX*2*pc->radius/nu;
    // procIBM->cd = 2*pc->f.z/(U_MAX*U_MAX*tArea);

    // Falling sphere
    // procIBM->reynolds = pc->vel.z*pc->radius*2/nu;
    // procIBM->cd = 2*pc->f.z/(RHO_0*pc->vel.z*pc->vel.z*tArea);
    // procIBM->clx = 2*pc->f.x/(RHO_0*pc->vel.z*pc->vel.z*tArea);
    // procIBM->cly = 2*pc->f.y/(RHO_0*pc->vel.z*pc->vel.z*tArea);
    procIBM->vel = pc->vel;
    procIBM->w = pc->w;
    procIBM->pos = pc->pos;
}

bool stopSimIBM(IBMProc* procIBM, ParticlesSoA particles)
{
    ParticleCenter* pc = &(particles.pCenterArray[0]);

    // Stop when the particle is near bottom wall
    if((pc->pos.z - pc->radius) < 2){
        return false;
    }
    return false;
}

void printTreatDataIBM(IBMProc* procIBM)
{
    /* PRINT TREATED DATA EXAMPLE */
    //printf("\n------------------------------- IBM TREATED DATA -------------------------------\n");
    printf("               Step: %d\n", *(procIBM->step));
    printf("       pos(x, y, z): (%.4f, %.4f, %.4f)\n", procIBM->pos.x, procIBM->pos.y, procIBM->pos.z);
    printf("       vel(x, y, z): (%.4e, %.4e, %.4e)\n", procIBM->vel.x, procIBM->vel.y, procIBM->vel.z);
    printf("         w(x, y, z): (%.4e, %.4e, %.4e)\n", procIBM->w.x, procIBM->w.y, procIBM->w.z);
    printf("--------------------------------------------------------------------------------\n");
}

void saveTreatDataIBM(IBMProc* procIBM)
{
    /* SAVE TO CSV EXAMPLE */
    std::string strFileIBMTreatData;
    strFileIBMTreatData = getVarFilename("ibmTreatData", *(procIBM->step), ".csv");

    FILE* fileIBMTreatData = nullptr;
    fileIBMTreatData = fopen(strFileIBMTreatData.c_str(), "w");
    if(fileIBMTreatData != nullptr)
    {
        // write header
        fprintf(fileIBMTreatData, "step\treynolds\tcd\tclx\tcly\n");
        // write data
        fprintf(fileIBMTreatData, "%d\t%.6e\t%.6e\t%.6e\t%.6e\n", 
            *(procIBM->step), procIBM->reynolds, procIBM->cd, procIBM->clx, procIBM->cly);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strFileIBMTreatData.c_str());
    }
}

#endif