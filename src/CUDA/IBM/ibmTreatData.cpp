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
    ParticleCenter* pc1 = &(particles.pCenterArray[1]);
    ParticleCenter* pc2 = &(particles.pCenterArray[2]);
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

    procIBM->vel1 = pc1->vel;
    procIBM->w1 = pc1->w;
    procIBM->pos1 = pc1->pos;

    procIBM->vel2 = pc2->vel;
    procIBM->w2 = pc2->w;
    procIBM->pos2 = pc2->pos;
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
    printf("\n------------------------------- IBM TREATED DATA -------------------------------\n");
    printf("               Step: %d\n", *(procIBM->step));
//    printf("           Reynolds: %.4e\n", procIBM->reynolds);
//    printf("                 Cd: %.4e\n", procIBM->cd);
//    printf("                Clx: %.4e\n", procIBM->clx);
//    printf("                Cly: %.4e\n", procIBM->cly);
    printf("       pos(x, y, z): (%.4f, %.4e, %.4f)\n", procIBM->pos.x, procIBM->pos.y, procIBM->pos.z);
    printf("       vel(x, y, z): (%.4e, %.4e, %.4e)\n", procIBM->vel.x, procIBM->vel.y, procIBM->vel.z);
    printf("      pos1(x, y, z): (%.4f, %.4e, %.4f)\n", procIBM->pos1.x, procIBM->pos1.y, procIBM->pos1.z);
    printf("      vel1(x, y, z): (%.4e, %.4e, %.4e)\n", procIBM->vel1.x, procIBM->vel1.y, procIBM->vel1.z);
    printf("      pos1(x, y, z): (%.4f, %.4e, %.4f)\n", procIBM->pos2.x, procIBM->pos2.y, procIBM->pos2.z);
    printf("      vel1(x, y, z): (%.4e, %.4e, %.4e)\n", procIBM->vel2.x, procIBM->vel2.y, procIBM->vel2.z);
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