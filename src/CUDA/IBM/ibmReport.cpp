#include "ibmReport.h"

std::string getStrDfloat3(dfloat3 val, std::string sep){
    return std::to_string(val.x) + sep + std::to_string(val.y) + sep + std::to_string(val.z)
}


void saveParticlesInfo(ParticlesSoA particles, unsigned int step, bool saveNodes){
        // Names of files
    std::string strFilePCenters;

    strFilePCenters = getVarFilename("pCenters", step, ".bin");

    std::ofstream outFilePCenter(strFilePCenters.c_str());
    
    // tab as separator
    std::string strValuesParticles = "";
    std::string sep = "\t";
    std::string strColumnNames = "p_number\tstep\t";
    strColumnNames += "pos_x\tpos_y\tpos_z\t";
    strColumnNames += "vel_x\tvel_y\tvel_z\t";
    strColumnNames += "w_x\tw_y\tw_z\t";
    strColumnNames += "f_x\tf_y\tf_z\t";
    strColumnNames += "M_x\tM_y\tM_z\t";
    strColumnNames += "I_x\tI_y\tI_z\t";
    strColumnNames += "S\t";
    strColumnNames += "radius\t";
    strColumnNames += "volume\t";
    strColumnNames += "movable\n";

    for(int p = 0; p < NUM_PARTICLES; p++){
        ParticleCenter pc = particles.pCenterArray[p];
        strValuesParticles += std::to_string(p) + sep;
        strValuesParticles += std::to_string(step) + sep;
        strValuesParticles += getStrDfloat3(pc.vel, sep) + sep;
        strValuesParticles += getStrDfloat3(pc.w, sep) + sep;
        strValuesParticles += getStrDfloat3(pc.f, sep) + sep;
        strValuesParticles += getStrDfloat3(pc.M, sep) + sep;
        strValuesParticles += getStrDfloat3(pc.I, sep) + sep;
        strValuesParticles += getStrDfloat3(pc.I, sep) + sep;
        strValuesParticles += std::to_string(pc.S) + sep;
        strValuesParticles += std::to_string(pc.radius) + sep;
        strValuesParticles += std::to_string(pc.volume) + sep;
        strValuesParticles += std::to_string(pc.movable) + "\n";
    }

    std::string strWrite = strColumnNames + strValuesParticles;
    outFilePCenter << strWrite;
}

void printParticlesInfo(ParticlesSoA particles, unsigned int step){

}
