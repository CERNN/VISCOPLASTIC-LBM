#include "ibmReport.h"

std::string getStrDfloat3(dfloat3 val, std::string sep){
    std::ostringstream strValues("");
    strValues << std::scientific;
    strValues << val.x << sep << val.y << sep << val.z;
    return strValues.str();
}


void saveParticlesInfo(ParticlesSoA particles, unsigned int step, bool saveNodes){
    // Names of file to save particle info
    std::string strFilePCenters = getVarFilename("pCenters", step, ".csv");

    // File to save particle info
    std::ofstream outFilePCenter(strFilePCenters.c_str());

    // String with all values as csv
    std::ostringstream strValuesParticles("");
    strValuesParticles << std::scientific;

    // csv separator
    std::string sep = ",";
    // Column names to use in csv
    std::string strColumnNames = "p_number" + sep + "step" + sep;
    strColumnNames += "pos_x" + sep  + "pos_y" + sep  + "pos_z" + sep;
    strColumnNames += "vel_x" + sep  + "vel_y" + sep  + "vel_z" + sep;
    strColumnNames += "w_x" + sep  + "w_y" + sep  + "w_z" + sep;
    strColumnNames += "f_x" + sep  + "f_y" + sep  + "f_z" + sep;
    strColumnNames += "M_x" + sep  + "M_y" + sep  + "M_z" + sep;
    strColumnNames += "I_x" + sep  + "I_y" + sep  + "I_z" + sep;
    strColumnNames += "S" + sep;
    strColumnNames += "radius" + sep;
    strColumnNames += "volume" + sep;
    strColumnNames += "movable\n";

    for(int p = 0; p < NUM_PARTICLES; p++){
        ParticleCenter pc = particles.pCenterArray[p];
        strValuesParticles << p << sep;
        strValuesParticles << step << sep;
        strValuesParticles << getStrDfloat3(pc.pos, sep) << sep;
        strValuesParticles << getStrDfloat3(pc.vel, sep) << sep;
        strValuesParticles << getStrDfloat3(pc.w, sep) << sep;
        strValuesParticles << getStrDfloat3(pc.f, sep) << sep;
        strValuesParticles << getStrDfloat3(pc.M, sep) << sep;
        strValuesParticles << getStrDfloat3(pc.I, sep) << sep;
        strValuesParticles << pc.S << sep;
        strValuesParticles << pc.radius << sep;
        strValuesParticles << pc.volume << sep;
        strValuesParticles << pc.movable << "\n";
    }

    outFilePCenter << strColumnNames << strValuesParticles.str();

    if(saveNodes){
        strColumnNames = "particle_index" + sep + "pos_x" + sep + "pos_y" + sep + "pos_z" + sep + "S\n";

        std::ostringstream strValuesMesh("");
        strValuesMesh << std::scientific;
        ParticleNodeSoA pnSoA = particles.nodesSoA;
        for(int i = 0; i < pnSoA.numNodes; i++){
            dfloat3 pos = pnSoA.pos.getValuesFromdIdx(i);
            strValuesMesh << pnSoA.particleCenterIdx[i] << sep;
            strValuesMesh << getStrDfloat3(pos, sep) << sep;
            strValuesMesh << pnSoA.S[i] << "\n";
        }
        
        // Names of file to save particle info
        std::string strFilePNodes = getVarFilename("pNodes", step, ".csv");

        // File to save particle info
        std::ofstream outFilePNodes(strFilePNodes);

        outFilePNodes << strColumnNames << strValuesMesh.str();
    }
}

