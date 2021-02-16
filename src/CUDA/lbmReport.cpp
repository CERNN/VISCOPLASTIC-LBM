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

#include "lbmReport.h"


void folderSetup()
{
// Windows
#if defined(_WIN32)
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "\\\\"; // adds "\\"
    strPath += ID_SIM;
    std::string cmd = "md ";
    cmd += strPath;
    system(cmd.c_str());
    return;
#endif // !_WIN32

// Unix
#if defined(__APPLE__) || defined(__MACH__) || defined(__linux__)
    std::string strPath;
    strPath = PATH_FILES;
    strPath += "/";
    strPath += ID_SIM;
    std::string cmd = "mkdir -p ";
    cmd += strPath;
    system(cmd.c_str());
    return;
#endif // !Unix
    printf("I don't know how to setup folders for your operational system :(\n");
    return;
}


std::string getVarFilename(
    const std::string varName, 
    unsigned int step,
    const std::string ext)
{
    unsigned int n_zeros = 0, pot_10 = 10;
    unsigned int aux1 = 1000000;  // 6 numbers on step
    // calculate number of zeros
    if (step != 0)
        for (n_zeros = 0; step * pot_10 < aux1; pot_10 *= 10)
            n_zeros++;
    else
        n_zeros = 6;

    // generates the file name as "PATH_FILES/id/id_varName000000.bin"
    std::string strFile = PATH_FILES;
    strFile += "/";
    strFile += ID_SIM;
    strFile += "/";
    strFile += ID_SIM;
    strFile += "_";
    strFile += varName;
    for (unsigned int i = 0; i < n_zeros; i++)
        strFile += "0";
    strFile += std::to_string(step);
    strFile += ext;

    return strFile;
}


void saveVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize,
    bool append)
{
    FILE* outFile = nullptr;
    if(append)
        outFile = fopen(strFile.c_str(), "ab");
    else
        outFile = fopen(strFile.c_str(), "wb");
    if(outFile != nullptr)
    {
        fwrite(var, memSize, 1, outFile);
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strFile.c_str());
    }
}


void savePopBin(
    Populations* pop, 
    unsigned int nSteps)
{
    std::string strFilePop, strFilePopAux;
    strFilePop = getVarFilename("pop", nSteps, ".bin");
    strFilePopAux = getVarFilename("pop_aux", nSteps, ".bin");

    dfloat* tmp = nullptr;
    checkCudaErrors(cudaMallocHost((void**)&(tmp), MEM_SIZE_POP));
    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaMemcpy(tmp, pop[i].pop, MEM_SIZE_POP, cudaMemcpyDeviceToHost));
        saveVarBin(strFilePop, tmp, MEM_SIZE_POP, i != 0);
    }

    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaMemcpy(tmp, pop[i].popAux, MEM_SIZE_POP, cudaMemcpyDeviceToHost));
        saveVarBin(strFilePopAux, tmp, MEM_SIZE_POP, i != 0);
    }

    checkCudaErrors(cudaFreeHost(tmp));
}


void saveAllMacrBin(
    Macroscopics* macr, 
    unsigned int nSteps)
{
    // Names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz;

    strFileRho = getVarFilename("rho", nSteps, ".bin");
    strFileUx = getVarFilename("ux", nSteps, ".bin");
    strFileUy = getVarFilename("uy", nSteps, ".bin");
    strFileUz = getVarFilename("uz", nSteps, ".bin");

    // saving files
    saveVarBin(strFileRho, macr->rho, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUx, macr->u.x, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUy, macr->u.y, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUz, macr->u.z, TOTAL_MEM_SIZE_SCALAR, false);

    #if defined(IBM) && EXPORT_FORCES
    std::string strFileFx = getVarFilename("fx", nSteps, ".bin");
    std::string strFileFy = getVarFilename("fy", nSteps, ".bin");
    std::string strFileFz = getVarFilename("fz", nSteps, ".bin");

    saveVarBin(strFileFx, macr->f.x, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileFy, macr->f.y, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileFz, macr->f.z, TOTAL_MEM_SIZE_SCALAR, false);
    #endif
    
    #ifdef NON_NEWTONIAN_FLUID
    std::string strFileOmega = getVarFilename("omega", nSteps, ".bin");

    saveVarBin(strFileOmega, macr->omega, TOTAL_MEM_SIZE_SCALAR, false);
    #endif

    #ifdef SCALAR_TRANSPORT
    std::string strFileG = getVarFilename("G", nSteps, ".bin");

    saveVarBin(strFileG, macr->G, TOTAL_MEM_SIZE_SCALAR, false);
    #endif
}


void saveAllMacrCsv(
    Macroscopics* macr, 
    unsigned int nSteps)
{
    std::string strOutFile;
    FILE *outFile = nullptr;
    
    strOutFile = getVarFilename("macr", nSteps, ".csv");

    outFile = fopen(strOutFile.c_str(), "w");
    if(outFile != nullptr)
    {
        std::string header = "x\ty\tz\trho\tux\tux\tuy\tuz\n";
        fprintf(outFile, "%s", header.c_str());
        for(int z = 0; z < NZ; z++)
            for(int y = 0; y < NY; y++)
                for(int x = 0; x < NX; x++)
                {
                    size_t idx = idxScalar(x, y, z);
                    fprintf(outFile, "%d\t%d\t%d\t%.6e\t%.6e\t%.6e\t%.6e\n", 
                        x, y, z, macr->rho[idx], macr->u.x[idx], macr->u.y[idx], 
                        macr->u.z[idx]);
                }
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strOutFile.c_str());
    }
}

std::string getSimInfoString(SimInfo* info)
{
    std::ostringstream strSimInfo("");
    
    strSimInfo << std::scientific;
    strSimInfo << std::setprecision(6);
    
    strSimInfo << "---------------------------- SIMULATION INFORMATION ----------------------------\n";
    strSimInfo << "      Simulation ID: " << ID_SIM << "\n";
    #ifdef D3Q19
    strSimInfo << "       Velocity set: D3Q19\n";
    #endif // !D3Q19
    #ifdef D3Q27
    strSimInfo << "       Velocity set: D3Q27\n";
    #endif // !D3Q27
    #ifdef SINGLE_PRECISION
        strSimInfo << "          Precision: float\n";
    #else
        strSimInfo << "          Precision: double\n";
    #endif
    strSimInfo << "                 NX: " << NX << "\n";
    strSimInfo << "                 NY: " << NY << "\n";
    strSimInfo << "                 NZ: " << NZ << "\n";
    strSimInfo << "           NZ_TOTAL: " << NZ_TOTAL << "\n";
    strSimInfo << std::scientific << std::setprecision(6);
    strSimInfo << "                Tau: " << TAU << "\n";
    strSimInfo << "               Umax: " << U_MAX << "\n";
    strSimInfo << "                 FX: " << FX << "\n";
    strSimInfo << "                 FY: " << FY << "\n";
    strSimInfo << "                 FZ: " << FZ << "\n";
    strSimInfo << "       Report steps: " << DATA_REPORT << "\n";
    strSimInfo << "         Save steps: " << MACR_SAVE << "\n";
    strSimInfo << "             Nsteps: " << info->totalSteps << "\n";
    strSimInfo << std::fixed << std::setprecision(1);
    strSimInfo << "              MLUPS: " << info->MLUPS << "\n";
    strSimInfo << "          Bandwidht: " << info->bandwidth << " (Gb/s)\n";
    strSimInfo << std::setprecision(3);
    strSimInfo << "       Time elapsed: " << info->timeElapsed << " (s)\n";
    strSimInfo << "            threads: (" << N_THREADS << " , 1, 1)\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";

    #ifdef NON_NEWTONIAN_FLUID
    strSimInfo << "\n------------------------------ NON NEWTONIAN FLUID -----------------------------\n";
    strSimInfo << std::scientific << std::setprecision(6);
    
    #ifdef POWERLAW
    strSimInfo << "              Model: Power-Law\n";
    strSimInfo << "        Power index: " << N_INDEX << "\n";
    strSimInfo << " Consistency factor: " << K_CONSISTENCY << "\n";
    strSimInfo << "            Gamma 0: " << GAMMA_0 << "\n";
    #endif // POWERLAW

    #ifdef BINGHAM
    strSimInfo << "              Model: Bingham\n";
    strSimInfo << "  Plastic viscosity: " << ETA_P << "\n";
    strSimInfo << "       Yield stress: " << S_Y << "\n";
    strSimInfo << "      Plastic omega: " << OMEGA_P << "\n";
    #endif // BINGHAM
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif // NON_NEWTONIAN_FLUID

    #ifdef IBM
    strSimInfo << "\n------------------------------------- IBM ------------------------------------\n";
    strSimInfo << std::scientific << std::setprecision(6);

    strSimInfo << "   Number of particles: " << NUM_PARTICLES << "\n";
    strSimInfo << "        IBM iterations: " << IBM_MAX_ITERATION << "\n";
    strSimInfo << "          Stencil size: ";

    #if defined STENCIL_2
    strSimInfo << "2" << "\n";
    #elif defined STENCIL_4
    strSimInfo << "4" << "\n";
    #else
    strSimInfo << "Invalid" << "\n";
    #endif

    strSimInfo << "  Particle density cte: " << PARTICLE_DENSITY << "\n";
    strSimInfo << "         Fluid density: " << FLUID_DENSITY << "\n";
    strSimInfo << "                    GX: " << GX << "\n";
    strSimInfo << "                    GY: " << GY << "\n";
    strSimInfo << "                    GZ: " << GZ << "\n";
    strSimInfo << std::fixed << std::setprecision(2);
    strSimInfo << "            Mesh scale: " << MESH_SCALE << "\n";
    strSimInfo << "          Mesh coulomb: " << MESH_COULOMB << "\n";
    strSimInfo << "         IBM thickness: " << IBM_THICKNESS << "\n";
    strSimInfo << "        Particles save: " << IBM_PARTICLES_SAVE << "\n";
    strSimInfo << "  Particles nodes save: " << IBM_PARTICLES_NODES_SAVE << "\n";
    strSimInfo << "       IBM data report: " << IBM_DATA_REPORT << "\n";
    strSimInfo << "         IBM data stop: " << IBM_DATA_STOP << "\n";
    strSimInfo << "         IBM data save: " << IBM_DATA_SAVE << "\n";
    strSimInfo << "IBM Euler optimization: " << IBM_EULER_OPTIMIZATION << "\n";
    #if IBM_EULER_OPTIMIZATION
    strSimInfo << "       Shell thickness: " << IBM_EULER_SHELL_THICKNESS << "\n";
    #endif
    #if defined SOFT_SPHERE
    strSimInfo << "--------------------------------- IBM Collision --------------------------------\n";
    strSimInfo << "\t       Friction Coef.: " << FRICTION_COEF << "\n";
    strSimInfo << "\t   Restitution  Coef.: " << REST_COEF << "\n";
    strSimInfo << "\t      Young's Modulus: " << YOUNG_MODULUS << "\n";
    strSimInfo << "\t        Poisson Ratio: " << POISSON_RATIO << "\n";
    strSimInfo << "\t       Shear Modulus.: " << SHEAR_MODULUS << "\n";
    /*strSimInfo << "\t                 Zeta: " << ZETA << "\n";
    strSimInfo << "\t           Stiff wall: " << STIFF_WALL << "\n";
    strSimInfo << "\t           Stiff soft: " << STIFF_SOFT << "\n";
    strSimInfo << "\t           Stiff hard: " << STIFF_HARD << "\n";*/
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif //SOFT_SPHERE
    #if defined HARD_SPHERE
    strSimInfo << "--------------------------------- IBM Collision --------------------------------\n";
    strSimInfo << "\t       Friction Coef.: " << FRICTION_COEF << "\n";
    strSimInfo << "\t   Restitution  Coef.: " << REST_COEF << "\n";
    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif //HARD_SPHERE

    strSimInfo << "--------------------------------------------------------------------------------\n";
    #endif // IBM

    strSimInfo << "\n------------------------------- CUDA INFORMATION -------------------------------\n";
    for(int i = 0; i < info->numDevices; i++)
    {
        strSimInfo << "\t      device number: "<< i <<"\n";
        strSimInfo << "\t               name: " << info->devices[i].name << "\n";
        strSimInfo << "\t    multiprocessors: " << info->devices[i].multiProcessorCount << "\n";
        strSimInfo << "\t compute capability: " << info->devices[i].major << "." << info->devices[i].minor << "\n";
        strSimInfo << "\t        ECC enabled: " << info->devices[i].ECCEnabled << "\n";
    }
    strSimInfo << "--------------------------------------------------------------------------------\n";

    return strSimInfo.str();
}

void saveSimInfo(SimInfo* info)
{
    std::string strInf = PATH_FILES;
    strInf += "/";
    strInf += ID_SIM;
    strInf += "/";
    strInf += ID_SIM;
    strInf += "_info.txt"; // generate file name (with path)
    FILE* outFile = nullptr;

    outFile = fopen(strInf.c_str(), "w");
    if(outFile != nullptr)
    {
        std::string strSimInfo = getSimInfoString(info);
        fprintf(outFile, strSimInfo.c_str());
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strInf.c_str());
    }
    
}


void printSimInfo(
    SimInfo* info)
{
    printf(getSimInfoString(info).c_str()); fflush(stdout);
}
