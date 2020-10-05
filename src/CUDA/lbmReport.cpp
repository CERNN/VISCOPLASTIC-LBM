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
    // names of files
    std::string strFileRho, strFileUx, strFileUy, strFileUz;
    
    strFileRho = getVarFilename("rho", nSteps, ".bin");
    strFileUx = getVarFilename("ux", nSteps, ".bin");
    strFileUy = getVarFilename("uy", nSteps, ".bin");
    strFileUz = getVarFilename("uz", nSteps, ".bin");

    // saving files
    saveVarBin(strFileRho, macr->rho, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUx, macr->ux, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUy, macr->uy, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileUz, macr->uz, TOTAL_MEM_SIZE_SCALAR, false);

    #ifdef IBM
    std::string strFileFx = getVarFilename("fx", nSteps, ".bin");
    std::string strFileFy = getVarFilename("fy", nSteps, ".bin");
    std::string strFileFz = getVarFilename("fz", nSteps, ".bin");

    saveVarBin(strFileFx, macr->fx, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileFy, macr->fy, TOTAL_MEM_SIZE_SCALAR, false);
    saveVarBin(strFileFz, macr->fz, TOTAL_MEM_SIZE_SCALAR, false);
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
                        x, y, z, macr->rho[idx], macr->ux[idx], macr->uy[idx], 
                        macr->uz[idx]);
                }
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strOutFile.c_str());
    }
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
        fprintf(outFile, "\n---------------------------- SIMULATION INFORMATION ----------------------------\n");
        fprintf(outFile, "      Simulation ID: %s\n", ID_SIM);
        #ifdef D3Q19
        fprintf(outFile, "       Velocity set: D3Q19\n");
        #endif // !D3Q19
        #ifdef D3Q27
        fprintf(outFile, "       Velocity set: D3Q27\n");
        #endif // !D3Q27
        if(sizeof(dfloat) == sizeof(float))
            fprintf(outFile, "          Precision: float\n");
        else if(sizeof(dfloat) == sizeof(double))
            fprintf(outFile, "          Precision: double\n");
        fprintf(outFile, "                 NX: %d\n", NX);
        fprintf(outFile, "                 NY: %d\n", NY);
        fprintf(outFile, "                 NZ: %d\n", NZ);
        fprintf(outFile, "           NZ_TOTAL: %d\n", NZ_TOTAL);
        fprintf(outFile, "                Tau: %.6f\n", TAU);
        fprintf(outFile, "               Umax: %.6e\n", U_MAX);
        fprintf(outFile, "                 FX: %.6e\n", FX);
        fprintf(outFile, "                 FY: %.6e\n", FY);
        fprintf(outFile, "                 FZ: %.6e\n", FZ);  
        fprintf(outFile, "       Report steps: %d\n", DATA_REPORT);
        fprintf(outFile, "         Save steps: %d\n", MACR_SAVE);
        fprintf(outFile, "             Nsteps: %d\n", info->totalSteps);
        fprintf(outFile, "              MLUPS: %.1f\n", info->MLUPS);
        fprintf(outFile, "          Bandwidht: %.1f (Gb/s)\n", info->bandwidth);
        fprintf(outFile, "       Time elapsed: %.3f (s)\n", info->timeElapsed);
        fprintf(outFile, "            threads: (%d, %d, %d)\n", N_THREADS, 1, 1);
        fprintf(outFile, "--------------------------------------------------------------------------------\n");
    
        fprintf(outFile, "\n------------------------------- CUDA INFORMATION -------------------------------\n");
        for(int i = 0; i < info->numDevices; i++)
        {
            fprintf(outFile, "\t      device number: %d\n", i);
            fprintf(outFile, "\t               name: %s\n", info->devices[i].name);
            fprintf(outFile, "\t    multiprocessors: %d\n", info->devices[i].multiProcessorCount);
            fprintf(outFile, "\t compute capability: %d.%d\n", info->devices[i].major, 
                                                               info->devices[i].minor);
            fprintf(outFile, "\t        ECC enabled: %d\n", info->devices[i].ECCEnabled);
        }
        fprintf(outFile, "--------------------------------------------------------------------------------\n");
        fclose(outFile);
    }
    else
    {
        printf("Error saving \"%s\" \nProbably wrong path!\n", strInf.c_str());
    }
    
}


void printParamInfo(
    SimInfo* info,
    bool hasEnded)
{          
    printf("\n---------------------------- SIMULATION INFORMATION ----------------------------\n");
    printf("      Simulation ID: %s\n", ID_SIM);
#ifdef D3Q19
    printf("       Velocity set: D3Q19\n");
#endif // !D3Q19
#ifdef D3Q27
    printf("       Velocity set: D3Q27\n");
#endif // !D3Q27
    if(sizeof(dfloat) == sizeof(float))
        printf("          Precision: float\n");
    else if(sizeof(dfloat) == sizeof(double))
        printf("          Precision: double\n");
    printf("                 NX: %d\n", NX);
    printf("                 NY: %d\n", NY);
    printf("                 NZ: %d\n", NZ);
    printf("           NZ_TOTAL: %d\n", NZ_TOTAL);
    printf("                Tau: %.6e\n", TAU);
    printf("               Umax: %.6e\n", U_MAX);
    printf("                 FX: %.6e\n", FX);
    printf("                 FY: %.6e\n", FY);
    printf("                 FZ: %.6e\n", FZ);  
    printf("     Residual steps: %d\n", DATA_REPORT);
    printf("         Save steps: %d\n", MACR_SAVE);
    printf("             Nsteps: %d\n", info->totalSteps);
    if(hasEnded)
    {
        printf("              MLUPS: %.1f\n", info->MLUPS);
        printf("          Bandwidht: %.1f (Gb/s)\n", info->bandwidth);
        printf("       Time elapsed: %.3f (s)\n", info->timeElapsed);
    }
    printf("            threads: (%d, %d, %d)\n", N_THREADS, 1, 1);
    printf("--------------------------------------------------------------------------------\n");
}


void printGPUInfo(SimInfo* info)
{
    printf("\n------------------------------- CUDA INFORMATION -------------------------------\n");
    for(int i = 0; i < info->numDevices; i++)
    {
        printf("\t      device number: %d\n", i);
        printf("\t               name: %s\n", info->devices[i].name);
        printf("\t    multiprocessors: %d\n", info->devices[i].multiProcessorCount);
        printf("\t compute capability: %d.%d\n", info->devices[i].major, 
                                                 info->devices[i].minor);
        printf("\t        ECC enabled: %d\n", info->devices[i].ECCEnabled);
        printf("--------------------------------------------------------------------------------\n");
    }
}