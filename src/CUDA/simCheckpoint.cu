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

#include "simCheckpoint.h"

#define __LOAD_CHECKPOINT 1
#define __SAVE_CHECKPOINT 2


void createFolder(std::string foldername){
    // Check if folder exists
    struct stat buffer;
    if(stat(foldername.c_str(), &buffer) == 0)
        return;
    #ifdef _WIN32
    std::string cmd = "md ";
    cmd += foldername;
    system(cmd.c_str());
    #else
    if(std::mkdir(foldername, 0777) == -1)
        std::cout << "Error creating folder '" << foldername << "'.\n";
    #endif
}


/**
*   @brief Get the filesize of file
*   
*   @param filename Filename to get size
*   @return size_t Filesize
*/
size_t getFileSize(std::string filename)
{
    std::streampos fsize = 0;
    std::ifstream file( filename, std::ios::binary );

    fsize = file.tellg();
    file.seekg( 0, std::ios::end);
    fsize = file.tellg() - fsize;
    file.close();

    return fsize;
}

/**
*   @brief Reads file content into GPU array
*
*   @param arr GPU array to write file content to
*   @param filename Filename to read from
*   @param arr_size_bytes Size in bytes to read from file. If zero, reads whole file
*   @param tmp Temporary array used to read file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__
void readFileIntoArray(void* arr, std::string filename, size_t arr_size_bytes, void* tmp){
    FILE* file = fopen((filename+".bin").c_str(), "rb");
    // Check if file exists
    if(file == nullptr){
        std::cout << "Error reading file '" << filename << ".bin'. Exiting\n";
    }
    // load file size into array, if it is zero
    if(arr_size_bytes == 0){
        arr_size_bytes = getFileSize(filename);
    }

    // Read file into temporary array
    fread(tmp, arr_size_bytes, 1, file);
    // Copy file content in tmp to GPU array
    checkCudaErrors(cudaMemcpy(arr, tmp, arr_size_bytes, cudaMemcpyDefault));

    fclose(file);
}

/**
*   @brief Writes GPU array content into file
*
*   @param arr GPU array to read content from
*   @param filename Filename to write to
*   @param arr_size_bytes Size in bytes to write to file
*   @param tmp Temporary array used to write to file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__
void writeFileIntoArray(void* arr, const std::string filename, const size_t arr_size_bytes, void* tmp){
    FILE* file = fopen((filename+".bin").c_str(), "wb");
    // Check if file exists
    if(file == nullptr){
        std::cout << "Error opening file '" << filename << ".bin' to write. Exiting\n";
    }

    // Copy file content from GPU array to tmp
    checkCudaErrors(cudaMemcpy(tmp, arr, arr_size_bytes, cudaMemcpyDefault));

    // Write temporary array into file
    fwrite(tmp, arr_size_bytes, 1, file);

    fclose(file);
}

/**
*   @brief Writes dfloat3SoA GPU arrays content into files
*
*   @param arr GPU arrays to read content from
*   @param foldername Foldername to save files to
*   @param arr_size_bytes Size in bytes to write for each file. If zero, reads whole file
*   @param tmp Temporary array used to read from file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__ 
void writeFilesIntoDfloat3SoA(dfloat3SoA arr, const std::string foldername, const size_t arr_size_bytes, void* tmp){
    // Write x, y and z to files
    createFolder(foldername);
    #ifdef _WIN32
    writeFileIntoArray(arr.x, foldername + "\\\\x", arr_size_bytes, tmp);
    writeFileIntoArray(arr.y, foldername + "\\\\y", arr_size_bytes, tmp);
    writeFileIntoArray(arr.z, foldername + "\\\\z", arr_size_bytes, tmp);
    #else
    writeFileIntoArray(arr.x, foldername + "/x", arr_size_bytes, tmp);
    writeFileIntoArray(arr.y, foldername + "/y", arr_size_bytes, tmp);
    writeFileIntoArray(arr.z, foldername + "/z", arr_size_bytes, tmp);
    #endif
}

/**
*   @brief Reads files contents into dfloat3SoA GPU arrays
*
*   @param arr GPU arrays to write content to
*   @param foldername Foldername to read files from
*   @param arr_size_bytes Size in bytes to read for each file. 
*   @param tmp Temporary array used to write to file (already allocated, 
*                make sure that the file content fits in it)
*/
__host__
void readFilesIntoDfloat3SoA(dfloat3SoA arr, const std::string foldername, const size_t arr_size_bytes, void* tmp){
    // Read to x, y and z in dfloat3SoA
    #ifdef _WIN32
    readFileIntoArray(arr.x, foldername + "\\\\x", arr_size_bytes, tmp);
    readFileIntoArray(arr.y, foldername + "\\\\y", arr_size_bytes, tmp);
    readFileIntoArray(arr.z, foldername + "\\\\z", arr_size_bytes, tmp);
    #else
    readFileIntoArray(arr.x, foldername + "/x", arr_size_bytes, tmp);
    readFileIntoArray(arr.y, foldername + "/y", arr_size_bytes, tmp);
    readFileIntoArray(arr.z, foldername + "/z", arr_size_bytes, tmp);
    #endif
}


/**
*   @brief Get the checkpoint filename to read from
*   
*   @param name Field name (such as "rho", "u", etc.)
*   @param n_gpu GPU number
*   @return std::string string checkpoint filename
*/
__host__
std::string getCheckpointFilenameRead(std::string name, int n_gpu){
    std::string filename = SIMULATION_FOLDER_LOAD_CHECKPOINT;
    #ifdef _WIN32
    return filename + "\\\\" + ID_SIM + "\\\\checkpoint\\\\" + 
        std::to_string(n_gpu) + "_" + name;
    #else
    return filename + "/" + ID_SIM + "/checkpoint/" + 
        std::to_string(n_gpu) + "_" + name;
    #endif
}


/**
*   @brief Get the checkpoint filename to write to
*   
*   @param name Field name (such as "rho", "u", etc.)
*   @param n_gpu GPU number
*   @return std::string string checkpoint filename
*/
__host__
std::string getCheckpointFilenameWrite(std::string name, int n_gpu){
    std::string filename = PATH_FILES;
    #ifdef _WIN32
    return filename + "\\\\" + ID_SIM + "\\\\checkpoint\\\\" + 
        std::to_string(n_gpu) + "_" + name;
    #else
    return filename + "/" + ID_SIM + "/checkpoint/" + 
        std::to_string(n_gpu) + "_" + name;
    #endif
}


/**
*   @brief Operation over checkpoint, save or load 
*   
*   @param oper operation to do, either __LOAD_CHECKPOINT or __SAVE_CHECKPOINT 
*   @param pop Populations array
*   @param macr Macroscopics array
*   @param particlesSoA Particles structure of arrays object
*   @param step Pointer to current step value in main
*/
__host__
void operateSimCheckpoint( 
    int oper,
    Populations* pop,
    Macroscopics* macr,
    ParticlesSoA particlesSoA,
    int* step
    )
{
    // Defining what functions to use (read or write to files)
    void (*f_arr)(void*, const std::string, size_t, void*);
    void (*f_dfloat3SoA)(dfloat3SoA, const std::string, size_t, void*);
    std::string (*f_filename)(std::string, int);

    if(oper == __LOAD_CHECKPOINT){
        f_arr = &readFileIntoArray;
        f_dfloat3SoA = &readFilesIntoDfloat3SoA;
        f_filename = &getCheckpointFilenameRead;
    }else if(oper == __SAVE_CHECKPOINT){
        f_arr = &writeFileIntoArray;
        f_dfloat3SoA = &writeFilesIntoDfloat3SoA;
        f_filename = &getCheckpointFilenameWrite;
    }else{
        std::cout << "Invalid operation. Exiting\n";
        exit(-1);
    }

    // Everything will fit in this array
    dfloat* tmp = (dfloat*)malloc(MEM_SIZE_POP);

    // Load/save current step
    f_arr(step, f_filename("curr_step", 0), sizeof(int), tmp);

    #ifdef IBM
    // Load particles centers positions
    checkCudaErrors(cudaSetDevice(GPUS_TO_USE[0]));
    f_arr(particlesSoA.pCenterArray, f_filename("IBM_particles_centers", 0), 
        NUM_PARTICLES*sizeof(ParticleCenter), tmp);
    #endif

    for(int i = 0; i < N_GPUS; i++){
        checkCudaErrors(cudaSetDevice(GPUS_TO_USE[i]));
        // Load/save pop
        f_arr(pop[i].pop, f_filename("pop", i), MEM_SIZE_POP, tmp);
        // Load/save popAux
        f_arr(pop[i].popAux, f_filename("popAux", i), MEM_SIZE_POP, tmp);
        // Load/save macroscopics
        f_arr(macr[i].rho, f_filename("rho", i), MEM_SIZE_IBM_SCALAR, tmp);
        // Load/save velocities
        f_dfloat3SoA(macr[i].u, f_filename("u", i), MEM_SIZE_IBM_SCALAR, tmp);

        #ifdef NON_NEWTONIAN_FLUID
        f_arr(macr[i].omega, f_filename("omega", i), MEM_SIZE_SCALAR, tmp);
        #endif

        #ifdef IBM
        f_dfloat3SoA(macr[i].f, f_filename("f", i), MEM_SIZE_IBM_SCALAR, tmp);

        ParticleNodeSoA nSoA = particlesSoA.nodesSoA[i];

        // IBM nodes bytes size
        if(oper == __LOAD_CHECKPOINT){
            size_t filesize = getFileSize(f_filename("IBM_nodes_centers_idx", i));
            nSoA.numNodes = (filesize) / sizeof(unsigned int);
        }
        size_t ibm_nodes_arr_size = nSoA.numNodes * sizeof(dfloat);
        size_t ibm_nodes_arr_size_uint = nSoA.numNodes * sizeof(unsigned int);
        // Load/save IBM nodes values
        f_arr(nSoA.particleCenterIdx, f_filename("IBM_nodes_centers_idx", i), ibm_nodes_arr_size_uint, tmp);
        f_dfloat3SoA(nSoA.pos, f_filename("IBM_nodes_pos", i), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.vel, f_filename("IBM_nodes_vel", i), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.vel_old, f_filename("IBM_nodes_vel_old", i), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.f, f_filename("IBM_nodes_f", i), ibm_nodes_arr_size, tmp);
        f_dfloat3SoA(nSoA.deltaF, f_filename("IBM_nodes_deltaF", i), ibm_nodes_arr_size, tmp);
        f_arr(nSoA.S, f_filename("IBM_nodes_S", i), ibm_nodes_arr_size, tmp);
        #endif
    }

    free(tmp);
}

__host__
void loadSimCheckpoint( 
    Populations pop[N_GPUS],
    Macroscopics macr[N_GPUS],
    ParticlesSoA particlesSoA,
    int *step
    ){
    operateSimCheckpoint(__LOAD_CHECKPOINT, pop, macr, particlesSoA, step);
}

__host__
void saveSimCheckpoint( 
    Populations pop[N_GPUS],
    Macroscopics macr[N_GPUS],
    ParticlesSoA particlesSoA,
    int *step
    ){
    std::string foldername = PATH_FILES; 
    foldername += "\\\\";
    foldername += ID_SIM;
    foldername += "\\\\checkpoint";
    createFolder(foldername);
    operateSimCheckpoint(__SAVE_CHECKPOINT, pop, macr, particlesSoA, step);
}