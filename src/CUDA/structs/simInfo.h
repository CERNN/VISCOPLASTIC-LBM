/*
*   @file simInfo.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Struct for informations of the simulation
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __SIM_INFO_H
#define __SIM_INFO_H
#include "../var.h"
#include <cuda.h>

/* 
*   Struct for simulation info that is evaluated in runtime
*   such as MLUPS, bandwidth, GPU used, etc.
*/
typedef struct simInfo{
    // Performance related
    float MLUPS;
    float bandwidth;
    float timeElapsed;

    // Devices (GPUs) related
    cudaDeviceProp* devices;
    int numDevices;

    // Simulation related
    int totalSteps;

    /* Constructor */
    simInfo()
    {
        MLUPS = 0;
        bandwidth = 0;
        timeElapsed = 0;
        devices = nullptr;
        numDevices = 0;
        totalSteps = 0;
    }

    /* Destructor */
    ~simInfo()
    {
        MLUPS = 0;
        bandwidth = 0;
        timeElapsed = 0;
        devices = nullptr;
        numDevices = 0;
        totalSteps = 0;
    }
}SimInfo;

#endif // !__SIM_INFO_H