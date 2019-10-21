/*
*   @file lbmReport.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Report (save and print) simulation information
*   @version 0.2.0
*   @date 16/08/2019
*/

#ifndef __LBM_REPORT_H
#define __LBM_REPORT_H

#include <string>
#include <cuda.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "structs/macroscopics.h"
#include "structs/populations.h"
#include "structs/simInfo.h"



/*
*   @brief Setup folder to save variables
*/
void folderSetup();


/*
*   @brief Get variable filename
*   @param var_name: name of the variable
*   @param step: steps number of the file
*   @param ext: file extension (with dot, e.g. ".bin", ".csv")
*   @return filename string
*/
std::string getVarFilename(
    const std::string varName, 
    unsigned int step,
    const std::string ext
);


/*
*   @brief Save array content to binary file
*   @param strFile: filename to save
*   @param var: float variable to save
*   @param memSize: sizeof var
*/
void saveVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize
);


/*
*   @brief Save populations in binary format
*   @param pop: macroscopics to save
*   @param nSteps: number of steps of the simulation
*   @obs Check CPU endianess
*   @obs The initial position of the array is x=0 and y=0 and z=0, 
*        so the variables starts on SWF and ends in NEB
*/
void savePopBin(
    Populations* pop, 
    unsigned int nSteps
);


/*
*   @brief Save all macroscopics in binary format
*   @param macr: macroscopics to save
*   @param nSteps: number of steps of the simulation
*   @obs Check CPU endianess
*   @obs The initial position of the array is x=0 and y=0 and z=0, 
*        so the variables starts on SWF and ends in NEB
*/
void saveAllMacrBin(
    Macroscopics* macr, 
    unsigned int nSteps
);


/*
*   Save all macroscopics in csv format (x y z macroscopics)
*   @param macr: macroscopics to save
*   @param nSteps: number of steps of the simulation
*   @obs Check CPU endianess
*   @obs The initial position of the array is x=0 and y=0 and z=0, 
*        so the variables starts on SWF and ends in NEB
*/
void saveAllMacrCsv(
    Macroscopics* macr, 
    unsigned int nSteps
);


/*
*   Save simulation's information
*   @param info: simulation's informations
*/
void saveSimInfo(
    SimInfo* info
);


/*
*   Print simulation parameters
*   @param info: simulation's informations
*   @param hasEnded: simulation has ended or not
*/
void printParamInfo(
    SimInfo* info,
    bool hasEnded
);


/*
*   Print GPUs information
*   @param info: simulation's informations
*   @param hasEnded: simulation has ended or not
*/
void printGPUInfo(
    SimInfo* info
);


#endif // __LBM_REPORT_H
