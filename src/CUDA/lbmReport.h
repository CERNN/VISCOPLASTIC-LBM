/*
*   @file lbmReport.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Report (save and print) simulation information
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __LBM_REPORT_H
#define __LBM_REPORT_H

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision
#include <cuda.h>
#include "globalFunctions.h"
#include "errorDef.h"
#include "structs/macroscopics.h"
#include "structs/populations.h"
#include "structs/simInfo.h"
#include "IBM/ibmVar.h"
#include "LES/lesVar.h"


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
*   @param append: content must be appended to file or overwrite
*/
void saveVarBin(
    std::string strFile, 
    dfloat* var, 
    size_t memSize,
    bool append
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
*   Get string with simulation information
*   @param info: simulation's informations
*   @return string with simulation info
*/
std::string getSimInfoString(SimInfo* info);

/*
*   Save simulation's information
*   @param info: simulation's informations
*/
void saveSimInfo(
    SimInfo* info
);


/*
*   Print simulation information
*   @param info: simulation's informations
*/
void printSimInfo(
    SimInfo* info
);


#endif // __LBM_REPORT_H
