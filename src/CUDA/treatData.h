/*
*   @file treatData.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Data/macroscopics treatment
*   @version 0.3.0
*   @date 16/12/2019
*/

#ifndef __TREAT_DATA_H
#define __TREAT_DATA_H

#include "structs/macrProc.h"
#include "lbmReport.h" // for getVarFilename()
#include <cmath>


/*
*   @brief Treat data required by the struct MacrProc
*   @param processing: struct to be updated with treated values
*/
void treatData(MacrProc* processing, int step);


/*
*   @brief Stop simulation by conditions of treated data
*   @param processing: struct with treated data
*   @return true to stop, false to continue
*/
bool stopSim(MacrProc* processing);


/*
*   @brief Print desired values
*   @param processing: struct with treated values and 
*                      macroscopics required to print
*/
void printTreatData(MacrProc* processing);


/*
*   @brief Saves in a ".csv" file the required treated data
*   @param processing: struct with treated data
*/
void saveTreatData(MacrProc* processing);

#endif //!__TREAT_DATA_H