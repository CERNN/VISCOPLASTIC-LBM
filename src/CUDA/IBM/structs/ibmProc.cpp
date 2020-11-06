#include "ibmProc.h"

#ifdef IBM

IBMProc::ibmProc()
{
    this->macrCurr = nullptr;
    this->pCenter = nullptr;
    this->step = nullptr;
}

IBMProc::~ibmProc()
{
    this->macrCurr = nullptr;
    this->pCenter = nullptr;
    this->step = nullptr;
}

void IBMProc::allocateIBMProc()
{
    printf("allocate ibm data\n");
}

void IBMProc::freeIBMProc()
{
    printf("free ibm data\n");
}

void IBMProc::treatData()
{
    printf("treat ibm data\n");
}

bool IBMProc::stopSim()
{
    printf("stopping sim ibm\n");
    return false;
}

void IBMProc::printTreatData()
{
    printf("print treat ibm data\n");

}

void IBMProc::saveTreatData()
{
    printf("save ibm data\n");
}

#endif