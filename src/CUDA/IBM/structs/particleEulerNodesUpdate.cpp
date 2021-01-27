#include "particleEulerNodesUpdate.h"

ParticleEulerNodesUpdate::particleEulerNodesUpdate(){
    #if IBM_EULER_OPTIMIZATION
    eulerMaskArray = (uint32_t*) malloc((size_t)NX*NY*NZ*sizeof(uint32_t));
    eulerIndexesUpdate = nullptr;
    nEulerNodes = 0;
    currEulerNodes = 0;

    pCenterMovable = nullptr;
    particlesLastPos = nullptr;
    nParticlesMovable = 0;
    #endif
}

ParticleEulerNodesUpdate::~particleEulerNodesUpdate(){
    #if IBM_EULER_OPTIMIZATION
    free(eulerMaskArray);
    #endif
}

#if IBM_EULER_OPTIMIZATION
    
__host__
void ParticleEulerNodesUpdate::initializeEulerNodes(ParticleCenter p[NUM_PARTICLES]){

}

__host__
void ParticleEulerNodesUpdate::checkParticlesMovement(){
    
}

__host__
void ParticleEulerNodesUpdate::removeUnneededEulerNodes(){
    
}

__host__
void ParticleEulerNodesUpdate::updateEulerNodes(ParticleCenter p, uint32_t mask){
    
}

__global__
void ibmEulerCopyVelocities(dfloat3SoA dst, dfloat3SoA src, size_t* eulerIdxsUpdate, unsigned int currEulerNodes){

}


#endif