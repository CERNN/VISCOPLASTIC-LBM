#include "particleNode.h"
#include "particle.h"

ParticleNodeSoA::particleNodeSoA()
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
}

ParticleNodeSoA::~particleNodeSoA()
{
    this->S = nullptr;
    this->particleCenterIdx = nullptr;
}

#ifdef IBM
void ParticleNodeSoA::allocateMemory(unsigned int numNodes)
{
    this->numNodes = numNodes;

    this->pos.allocateMemory((size_t) numNodes);
    this->vel.allocateMemory((size_t) numNodes);
    this->vel_old.allocateMemory((size_t) numNodes);
    this->f.allocateMemory((size_t) numNodes);
    this->deltaF.allocateMemory((size_t) numNodes);

    checkCudaErrors(
        cudaMallocManaged((void**)&(this->S), sizeof(dfloat) * numNodes));
    checkCudaErrors(
        cudaMallocManaged((void**)&(this->particleCenterIdx), sizeof(unsigned int) * numNodes));
}

void ParticleNodeSoA::freeMemory()
{
    this->numNodes = 0;

    this->pos.freeMemory();
    this->vel.freeMemory();
    this->vel_old.freeMemory();
    this->f.freeMemory();
    this->deltaF.freeMemory();

    cudaFree(this->S);
    cudaFree(this->particleCenterIdx);
}

void ParticleNodeSoA::copyNodesFromParticle(Particle p, unsigned int pCenterIdx, unsigned int baseIdx)
{
    for (int i = 0; i < p.numNodes; i++)
    {
        this->particleCenterIdx[i+baseIdx] = pCenterIdx;

        this->pos.copyValuesFromdFloat3(p.nodes[i].pos, i+baseIdx);
        this->vel.copyValuesFromdFloat3(p.nodes[i].vel, i+baseIdx);
        this->vel_old.copyValuesFromdFloat3(p.nodes[i].vel_old, i+baseIdx);
        this->f.copyValuesFromdFloat3(p.nodes[i].f, i+baseIdx);
        this->deltaF.copyValuesFromdFloat3(p.nodes[i].deltaF, i+baseIdx);
        this->S[i+baseIdx] = p.nodes[i].S;
    }
}

#endif // !IBM