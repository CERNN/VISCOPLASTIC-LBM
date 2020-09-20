#ifdef IBM

#include "ibmParticleCreation.h"

Particle* createParticles()
{
    printf("-------------------------------- IBM INFORMATION -------------------------------\n");

    Particle *particles;
    dfloat3 bCenter[NUM_PARTICLES];

    particles = (Particle *)malloc(NUM_PARTICLES * sizeof(Particle));

    unsigned int totalIbmNodes = 0;

    int id = 0;

    for (int k = 0; k < 4; k++)
    {
        for (int j = 0; j < 5; j++)
        {
            for (int i = 0; i < 5; i++)
            {
                bCenter[id].x = 10.0 + (dfloat)i * 25.0 + ((dfloat)rand() / (RAND_MAX));
                bCenter[id].y = 10.0 + (dfloat)j * 25.0 + ((dfloat)rand() / (RAND_MAX));
                bCenter[id].z = 10.0 + (dfloat)k * 25.0 + ((dfloat)rand() / (RAND_MAX));

                id++;
            }
        }
    }

    printf("Creating particles: \n");
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        particles[p] = makeSpherePolar(PARTICLE_DIAMETER, bCenter[p], MESH_COULOMB, true);
    }

    Particle::updateParticleStaticVars(particles);
}

void updateParticleNodesSoA(Particle *particles)
{
    unsigned int totalIbmNodes = 0;

    // Determine the total number of nodes and the
    // max number of nodes in 1 particle
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        totalIbmNodes += particles[p].numNodes;
    }

    // Allocate on device
    checkCudaErrors(cudaMallocManaged((void **)&particleCenter, NUM_PARTICLES * sizeof(ParticleCenter)));

    printf("Total number of nodes: %u\n", totalIbmNodes);
    printf("Total memory used for Particles: %lu Mb\n",
           (unsigned long)((totalIbmNodes * sizeof(particleNode) + NUM_PARTICLES * sizeof(particleCenter)) / BYTES_PER_MB));
    fflush(stdout);

    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        for (int n = 0; n < particles[p].numNodes; n++)
        {
            nodePosition[numNodes[p] + n] = particles[p].node[n].pos;
            nodeVelocity[numNodes[p] + n] = particles[p].node[n].vel;
            nodeVelocityOld[numNodes[p] + n] = particles[p].node[n].vel_old;
            nodeForce[numNodes[p] + n] = particles[p].node[n].f;
            nodeCumulativeForce[numNodes[p] + n] = particles[p].node[n].cf;
            nodeSurface[numNodes[p] + n] = particles[p].node[n].S;

            //printf("%f - %f - %f \n",nodePosition[numNodes[p] + n].x,nodePosition[numNodes[p] + n].y,nodePosition[numNodes[p] + n].z);
        }
        particleCenter[p] = particles[p].bodyCenter;
    }

    printf("Particle positions :\n");
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        printf("%u -- x: %f -- y: %f -- z: %f \n",
               p, particleCenter[p].pos.x, particleCenter[p].pos.y, particleCenter[p].pos.z);
    }
    fflush(stdout);
}

#endif // !IBM