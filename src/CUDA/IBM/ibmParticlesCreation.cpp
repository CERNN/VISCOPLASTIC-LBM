#include "ibmParticlesCreation.h"

#ifdef IBM

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

    return particles;
}

#endif // !IBM