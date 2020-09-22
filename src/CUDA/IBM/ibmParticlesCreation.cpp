#include "ibmParticlesCreation.h"

#ifdef IBM

void createParticles(Particle particles[NUM_PARTICLES])
{
    printf("-------------------------------- IBM INFORMATION -------------------------------\n");

    dfloat3 bCenter[NUM_PARTICLES];
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
                if(id >= NUM_PARTICLES)
                    break;
            }
            if(id >= NUM_PARTICLES)
                break;
        }
        if(id >= NUM_PARTICLES)
            break;
    }

    printf("Creating particles...\t"); fflush(stdout);
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        particles[p] = makeSpherePolar(PARTICLE_DIAMETER, bCenter[p], MESH_COULOMB, true);
    }
    printf("Particles created!\n"); fflush(stdout);
}

#endif // !IBM