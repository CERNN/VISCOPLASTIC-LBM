#include "ibmParticlesCreation.h"

#ifdef IBM

void createParticles(Particle particles[NUM_PARTICLES])
{
    dfloat3 bCenter[NUM_PARTICLES];
    unsigned int totalIbmNodes = 0;

    int id = 0;
    /*
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
    */
    for (int p = 0; p < NUM_PARTICLES; p++)
    {
        dfloat3 center;
        center.x = (NX-1.0)/2.0;
        center.y = (NY-1.0)/2.0;
        center.z = (NZ-1.0)*0.796875;
        particles[p] = makeSpherePolar(PARTICLE_DIAMETER, center , MESH_COULOMB, true);
    }
    /*
    particles[0] = makeSpherePolar(
        PARTICLE_DIAMETER, 
        dfloat3((NX-1.0)/2.0, (NY-1.0)/2.0, (NZ-1.0)/2.0), 
        MESH_COULOMB, true);
    */
}

#endif // !IBM