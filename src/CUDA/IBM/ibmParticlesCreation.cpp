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

    /*
    // Falling sphere
    dfloat3 center;
    center.x = (NX-1.0)/2.0;
    center.y = (NY-1.0)/2.0;
    center.z = 3*(NZ-1.0)/4.0+PARTICLE_DIAMETER/2.0;
    particles[0] = makeSpherePolar(PARTICLE_DIAMETER, center , MESH_COULOMB, true);
    */

    // Fixed sphere
    particles[0] = makeSpherePolar(
        PARTICLE_DIAMETER, 
        dfloat3((NX-1.0)/2.0, (NY-1.0)/2.0, (NZ-1.0)/4.0), 
        MESH_COULOMB, false);
}

#endif // !IBM