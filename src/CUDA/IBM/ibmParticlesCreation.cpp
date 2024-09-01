#include "ibmParticlesCreation.h"

#ifdef IBM

void createParticles(Particle *particles)
{

    dfloat3 *center;
    center = (dfloat3*) malloc(sizeof(dfloat3) * NUM_PARTICLES);

    dfloat3 vel, w;

    unsigned int totalIbmNodes = 0;

    center[0].x = (N-1)/2.0; center[0].y = (N-1)/2.0; center[0].z = NZ - (N-1)/2.0;

    vel.x = 0.0;
    vel.y = 0.0; //0.01*sin(angle*M_PI/180.0);
    vel.z = 0.0; //-0.01*cos(angle*M_PI/180.0);
        
    w.x = 0.0;
    w.y = 0.0;
    w.z = 0.0;

    dfloat length = 10;
    dfloat3 center1;
    dfloat3 center2;

    center1.x = 0;
    center1.y = 0; 
    center1.z = 0;

    center2.x = 30;
    center2.y = 30; 
    center2.z = 30;


    for(int i = 0; i <NUM_PARTICLES ; i++){
        //particles[i].makeSpherePolar(PARTICLE_DIAMETER, center[i], MESH_COULOMB, true, PARTICLE_DENSITY, vel, w);
        particles[i].makeEllipsoid(dfloat3(40.0,20.0,10.0), center[i], dfloat3(0.5,1.0,0.6), 0.3*M_PI/4,true, PARTICLE_DENSITY, vel, w);
        //particles[i].makeCapsule(PARTICLE_DIAMETER, center1, center2, true,PARTICLE_DENSITY, vel, w);
    }
}
#endif // !IBM