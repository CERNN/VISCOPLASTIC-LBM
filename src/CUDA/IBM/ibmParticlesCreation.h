
#ifndef __IBM_PARTICLES_CREATION_H
#define __IBM_PARTICLES_CREATION_H

#include "structs/particle.h"

Particle* createParticles();

void updateParticleNodesSoA(Particle* particles, unsigned int numParticles);

#endif // !__IBM_PARTICLES_CREATION_H