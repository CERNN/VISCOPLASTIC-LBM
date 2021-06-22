/*
*   @file ibmParticlesCreation.h
*   @author Waine Jr. (waine@alunos.utfpr.edu.br)
*   @brief Functions for creating IBM particles
*   @version 0.3.0
*   @date 26/08/2020
*/

#ifndef __IBM_PARTICLES_CREATION_H
#define __IBM_PARTICLES_CREATION_H

#include "structs/particle.h"
#include "ibmVar.h"
#include <random>

/**
*   @brief Create particles for simulation
*
*   @param particles: array with particles to write to
*/
void createParticles(Particle particles[NUM_PARTICLES]);

#endif // !__IBM_PARTICLES_CREATION_H