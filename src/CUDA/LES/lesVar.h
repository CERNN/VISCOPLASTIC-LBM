/*
*   @file lesVar.h
*   @author Marco Aurelio Ferrari (marcoferrari@alunos.utfpr.edu.br)
*   @brief Configurations for the LES
*   @version 0.1.0
*   @date 07/06/2022
*/

#ifndef __LES_VAR_H
#define __LES_VAR_H

//MODEL TYPE
#define MODEL_CONST_SMAGORINSKY


//#define LES_EXPORT_VISC_TURBULENT

//MODEL DEFINITIONS
#ifdef MODEL_CONST_SMAGORINSKY
constexpr dfloat CONST_SMAGORINSKY = 0.1;
constexpr dfloat INIT_VISC_TURB = 0.0;


constexpr dfloat Implicit_const = 2.0*SQRT_2*3*3/(RHO_0)*CONST_SMAGORINSKY*CONST_SMAGORINSKY;

#endif


#endif // !__LES_VAR_H
