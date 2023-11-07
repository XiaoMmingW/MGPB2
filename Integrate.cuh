//
// Created by wxm on 2023/6/20.
//

#ifndef MULTI_GPU_INTEGRATE_CUH
#define MULTI_GPU_INTEGRATE_CUH
#include "Base.cuh"
#include "GPUS_Function.cuh"
#include "Base_Function.cuh"

void integrate_CD(int GPUS , real dens, real dt, int Dim, int **exchange_flag, IHP_SIZE &ihpSize,
                  Grid &grid, Mech_PD &pd, Stream &st);
void integrate_CD_contact(int GPUS , real dens, real dt, int Dim, int **exchange_flag, IHP_SIZE &ihpSize,
                          Grid &grid, Mech_PD &pd, Stream &st);
void static_integrate(int GPUS, int Dim, real dt, int ct, int **exchange_flag, State_Fatigue_PD &pd, IHP_SIZE &ihpSize,
                      Stream &st, Grid &grid);
void integrate_T_GPU(int GPUS , real dens, real cv, real dt, int **exchange_flag, IHP_SIZE &ihpSize,
                     Grid &grid, State_Thermal_Diffusion_PD2 &pd, Stream &st);
void integrate_CD_cpu(int N, real dens, real dt,  int Dim, Bond_PD_CPU &pd, int omp);
void integrate_CD_cpu(int N, real dens, real dt, int Dim, State_PD_CPU &pd, int omp);

#endif //MULTI_GPU_INTEGRATE_CUH
