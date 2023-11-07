//
// Created by wxm on 2023/8/7.
//

#ifndef MULTI_GPU_FORCE_STATE_CUH
#define MULTI_GPU_FORCE_STATE_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
void cal_weight_gpu(int GPUS,  int Dim, int MN, int **exchange_flag, IHP_SIZE &ihpSize,
                    Grid &grid, Stream &st, State_PD &spd, Base_PD &pd);
void force_state_gpu(int GPUS, real E, real pratio, real sc, int Dim, int MN, real thick, State_Mech_PD &pd, IHP_SIZE &ihpSize,
                     Stream &st, Grid &grid, int **exchange_flag);
void force_state_gpu1(int GPUS, real E, real pratio, real sc, int Dim, int MN, real thick, State_Mech_PD &pd, IHP_SIZE &ihpSize,
                     Stream &st, Grid &grid, int **exchange_flag);
void force_state_gpu2(int GPUS, real E, real pratio, real sc, int Dim, int MN, real thick, State_Mech_PD &pd, IHP_SIZE &ihpSize,
                     Stream &st, Grid &grid, int **exchange_flag);
__global__  void kernel_state_force_2D(
        int N, real K, real G, int MN, real horizon, real thick, real pi, real sc, int *NN, real *m, real *theta,
        real *vol, int *fail, int *NL, real *w, real *idist, real *fac, real *x, real *disp, real *pforce, real *dx
);
__global__ void kernel_cal_theta_2D(
        int N, int NT, int MN, int begin, int *NN, real *m, real *vol,real *theta, int *NL, real *w, int *fail,
        real *idist, real *fac, real *x, real *disp);
#endif //MULTI_GPU_FORCE_STATE_CUH
