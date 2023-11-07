//
// Created by wxm on 2023/8/7.
//

#ifndef MULTI_GPU_FORCE_BOND_CUH
#define MULTI_GPU_FORCE_BOND_CUH
#include "Base.cuh"
#include "Base_Function.cuh"
#include "GPUS_Function.cuh"
void surfac_correct(int GPUS , real E, real pratio, int Dim, int MN, real thick, real sedload, int **exchange_flag, IHP_SIZE &ihpSize,
                    Grid &grid, Bond_Mech_PD &pd, Stream &st);
void force_gpu(int GPUS, real sc, int Dim, int MN, Bond_Mech_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid);
__global__  void kernel_bond_force_3D_atom(
        int N, int MN, real horizon,  real pi, real sc, int *NN, real *vol, int *fail, int *NL, real *idist, real *fac,
        real *x, real *disp, real *pforce, real *dx, real *scr, real *bc
);
#endif //MULTI_GPU_FORCE_BOND_CUH
