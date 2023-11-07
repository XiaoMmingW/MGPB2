//
// Created by wxm on 2023/6/19.
//

#ifndef MULTI_GPU_COORD_CUH
#define MULTI_GPU_COORD_CUH
#include "Base.cuh"
struct node_data
{
    int id;
    real x;
    real y;
    real z;
};

__global__ void kernel_coord_plate_crack(real *x, real *dxx, real *voll, int N, int nx, int ny, real dx, real vol);
void coord_plate(real *x, real *dxx, real *voll, int N, int nx, int ny, real dx, real vol);
void data_segm_2D(int N, int GPUS, int segm_num, real *k, real *b, real *x, real *dx, POINT_ID pid[], IHP_SIZE &ihpSize,
                  int **exchange_flag);
void coord_transfer(int N, int GPUS, real horizon, int Dim, real *x, real *dx, real *vol, POINT_ID pid1[],
                    IHP_SIZE &ihpSize, int **exchange_flag, Base_PD &pd);
void data_segm(int N, int GPUS, real horizon, real *sx, real *x, real *dx, POINT_ID pid[], IHP_SIZE &ihpSize,
               int **exchange_flag);
void coord_transfer2(int N, int GPUS, int Dim, real *x, real *dx, real *vol, POINT_ID pid1[],
                     IHP_SIZE &ihpSize, int **exchange_flag, Base_PD &pd, node_data** data_set);
void coord_transfer3(int N, int GPUS, int Dim, real *x, real *dx, real *vol, POINT_ID pid1[],
                     IHP_SIZE &ihpSize, int **exchange_flag, State_Thermal_Diffusion_PD2 &pd, node_data** data_set);
#endif //MULTI_GPU_COORD_CUH
