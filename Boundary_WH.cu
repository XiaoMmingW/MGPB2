//
// Created by wxm on 2023/8/14.
//

#include "Boundary_WH.cuh"


__global__ void kernel_move_heat_source(int N, int Dim, real x0, real miu, real vs, real p0, real b0, real a,
                                        real *x, real *tbforce, real *dx)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<N)
    {
        real xx = fabs(x[i]-x0);
        if (Dim==2)
        {
            if (xx<a and fabs(x[i+N])<dx[i])
            {
                tbforce[i] = 0.5 * miu * vs * p0 * sqrt(1.0 - square(xx/a))/dx[i];
            }
        } else if (Dim==3)
        {
            if (fabs(x[i+N*2])<b0/2.0 and xx<a and fabs(x[i+N])<dx[i])
            {
                tbforce[i] = 0.5 * miu * vs * p0 * sqrt(1.0 - square(xx/a))/dx[i];
            }
        }

    }
}

__global__ void kernel_move_heat_source_hertz(int N, int Dim, real x0, real miu, real vs, real p0, real b0, real a,
                                        real *x, real *tbforce, real *dx)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<N)
    {
        real xx = fabs(x[i]-x0);
        if (Dim==2)
        {
            if (xx<a and fabs(x[i+N])<dx[i])
            {
                tbforce[i] = 0.5 * miu * vs * p0 * sqrt(1.0 - square(xx/a))/dx[i];
            }
        } else if (Dim==3)
        {
            real b = 0.5*b0;
            if (fabs(x[i+N*2])<b and xx<a and 1.0>square(xx/a)+ square(x[i+N*2]/b) and fabs(x[i+N])<dx[i])
            {
                tbforce[i] = 0.5 * miu * vs * p0 * sqrt(1.0 - square(xx/a)- square(x[i+N*2]/b))/dx[i];
            }
        }

    }
}

void move_heat_source(int GPUS, int Dim, real x0, real miu, real vs, real p0, real b0, real a, IHP_SIZE &ihpSize, Grid &grid,
                      State_Thermal_Diffusion_PD2 &pd, cudaStream_t *st_body)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.tbforce[i], 0, ihpSize.t_size[i]*sizeof(real)));
//        kernel_move_heat_source<<<grid.p_t[i], block_size, 0, st_body[i]>>>
//                (ihpSize.t_size[i], Dim, x0, miu, vs, p0, b0, a, pd.x[i], pd.tbforce[i], pd.dx[i]);
        kernel_move_heat_source_hertz<<<grid.p_t[i], block_size, 0, st_body[i]>>>
                (ihpSize.t_size[i], Dim, x0, miu, vs, p0, b0, a, pd.x[i], pd.tbforce[i], pd.dx[i]);
    }
    device_sync(GPUS);
}