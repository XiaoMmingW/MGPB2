//
// Created by wxm on 2023/8/12.
//

#include "Boundary_CT.cuh"

__global__ void kernel_load_CT(int N, int NT, real loc_x, real loc_y, real rad, real load, real *x, real *bforce)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<N)
    {
        if (square(x[i]-loc_x)+ square(x[i+NT]-loc_y)<=rad*rad)
        {
            bforce[i+NT] = load;
        } else if (square(x[i]-loc_x)+ square(x[i+NT]+loc_y)<=rad*rad)
        {
            bforce[i+NT] = -load;
        }
    }
}

void load_CT_GPU(int GPUS, real load, real width, real rad, IHP_SIZE &ihpSize, Grid &grid, Mech_PD &pd, cudaStream_t *st_body)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_load_CT<<<grid.p_t[i], block_size, 0, st_body[i]>>>(
                ihpSize.t_size[i], ihpSize.t_size[i], width, 0.55/2.0*width, rad, load, pd.x[i], pd.bforce[i]);
    }
    device_sync(GPUS);
}

