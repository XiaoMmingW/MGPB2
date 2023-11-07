//
// Created by wxm on 2023/6/19.
//

#include "Boundary.cuh"


__global__ void load_vel(real *disp, real *x, real ct, int N, real vel_load, real load_area)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N)
    {
        if (x[i+N]>load_area)
        {
            disp[i+N] = ct*vel_load;
        } else if (x[i+N]<-load_area)
        {
            disp[i+N] = -ct*vel_load;
        }
    }
}



__global__ void load_vel_new(real *disp, real *x, real ct, int N, int NT, real vel_load, real load_area)
{
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N)
    {
        if (x[i+NT]>load_area)
        {
            disp[i+NT] = ct*vel_load;
        } else if (x[i+NT]<-load_area)
        {
            disp[i+NT] = -ct*vel_load;
        }
    }
}

void load_vel(int GPUS,  real ct, real vel_load, real load_area, IHP_SIZE &ihpSize, Grid &grid, Mech_PD &pd, Stream &st)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        load_vel_new<<<grid.p_t[i], block_size, 0, st.body[i]>>>
                (pd.disp[i], pd.x[i], ct, ihpSize.t_size[i], ihpSize.t_size[i], vel_load, load_area);
    }
}
