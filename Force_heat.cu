//
// Created by wxm on 2023/8/13.
//

#include "Force_heat.cuh"

__global__ void kernel_initial_weight_heat_2D(
        int N, int NT, int MN, real horizon, int begin, int *NN, int *NL, real *m, real *dx, real *vol, real *x)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;

    real temp_m = 0.0;
    if (i<N)
    {
        i += begin;
        if (j<NN[i])
        {
            idx = i*MN+j;
            int cnode = NL[idx];
            real delta = horizon*dx[cnode];
            real idist = sqrt(square(x[cnode]-x[i])+square(x[cnode+NT]-x[i+NT]));
            real fac;
            if (idist <= delta-dx[cnode]/2.0)
                fac = 1.0;
            else if (idist <= delta+dx[cnode]/2.0)
                fac = (delta+dx[cnode]/2.0-idist) / dx[cnode];
            else
                fac = 0.0;
            real w = exp(-square(idist) / square(delta));
            temp_m = w * square(idist) * vol[cnode] * fac;
            //if (cnode==0 and i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode);
        }
        for (int offset=16; offset>0; offset>>=1)
            temp_m += __shfl_down_sync(FULL_MASK, temp_m, offset);
        if (j==0)
            m[i] = temp_m;
    }
}

__global__ void kernel_initial_weight_heat_3D(
        int N, int NT, int MN, real horizon, int begin, int *NN, int *NL, real *m, real *dx, real *vol, real *x)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    real temp_m = 0.0;
    __shared__ real mm[128];
    if (i<N)
    {
        i += begin;
        if (j<NN[i])
        {
            idx = i*MN+j;
            int cnode = NL[idx];
            real delta = horizon*dx[cnode];
            real idist = sqrt(square(x[cnode]-x[i])+square(x[cnode+NT]-x[i+NT])+square(x[cnode+NT*2]-x[i+NT*2]));
            real fac;
            if (idist <= delta-dx[cnode]/2.0)
                fac = 1.0;
            else if (idist <= delta+dx[cnode]/2.0)
                fac = (delta+dx[cnode]/2.0-idist) / dx[cnode];
            else
                fac = 0.0;
            real w = exp(-square(idist) / square(delta));
            temp_m = w * square(idist) * vol[cnode] * fac;
            //if (cnode==0 and i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode);
        }
        for (int offset=16; offset>0; offset>>=1)
            temp_m += __shfl_down_sync(FULL_MASK, temp_m, offset);
        if(j%32==0)
            mm[j] = temp_m;
        __syncthreads();
        if (j==0)
            m[i] = mm[0]+mm[32]+mm[64]+mm[96];
    }
}

void cal_weight_heat_gpu(int GPUS,  int Dim, int MN, int **exchange_flag, IHP_SIZE &ihpSize,
                    Grid &grid, Stream &st, State_Thermal_Diffusion_PD2 &pd)
{
    long double start = cpuSecond();
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        if (Dim==2)
        {
            kernel_initial_weight_heat_2D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
                    ihpSize.i_size[i], ihpSize.t_size[i], MN, horizon, 0, pd.NN[i], pd.NL[i], pd.m[i], pd.dx[i],
                    pd.vol[i], pd.x[i]);
            for (int j = 0; j < GPUS; j++) {
                if (exchange_flag[i][j] == 1) {
                    kernel_initial_weight_heat_2D<<<grid.b_h[i][j], block_size, 0, st.halo[i][j]>>>(
                            ihpSize.h_size[i][j], ihpSize.t_size[i], MN, horizon, ihpSize.h_begin[i][j], pd.NN[i], pd.NL[i], pd.m[i], pd.dx[i],
                            pd.vol[i], pd.x[i]);
                    CHECK(cudaMemcpyAsync(&(pd.m[j])[ihpSize.p_begin[j][i]],
                                          &(pd.m[i])[ihpSize.h_begin[i][j]],
                                          ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                          st.halo[i][j]));
                }
            }
        } else if (Dim==3)
        {
            kernel_initial_weight_heat_3D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
                    ihpSize.i_size[i], ihpSize.t_size[i], MN, horizon, 0, pd.NN[i], pd.NL[i], pd.m[i], pd.dx[i],
                    pd.vol[i], pd.x[i]);
            for (int j = 0; j < GPUS; j++) {
                if (exchange_flag[i][j] == 1) {
                    kernel_initial_weight_heat_3D<<<grid.b_h[i][j], block_size, 0, st.halo[i][j]>>>(
                            ihpSize.h_size[i][j], ihpSize.t_size[i], MN, horizon, ihpSize.h_begin[i][j], pd.NN[i], pd.NL[i], pd.m[i], pd.dx[i],
                            pd.vol[i], pd.x[i]);
                    CHECK(cudaMemcpyAsync(&(pd.m[j])[ihpSize.p_begin[j][i]],
                                          &(pd.m[i])[ihpSize.h_begin[i][j]],
                                          ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                          st.halo[i][j]));
                }
            }
        }
    }
    device_sync(GPUS);
    cout<<"weight time: "<<(cpuSecond()-start)*1000<<endl;

}



__global__ void Temperature(int N, int Dim, real *T, real *x)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N)
    {
        for (int j=0; j<Dim; j++) T[i] += x[i+j*N];
    }
}


__global__ void Surface_T_State_2D(
        int N, int MN, real k, real horizon, int *NN, int *NL,  real *T, real *tfncst, real *vol,
        real *m, real *x, real *dx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    int  cnode = 0;
    real tempdens = 0.0;
    real tempdens_i = 0.0;
    real tempdens_j = 0.0;
    real flux = 0;
    if(i<N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            real delta = horizon*dx[cnode];
            real idist = sqrt(square(x[cnode]-x[i])+square(x[cnode+N]-x[i+N]));
            real fac;
            if (idist <= delta-dx[cnode]/2.0)
                fac = 1.0;
            else if (idist <= delta+dx[cnode]/2.0)
                fac = (delta+dx[cnode]/2.0-idist) / dx[cnode];
            else
                fac = 0.0;
            real w = exp(-square(idist) / square(delta));
            flux = k * (T[cnode] - T[i]) / m[i] * w;
            tempdens =  0.25 * flux * (T[cnode] - T[i])  * fac;
            tempdens_i = tempdens * vol[cnode];
            tempdens_j = tempdens * vol[i];
            //tempdens_Cal += 0.25*k*gsquare(T[cnode]-T[i])/idist[idx]*vol[j]*fac[idx];
            atomicAdd(&tfncst[cnode], tempdens_j);
            //if (cnode==0 or i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode, tempdens_i, tempdens_j, m[i]);
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            tempdens_i += __shfl_down_sync(FULL_MASK, tempdens_i, offset);
        }
        __syncthreads();
        if (j==0)
        {
            atomicAdd(&tfncst[i], tempdens_i);
            //if (i==0) printf("i %d cnode %d ff %e %e\n", i, j, tfncst[i], f[0]+f[32]+f[64]+f[96]);
        }
    }
}

__global__ void Surface_T_State_3D(
        int N, int MN, real k, real horizon, int *NN, int *NL,  real *T, real *tfncst, real *vol,
        real *m,  real *x, real *dx)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    int  cnode = 0;
    real tempdens = 0.0;
    real tempdens_i = 0.0;
    real tempdens_j = 0.0;

    real flux = 0;
    __shared__ real f[128];
    if(i<N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            real delta = horizon*dx[cnode];
            real idist = sqrt(square(x[cnode]-x[i])+square(x[cnode+N]-x[i+N]));
            real fac;
            if (idist <= delta-dx[cnode]/2.0)
                fac = 1.0;
            else if (idist <= delta+dx[cnode]/2.0)
                fac = (delta+dx[cnode]/2.0-idist) / dx[cnode];
            else
                fac = 0.0;
            real w = exp(-square(idist) / square(delta));
            flux = k * (T[cnode] - T[i]) / m[i] * w;
            tempdens =  0.25 * flux * (T[cnode] - T[i])  * fac;
            tempdens_i = tempdens * vol[cnode];
            tempdens_j = tempdens * vol[i];
            //tempdens_Cal += 0.25*k*gsquare(T[cnode]-T[i])/idist[idx]*vol[j]*fac[idx];
            atomicAdd(&tfncst[cnode], tempdens_j);
            //if (cnode==0 or i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode, tempdens_i, tempdens_j, m[i]);
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            tempdens_i += __shfl_down_sync(FULL_MASK, tempdens_i, offset);

        }
        if(j%32==0)
        {
            f[j] = tempdens_i;
        }
        __syncthreads();
        if (j==0)
        {
            atomicAdd(&tfncst[i], f[0]+f[32]+f[64]+f[96]);
            //if (i==0) printf("i %d cnode %d ff %e %e\n", i, j, tfncst[i], f[0]+f[32]+f[64]+f[96]);
        }
    }
}

static __global__ void cal_tfncst(int N, real tempload, int begin, real* tfncst)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        i += begin;
        tfncst[i] = tempload / tfncst[i];
        if (i==begin) printf("i %d %d ff %e \n", i, N, tfncst[i]);
    }
}



__global__ void T_corr(
        int N, int MN, int *NN, int *NL, real *tfncst, real *scr)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;

    if (i<N)
    {
        if (j<NN[i])
        {
            scr[idx] = (tfncst[i] + tfncst[NL[idx]])/2.0;
            if (i==0) printf("scr %e\n", scr[idx]);
        }
    }
}

void temp_surfac_correct_state(int GPUS, int MN, real k, real tempload, int Dim, int **exchange_flag, IHP_SIZE &ihpSize,
                               Grid &grid, State_Thermal_Diffusion_PD2 &pd, Stream &st)
{

    long double strat = cpuSecond();
//#pragma omp parallel for
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        Temperature<<<grid.p_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], Dim, pd.T[i], pd.x[i]);
        if (Dim==2)
        {
            Surface_T_State_2D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i],  MN, k, horizon, pd.NN[i], pd.NL[i], pd.T[i], pd.fncst[i], pd.vol[i],
                     pd.m[i],  pd.x[i], pd.dx[i]);
        } else if (Dim==3) {
            Surface_T_State_3D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], MN, k, horizon, pd.NN[i], pd.NL[i], pd.T[i], pd.fncst[i], pd.vol[i],
                     pd.m[i],  pd.x[i], pd.dx[i]);
        }
        CHECK(cudaMemset(pd.T[i], 0, ihpSize.t_size[i]*sizeof(real)));

    }
    device_sync(GPUS);
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        cal_tfncst<<<grid.p_i[i], block_size, 0, st.body[i]>>>(
                ihpSize.i_size[i], tempload, 0, pd.fncst[i]);
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                cal_tfncst<<<grid.p_h[i][j], block_size, 0, st.halo[i][j]>>>
                        ( ihpSize.h_size[i][j],  tempload, ihpSize.h_begin[i][j], pd.fncst[i]);
                CHECK(cudaMemcpyAsync(&(pd.fncst[j])[ihpSize.p_begin[j][i]],
                                      &(pd.fncst[i])[ihpSize.h_begin[i][j]],
                                      ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                      st.halo[i][j]));
            }
        }
    }
    device_sync(GPUS);
    cout<<"cost time: "<<cpuSecond()-strat<<endl;
}

__global__ void kernel_heat_conduction_state_2D
        (int N, int MN,  real k, real horizon, real *T, int *NN, int *NL, real *energy,  real *vol,
         real *m, real *x, real *dx, real *fncst
        )
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    real flux = 0.0;
    real flux_i = 0.0;
    real flux_j = 0.0;
    if (i<N)
    {
        if(j<NN[i])
        {
            int cnode = NL[idx];
            //flux = k*(T[cnode]-T[i])/m[i]*w[idx]*fac[idx];
            real delta = horizon*dx[cnode];
            real idist = sqrt(square(x[cnode]-x[i])+square(x[cnode+N]-x[i+N]));
            real fac;
            if (idist <= delta-dx[cnode]/2.0)
                fac = 1.0;
            else if (idist <= delta+dx[cnode]/2.0)
                fac = (delta+dx[cnode]/2.0-idist) / dx[cnode];
            else
                fac = 0.0;
            real w = exp(-square(idist) / square(delta));
            real scr = (fncst[i]+fncst[cnode])/2.0;
            flux = k*scr*(T[cnode]-T[i])/m[i]*w*fac;
            flux_i = flux*vol[cnode];
            flux_j = -flux*vol[i];
            atomicAdd(&energy[cnode], flux_j);
            //if (cnode==0 and i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode);
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            flux_i += __shfl_down_sync(FULL_MASK, flux_i, offset);
        }
        __syncthreads();
        if (j==0) {
            atomicAdd(&energy[i], flux_i);
        }
    }
}


__global__ void kernel_heat_conduction_state_3D
        (int N, int MN,  real k, real horizon, real *T, int *NN, int *NL, real *energy, real *vol,
         real *m, real *x, real *dx, real *fncst
        )
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    real flux = 0.0;
    real flux_i = 0.0;
    real flux_j = 0.0;
    __shared__ real fx[128];
    if (i<N)
    {
        if(j<NN[i])
        {
            int cnode = NL[idx];
            //flux = k*(T[cnode]-T[i])/m[i]*w[idx]*fac[idx];
            real delta = horizon*dx[cnode];
            real idist = sqrt(square(x[cnode]-x[i])+square(x[cnode+N]-x[i+N]));
            real fac;
            if (idist <= delta-dx[cnode]/2.0)
                fac = 1.0;
            else if (idist <= delta+dx[cnode]/2.0)
                fac = (delta+dx[cnode]/2.0-idist) / dx[cnode];
            else
                fac = 0.0;
            real w = exp(-square(idist) / square(delta));
            real scr = (fncst[i]+fncst[cnode])/2.0;
            flux = k*scr*(T[cnode]-T[i])/m[i]*w*fac;
            flux_i = flux*vol[cnode];
            flux_j = -flux*vol[i];
            atomicAdd(&energy[cnode], flux_j);
            //if (cnode==0 and i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode);
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            flux_i += __shfl_down_sync(FULL_MASK, flux_i, offset);
        }

        if(j%32==0) {
            fx[j] = flux_i;
        }
        __syncthreads();
        if (j==0) {
            atomicAdd(&energy[i], fx[0] + fx[32] + fx[64] + fx[96]);
        }
    }
}

void thermal_diffusion_gpu(int GPUS, int Dim, int MN, real k, State_Thermal_Diffusion_PD2 &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.energy[i], 0, ihpSize.t_size[i]*sizeof(real)));
        if (Dim==2) {
            kernel_heat_conduction_state_2D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], MN, k, horizon, pd.T[i], pd.NN[i], pd.NL[i], pd.energy[i],
                     pd.vol[i], pd.m[i], pd.x[i], pd.dx[i], pd.fncst[i]
            );
        } else if (Dim==3)
        {
            kernel_heat_conduction_state_3D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], MN, k, horizon, pd.T[i], pd.NN[i], pd.NL[i], pd.energy[i],
                     pd.vol[i], pd.m[i], pd.x[i], pd.dx[i], pd.fncst[i]
            );
        }
    }
    device_sync(GPUS);
}

