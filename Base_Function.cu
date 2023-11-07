//
// Created by wxm on 2023/7/26.
//

#include "Base_Function.cuh"

__device__ real atomicAdd2(real* address, real val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
//计时函数，以微秒返回当前时间
long double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (tp.tv_sec*1.0e6+(long double)tp.tv_usec)/1.0e6;
}

__global__ void kernel_initial_fail(int N, int MN, int *fail)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx<N*MN) fail[idx] = 1;
}



__global__ void kernel_cal_dmg_2D(int N, int MN, int begin, int *NN,real *dmg, real *vol, int *NL,int *fail, real *fac)
{
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;

    if (i<N)
    {
        real dmgpar1 = 0.0;
        real dmgpar2 = 0.0;
        i += begin;
        if (j<NN[i])
        {
            idx = i*MN+j;
            int cnode = NL[idx];
            dmgpar1 = fail[idx]*vol[cnode]*fac[idx];
            dmgpar2 = vol[cnode] * fac[idx];
        }

        for (int offset=16; offset>0; offset>>=1)
        {
            dmgpar1 += __shfl_down_sync(FULL_MASK, dmgpar1, offset);
            dmgpar2 += __shfl_down_sync(FULL_MASK, dmgpar2, offset);
        }
        if (j==0) dmg[i] = 1.0 - dmgpar1/dmgpar2;
    }
}

__global__ void kernel_cal_dmg_3D(int N, int MN, int *NN,real *dmg, real *vol, int *NL,int *fail, real *fac)
{
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;

    __shared__ real d1[128];
    __shared__ real d2[128];
    if (i<N)
    {
        real dmgpar1 = 0.0;
        real dmgpar2 = 0.0;

        if (j<NN[i])
        {
            int cnode = NL[idx];
            dmgpar1 = fail[idx]*vol[cnode]*fac[idx];
            dmgpar2 = vol[cnode] * fac[idx];
        }

        for (int offset=16; offset>0; offset>>=1)
        {
            dmgpar1 += __shfl_down_sync(FULL_MASK, dmgpar1, offset);
            dmgpar2 += __shfl_down_sync(FULL_MASK, dmgpar2, offset);
        }
        if(j%32==0)
        {
            d1[j] = dmgpar1;
            d2[j] = dmgpar2;
        }
        __syncthreads();
        if (j==0)
        {
            dmg[i] = 1.0 - (d1[0]+d1[32]+d1[64]+d1[96])/(d2[0]+d2[32]+d2[64]+d2[96]);
        }
    }
}

void cal_dmg_gpu(int GPUS, int MN, int Dim, IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd, Stream &st, int **exchange_flag)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        if (Dim==2) {
            kernel_cal_dmg_2D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
                    ihpSize.i_size[i], MN, 0, pd.NN[i], pd.dmg[i], pd.vol[i], pd.NL[i], pd.fail[i], pd.fac[i]);
            for (int j = 0; j < GPUS; j++) {
                if (exchange_flag[i][j] == 1) {
                    kernel_cal_dmg_2D<<<grid.b_h[i][j], block_size, 0, st.halo[i][j]>>>(
                            ihpSize.h_size[i][j], MN, ihpSize.h_begin[i][j], pd.NN[i], pd.dmg[i], pd.vol[i], pd.NL[i], pd.fail[i], pd.fac[i]);
                    CHECK(cudaMemcpyAsync(&(pd.dmg[j])[ihpSize.p_begin[j][i]],
                                          &(pd.dmg[i])[ihpSize.h_begin[i][j]],
                                          ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                          st.halo[i][j]));
                }
            }

        } else if (Dim==3)
        {
            kernel_cal_dmg_3D<<<grid.b_ih[i], block_size, 0, st.body[i]>>>(
                    ihpSize.ih_size[i], MN, pd.NN[i], pd.dmg[i], pd.vol[i], pd.NL[i], pd.fail[i], pd.fac[i]);
        }
    }
    device_sync(GPUS);
}

void cal_dmg_cpu(int N, int MN, Bond_PD_CPU &pd, int omp) {
    int cnode = 0;
    real d1 = 0.0;
    real d2 = 0.0;
    int idx = 0;
    if (omp == 0) {
        for (int i = 0; i < N; i++) {
            d1 = 0.0;
            d2 = 0.0;
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                d1 += pd.fail[idx] * pd.vol[cnode] * pd.fac[idx];
                d2 += pd.vol[cnode] * pd.fac[idx];
                //if (i==0) cout<<"fail "<<pd.fail[idx]<<" d1 "<<d1<<" d2 "<<d2<<endl;
            }
            pd.dmg[i] = 1.0-d1/d2;
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx, d1, d2)
        for (int i = 0; i < N; i++) {
            d1 = 0.0;
            d2 = 0.0;
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                d1 += pd.fail[idx] * pd.vol[cnode] * pd.fac[idx];
                d2 += pd.vol[cnode] * pd.fac[idx];
            }
            pd.dmg[i] = 1.0-d1/d2;
        }
    }
}

void cal_dmg_cpu(int N, int MN, State_PD_CPU &pd, int omp) {
    int cnode = 0;
    real d1 = 0.0;
    real d2 = 0.0;
    int idx = 0;
    if (omp == 0) {
        for (int i = 0; i < N; i++) {
            d1 = 0.0;
            d2 = 0.0;
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                d1 += pd.fail[idx] * pd.vol[cnode] * pd.fac[idx];
                d2 += pd.vol[cnode] * pd.fac[idx];
                //if (i==0) cout<<"fail "<<pd.fail[idx]<<" d1 "<<d1<<" d2 "<<d2<<endl;
            }
            pd.dmg[i] = 1.0-d1/d2;
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx, d1, d2)
        for (int i = 0; i < N; i++) {
            d1 = 0.0;
            d2 = 0.0;
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                d1 += pd.fail[idx] * pd.vol[cnode] * pd.fac[idx];
                d2 += pd.vol[cnode] * pd.fac[idx];
            }
            pd.dmg[i] = 1.0-d1/d2;
        }
    }
}
__global__ void kernel_set_crack(
        int N, int MN, real pi, real clength, real loc_x, real loc_y,real theta, int rint, int *g_NN, int *g_fail,
        int *g_NL, real *x)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i<N)
    {
        int cnode = 0;
        for (int j=0; j<g_NN[i]; j++)
        {
            cnode = g_NL[i*MN+j];

            if(fabs(theta-pi/2.0)<0.0001)
            {
                if(fabs(x[cnode+rint]-loc_y) < clength*sin(theta)/2.0 && fabs(x[i+rint]-loc_y) < clength*sin(theta)/2.0)
                {
                    if((x[cnode]-loc_x)*(x[i]-loc_x) < 0.0 )
                    {
                        g_fail[i*MN+j] = 0;
                        if ((x[i+rint]+x[cnode+rint])/2.0<loc_y*1.5)
                        {
//                            real dist =sqrt(gsquare(x[cnode]-x[i]) + gsquare(x[rint+cnode]-x[rint+i]));
//                            real angel = atan((x[cnode+rint]-x[i+rint])/(x[cnode]-x[i]));
//                            printf(" %f %f %f %f %d %d %f %f \n", x[i], x[i+rint], x[cnode], x[cnode+rint], i, cnode, angel/pi, dist);
                        }
                    }
                }
            }
            else
            {
                real dist_cnode = square(x[cnode+rint]-loc_y)+ square(x[cnode]-loc_x);
                real dist_i = square(x[i+rint]-loc_y)+ square(x[i]-loc_x);
                real vertical_dcnode = square((tan(theta)*x[cnode]-x[cnode+rint]-loc_x*tan(theta)+loc_y) / sqrt(1+ tan(theta)*tan(theta)));
                real vertical_di = square((tan(theta)*x[i]-x[i+rint]-loc_x*tan(theta)+loc_y) / sqrt(1+ tan(theta)*tan(theta)));
                real length_cnode = sqrt(dist_cnode - vertical_dcnode);
                real length_i = sqrt(dist_i - vertical_di);
                if(length_cnode<clength/2.0 && length_i<clength/2.0)
                {
                    real bcnode= tan(theta)*x[cnode]-loc_x*tan(theta)+loc_y - x[cnode+rint];
                    real bi= tan(theta)*x[i]-loc_x*tan(theta)+loc_y - x[i+rint];
                    if(bcnode * bi<=0)
                    {
                        g_fail[i*MN+j] = 0;
                        if ((x[i+rint]+x[cnode+rint])/2.0<loc_y*1.7)
                        {
//                            real dist =sqrt(gsquare(x[cnode]-x[i]) + gsquare(x[rint+cnode]-x[rint+i]));
//                                    real angel = atan((x[cnode+rint]-x[i+rint])/(x[cnode]-x[i]));
//                                    printf(" %f %f %f %f %d %d %f %f %f %f \n", x[i], x[i+rint], x[cnode], x[cnode+rint], i,
//                                           cnode, angel/pi, dist, vertical_di,  vertical_dcnode);
                        }
                    }
                }
            }
        }
    }
}

void set_crack_2D(int GPUS, int MN, real clength, real loc_x, real loc_y, real theta, IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd, Stream &st)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_set_crack<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, pi, clength, loc_x, loc_y, theta, ihpSize.t_size[i], pd.NN[i],
                pd.fail[i],pd.NL[i], pd.x[i]);
    }
}

__device__ __host__ real cal_dist( int NT, int id1, int id2, real *x, int Dim)
{
    real dist = 0.0;
    for (int i=0; i<Dim; i++)
    {
        dist += (x[id1+i*NT]-x[id2+i*NT])*(x[id1+i*NT]-x[id2+i*NT]);
    }
    return  dist;
}


__global__ void kernel_vol_Corr
        (int N, int MN, real horizon, int Dim, int *NN, int *NL, real *x, real *dx, real *idist, real *fac)
{
    unsigned int idx =  blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    if (i<N)
    {
        if (j<NN[i])
        {
            int cnode = NL[idx];
            real delta = horizon*dx[i];
            //idist[idx] = sqrt(cal_dist(N, i, cnode, x, Dim));
            if (Dim==2)
            {
                idist[idx]=sqrt(square(x[cnode]-x[i])+
                                square(x[cnode+N]-x[i+N]));
            } else if (Dim==3)
            {
                idist[idx]=sqrt(square(x[cnode]-x[i])+
                                square(x[cnode+N]-x[i+N]) +
                                square(x[cnode+N*2]-x[i+N*2]));
            }

            if (idist[idx] <= delta-dx[cnode]/2.0)
                fac[idx] = 1.0;
            else if (idist[idx] <= delta+dx[cnode]/2.0)
                fac[idx] = (delta+dx[cnode]/2.0-idist[idx]) / dx[cnode];
            else
                fac[idx] = 0.0;
        }
    }
}

void vol_Corr(int GPUS, int Dim, int MN, IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd, Stream &st)
{
    //long double strat = cpuSecond();
//#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_vol_Corr<<<grid.b_t[i], block_size, 0, st.body[i]>>>
                (ihpSize.t_size[i], MN, horizon, Dim, pd.NN[i], pd.NL[i], pd.x[i], pd.dx[i], pd.idist[i], pd.fac[i]);
    }
    device_sync(GPUS);
    //cout<<"vol corr time: "<<cpuSecond()-strat<<endl;
}

__global__ void kernel_mass(int N, int Dim, real E, real pratio, real delta, real thick, real pi, real size_min, real *mass)
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i<N)
    {
        if (Dim==2) {
            real bc = 6.0 * E / (pi * thick * cubic(delta) * (1.0 - pratio));
            mass[i] = 0.25 * (pi * delta * delta * thick) * (bc) / size_min * 1.0;
        } else if (Dim==3)
        {
            real bc = 6.0 * E / (pi * delta * cubic(delta) * (1.0 - 2.0*pratio));
            mass[i] = 0.25 * (pi * delta * delta * delta) * (bc) / size_min * 1.0;
        }
    }
}

void cal_mass_GPU(int GPUS, int Dim, real E, real pratio, real size_min, real thick, IHP_SIZE &ihpSize, Grid &grid,
                  Static_PD &pd, cudaStream_t *st_body)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_mass<<<grid.p_t[i], block_size, 0, st_body[i]>>>(
                ihpSize.t_size[i], Dim, E, pratio, horizon*size_min, thick, pi, size_min, pd.mass[i]);
    }
    device_sync(GPUS);
}



