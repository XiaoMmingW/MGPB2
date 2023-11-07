//
// Created by wxm on 2023/8/7.
//

#include "Force_state.cuh"

__global__ void kernel_initial_weight_2D(
        int N, int MN, real horizon, int begin, int *NN, int *NL, real *m, real *dx, real *vol, real *w, real *idist, real *fac)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    i += begin;
    real temp_m = 0.0;
    if (i<N)
    {
        if (j<NN[i])
        {
            int cnode = NL[idx];
            real delta = horizon*dx[cnode];
            w[idx] = exp(-square(idist[idx]) / square(delta));
            temp_m = w[idx] * square(idist[idx]) * vol[cnode] * fac[idx];
        }

        for (int offset=16; offset>0; offset>>=1)
            temp_m += __shfl_down_sync(FULL_MASK, temp_m, offset);
        if (j==0)
            m[i] = temp_m;
    }
}

__global__ void kernel_initial_weight_3D(
        int N, int MN, real horizon, int begin, int *NN, int *NL, real *m, real *dx, real *vol, real *w, real *idist, real *fac)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    i += begin;
    real temp_m = 0.0;
    __shared__ real mm[128];
    if (i<N)
    {
        if (j<NN[i])
        {
            int cnode = NL[idx];
            real delta = horizon*dx[cnode];
            w[idx] = exp(-square(idist[idx]) / square(delta));
            temp_m = w[idx] * square(idist[idx]) * vol[cnode] * fac[idx];
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

void cal_weight_gpu(int GPUS,  int Dim, int MN, int **exchange_flag, IHP_SIZE &ihpSize,
                    Grid &grid, Stream &st, State_PD &spd, Base_PD &pd)
{
    long double start = cpuSecond();
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        if (Dim==2)
        {
            kernel_initial_weight_2D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], MN, horizon, 0, pd.NN[i], pd.NL[i], spd.m[i], pd.dx[i],
                    pd.vol[i], spd.w[i], pd.idist[i], pd.fac[i]);
        } else if (Dim==3)
        {
            kernel_initial_weight_3D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], MN, horizon, 0, pd.NN[i], pd.NL[i], spd.m[i], pd.dx[i],
                    pd.vol[i], spd.w[i], pd.idist[i], pd.fac[i]);
        }
    }
    device_sync(GPUS);
    cout<<"weight time: "<<(cpuSecond()-start)*1000<<endl;
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                CHECK(cudaMemcpyAsync(&(spd.m[j])[ihpSize.p_begin[j][i]],
                                      &(spd.m[i])[ihpSize.h_begin[i][j]],
                                      ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                      st.halo[i][j]));
            }
        }
    }
    device_sync(GPUS);

}


__global__ void kernel_cal_theta_2D(
        int N, int NT, int MN, int begin, int *NN, real *m, real *vol,real *theta, int *NL, real *w, int *fail,
        real *idist, real *fac, real *x, real *disp)

{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    unsigned int cnode = 0;

    real temp_theta = 0.0;
    if (i < N)
    {
        i += begin;
        if (j < NN[i])
        {
            idx = i*MN+j;
            cnode = NL[idx];
            real nlength = sqrt(square(x[cnode] - x[i] + disp[cnode] - disp[i]) +
                                square(x[NT+cnode] - x[NT+i] + disp[NT+cnode] - disp[NT+i]));
            if (fail[idx]==1)
            {
                temp_theta = 2.0/m[i]*w[idx]*idist[idx]*(nlength-idist[idx])*fac[idx]*vol[cnode];
                //if (i==0) printf("theta %e  \n",  temp_theta);
                //if (i==0) printf("theta %e  %d %e %e %e %e\n",  nlength[idx]-idist[idx], cnode, disp[cnode], disp[i], disp[N+cnode],
                //disp[i+N]);
            }
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
            temp_theta += __shfl_down_sync(FULL_MASK, temp_theta, offset);
        if (j==0)
        {
            theta[i] = temp_theta;
            //printf("theta %d \n", i);

        }
    }
}



__global__ void kernel_cal_theta_3D(
        int N, int NT, int MN, int begin, int *NN, real *m, real *vol,real *theta, int *NL, real *w, int *fail,
       real *idist, real *fac, real *x, real *disp)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int i = idx / MN;
    unsigned int j = idx % MN;
    unsigned int cnode = 0;

    real temp_theta = 0.0;
    __shared__ real mm[128];
    if (i < N) {
        i += begin;
        if (j < NN[i]) {
            idx = i*MN+j;
            cnode = NL[idx];
            real nlength = sqrt(square(x[cnode] - x[i] + disp[cnode] - disp[i]) +
                                square(x[NT+cnode] - x[NT+i] + disp[NT+cnode] - disp[NT+i]) +
                                square(x[NT*2+cnode] - x[NT*2+i] + disp[NT*2+cnode] - disp[NT*2+i]));
            if (fail[idx] == 1) {
                temp_theta = 3.0 / m[i] * w[idx] * idist[idx] * (nlength- idist[idx]) * fac[idx] * vol[cnode];
                //if (i==0) printf("theta %e  \n",  temp_theta);
                //if (i==0) printf("theta %e  %d %e %e %e %e\n",  nlength-idist[idx], cnode, disp[cnode], disp[i], m[i]);
                //disp[i+N]);
                //if (cnode==0) printf("pforce %e %e %e\n",  theta[i], theta[cnode], (nlength- idist[idx]));
            }
        }
        __syncwarp();
        for (int offset = 16; offset > 0; offset >>= 1)
            temp_theta += __shfl_down_sync(FULL_MASK, temp_theta, offset);

        if(j%32==0)
            mm[j] = temp_theta;
        __syncthreads();
        if (j==0)
            theta[i] = mm[0]+mm[32]+mm[64]+mm[96];
    }
}

__global__  void kernel_state_force_2D(
        int N, real K, real G, int MN, real horizon, real thick, real pi, real sc, int *NN, real *m, real *theta,
        real *vol, int *fail, int *NL, real *w, real *idist, real *fac, real *x, real *disp, real *pforce, real *dx
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    unsigned int cnode = 0;

    real force_xi = 0.0;
    real force_yi = 0.0;
    real force_xj = 0.0;
    real force_yj = 0.0;

    if (i<N)
    {
        if (j < NN[i])
        {
            cnode = NL[idx];
            real ex = x[cnode] - x[i] + disp[cnode] - disp[i];
            real ey = x[N+cnode] - x[N+i] + disp[N+cnode] - disp[N+i];
            real nlength = sqrt(ex*ex + ey*ey);
            real nx = ex / nlength;
            real ny = ey / nlength;
            if (fail[idx])
            {
                real s = (nlength - idist[idx])/idist[idx];
                if (s>=sc and fabs(x[i])>0.02)  fail[idx] = 0;
                real ed = nlength-idist[idx] - theta[i]*idist[idx]/3.0;
                real t =  ((2.0*(3.0*K-G)/3.0*theta[i])/m[i]*w[idx]*idist[idx] + 8*G/m[i]*ed*w[idx]);
                force_xi = (t)*nx*vol[cnode]*fac[idx];
                force_yi = (t)*ny*vol[cnode]*fac[idx];
                force_xj = -t*nx*vol[i]*fac[idx];
                force_yj = -t*ny*vol[i]*fac[idx];
                atomicAdd(&pforce[cnode], force_xj);
                atomicAdd(&pforce[N+cnode], force_yj);
                //if (i==452) printf("%e %e %e %e %e %e\n",  ed, nlength[idx], idist[idx], theta[i], disp[cnode], disp[i]);
                //if (fabs(t)>0.1 | t== NAN) printf("i %d cnode %d %e\n", i, cnode, t);

            }

        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            force_xi += __shfl_down_sync(FULL_MASK, force_xi, offset);
            force_yi += __shfl_down_sync(FULL_MASK, force_yi, offset);
        }
        if (j==0)
        {
            atomicAdd(&pforce[i], force_xi);
            atomicAdd(&pforce[i+N], force_yi);
            //if (i==0) printf("%e \n", pforce[i]);
        }
    }
}

__global__  void kernel_state_force_3D(
        int N, real K, real G, int MN, real horizon, real thick, real pi, real sc, int *NN, real *m, real *theta,
        real *vol, int *fail, int *NL, real *w, real *idist, real *fac, real *x, real *disp, real *pforce, real *dx
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    unsigned int cnode = 0;

    real force_xi = 0.0;
    real force_yi = 0.0;
    real force_zi = 0.0;
    real force_xj = 0.0;
    real force_yj = 0.0;
    real force_zj = 0.0;
    __shared__ real fx[128];
    __shared__ real fy[128];
    __shared__ real fz[128];
    if (i<N)
    {
        if (j < NN[i])
        {
            cnode = NL[idx];
            real ex = x[cnode] - x[i] + disp[cnode] - disp[i];
            real ey = x[N+cnode] - x[N+i] + disp[N+cnode] - disp[N+i];
            real ez = x[N*2+cnode] - x[N*2+i] + disp[N*2+cnode] - disp[N*2+i];
            real nlength = sqrt(ex*ex + ey*ey +ez*ez);
            real nx = ex / nlength;
            real ny = ey / nlength;
            real nz = ez / nlength;
            if (fail[idx])
            {
                real s = (nlength - idist[idx])/idist[idx];
                if (s>sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real ed = nlength-idist[idx] - theta[i]*idist[idx]/3.0;
                real t =  ((3.0*K*theta[i])/m[i]*w[idx]*idist[idx] + 15.0*G/m[i]*ed*w[idx]);
                force_xi = (t)*nx*vol[cnode]*fac[idx];
                force_yi = (t)*ny*vol[cnode]*fac[idx];
                force_zi = (t)*nz*vol[cnode]*fac[idx];
                force_xj = -t*nx*vol[i]*fac[idx];
                force_yj = -t*ny*vol[i]*fac[idx];
                force_zj = -t*nz*vol[i]*fac[idx];
                atomicAdd(&pforce[cnode], force_xj);
                atomicAdd(&pforce[N+cnode], force_yj);
                atomicAdd(&pforce[N*2+cnode], force_zj);
                //if (i==452) printf("%e %e %e %e %e %e\n",  ed, nlength[idx], idist[idx], theta[i], disp[cnode], disp[i]);
                //if (fabs(t)>0.1 | t== NAN) printf("i %d cnode %d %e\n", i, cnode, t);
                //if (cnode==0) printf("pforce %e %e %e %e\n",  theta[i], theta[cnode], t, nlength-idist[idx]);

            }
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            force_xi += __shfl_down_sync(FULL_MASK, force_xi, offset);
            force_yi += __shfl_down_sync(FULL_MASK, force_yi, offset);
            force_zi += __shfl_down_sync(FULL_MASK, force_zi, offset);
        }
        if(j%32==0)
        {
            fx[j] = force_xi;
            fy[j] = force_yi;
            fz[j] = force_zi;
        }
        __syncthreads();
        if (j==0)
        {
            atomicAdd(&pforce[i], fx[0]+fx[32]+fx[64]+fx[96]);
            atomicAdd(&pforce[i+N], fy[0]+fy[32]+fy[64]+fy[96]);
            atomicAdd(&pforce[i+N*2], fz[0]+fz[32]+fz[64]+fz[96]);
            //if (i==0) printf("%e \n", pforce[i]);
        }
    }
}

__global__  void kernel_state_force_3D_atom(
        int N, real K, real G, int MN, real horizon, real thick, real pi, real sc, int *NN, real *m, real *theta,
        real *vol, int *fail, int *NL, real *w, real *idist, real *fac, real *x, real *disp, real *pforce, real *dx
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    unsigned int cnode = 0;

    real force_xi = 0.0;
    real force_yi = 0.0;
    real force_zi = 0.0;

    __shared__ real fx[128];
    __shared__ real fy[128];
    __shared__ real fz[128];
    if (i<N)
    {
        if (j < NN[i])
        {
            cnode = NL[idx];
            real ex = x[cnode] - x[i] + disp[cnode] - disp[i];
            real ey = x[N+cnode] - x[N+i] + disp[N+cnode] - disp[N+i];
            real ez = x[N*2+cnode] - x[N*2+i] + disp[N*2+cnode] - disp[N*2+i];
            real nlength = sqrt(ex*ex + ey*ey +ez*ez);
            real nx = ex / nlength;
            real ny = ey / nlength;
            real nz = ez / nlength;
            if (fail[idx])
            {
                real s = (nlength - idist[idx])/idist[idx];
                if (s>sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real ed = nlength-idist[idx] - theta[i]*idist[idx]/3.0;
                real ed2 = nlength-idist[idx] - theta[cnode]*idist[idx]/3.0;
                real t =  ((3.0*K*theta[i])/m[i]*w[idx]*idist[idx] + 15.0*G/m[i]*ed*w[idx]);
                real t2 = ((3.0*K*theta[cnode])/m[cnode]*w[idx]*idist[idx] + 15.0*G/m[cnode]*ed2*w[idx]);
                force_xi = (t+t2)*nx*vol[cnode]*fac[idx];
                force_yi = (t+t2)*ny*vol[cnode]*fac[idx];
                force_zi = (t+t2)*nz*vol[cnode]*fac[idx];

//                atomicAdd(&pforce[cnode], force_xj);
//                atomicAdd(&pforce[N+cnode], force_yj);
//                atomicAdd(&pforce[N*2+cnode], force_zj);
                //if (i==452) printf("%e %e %e %e %e %e\n",  ed, nlength[idx], idist[idx], theta[i], disp[cnode], disp[i]);
                //if (fabs(t)>0.1 | t== NAN) printf("i %d cnode %d %e\n", i, cnode, t);
                //if (cnode==0) printf("pforce %e %e %e %e\n",  theta[i], theta[cnode], t, nlength-idist[idx]);

            }
        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            force_xi += __shfl_down_sync(FULL_MASK, force_xi, offset);
            force_yi += __shfl_down_sync(FULL_MASK, force_yi, offset);
            force_zi += __shfl_down_sync(FULL_MASK, force_zi, offset);
        }
        if(j%32==0)
        {
            fx[j] = force_xi;
            fy[j] = force_yi;
            fz[j] = force_zi;
        }
        __syncthreads();
        if (j==0)
        {
            pforce[i] = fx[0]+fx[32]+fx[64]+fx[96];
            pforce[i+N] = fy[0]+fy[32]+fy[64]+fy[96];
            pforce[i+N*2] = fz[0]+fz[32]+fz[64]+fz[96];
            //if (i==0) printf("%e \n", pforce[i]);
        }
    }
}

__global__  void kernel_state_force_3D_atom2(
        int N, real K, real G, int MN, real horizon, real thick, real pi, real sc, int *NN, real *m, real *theta,
        real *vol, int *fail, int *NL, real *w, real *idist, real *fac, real *x, real *disp, real *pforce, real *dx
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    unsigned int cnode = 0;

    real force_xi = 0.0;
    real force_yi = 0.0;
    real force_zi = 0.0;
    real force_xj = 0.0;
    real force_yj = 0.0;
    real force_zj = 0.0;
    __shared__ real fx[128];
    __shared__ real fy[128];
    __shared__ real fz[128];
    if (i<N)
    {
        if (j < NN[i])
        {
            cnode = NL[idx];
            real ex = x[cnode] - x[i] + disp[cnode] - disp[i];
            real ey = x[N+cnode] - x[N+i] + disp[N+cnode] - disp[N+i];
            real ez = x[N*2+cnode] - x[N*2+i] + disp[N*2+cnode] - disp[N*2+i];
            real nlength = sqrt(ex*ex + ey*ey +ez*ez);
            real nx = ex / nlength;
            real ny = ey / nlength;
            real nz = ez / nlength;
            if (fail[idx])
            {
//                real s = (nlength - idist[idx])/idist[idx];
//                if (s>sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real ed = nlength-idist[idx] - theta[i]*idist[idx]/3.0;
                real t =  ((3.0*K*theta[i])/m[i]*w[idx]*idist[idx] + 15.0*G/m[i]*ed*w[idx]);

                force_xi = (t)*nx*vol[cnode]*fac[idx];
                force_yi = (t)*ny*vol[cnode]*fac[idx];
                force_zi = (t)*nz*vol[cnode]*fac[idx];
            }
        }

        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            force_xi += __shfl_down_sync(FULL_MASK, force_xi, offset);
            force_yi += __shfl_down_sync(FULL_MASK, force_yi, offset);
            force_zi += __shfl_down_sync(FULL_MASK, force_zi, offset);
        }
        if(j%32==0)
        {
            fx[j] = force_xi;
            fy[j] = force_yi;
            fz[j] = force_zi;
        }
        __syncthreads();
        if (j==0)
        {
            pforce[i] = fx[0]+fx[32]+fx[64]+fx[96];
            pforce[i+N] = fy[0]+fy[32]+fy[64]+fy[96];
            pforce[i+N*2] = fz[0]+fz[32]+fz[64]+fz[96];
            //if (i==0) printf("%e \n", pforce[i]);
        }
        __syncthreads();
        if (j < NN[i])
        {
            cnode = NL[idx];
            real ex = x[cnode] - x[i] + disp[cnode] - disp[i];
            real ey = x[N+cnode] - x[N+i] + disp[N+cnode] - disp[N+i];
            real ez = x[N*2+cnode] - x[N*2+i] + disp[N*2+cnode] - disp[N*2+i];
            real nlength = sqrt(ex*ex + ey*ey +ez*ez);
            real nx = ex / nlength;
            real ny = ey / nlength;
            real nz = ez / nlength;
            if (fail[idx])
            {
                real s = (nlength - idist[idx])/idist[idx];
                if (s>sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real ed = nlength-idist[idx] - theta[cnode]*idist[idx]/3.0;
                real t = ((3.0*K*theta[cnode])/m[cnode]*w[idx]*idist[idx] + 15.0*G/m[cnode]*ed*w[idx]);
                force_xi = (t)*nx*vol[cnode]*fac[idx];
                force_yi = (t)*ny*vol[cnode]*fac[idx];
                force_zi = (t)*nz*vol[cnode]*fac[idx];
            } else
            {
                force_xi = 0.0;
                force_yi = 0.0;
                force_zi = 0.0;
            }

        }
        __syncwarp();
        for (int offset = 16; offset>0; offset>>=1)
        {
            force_xi += __shfl_down_sync(FULL_MASK, force_xi, offset);
            force_yi += __shfl_down_sync(FULL_MASK, force_yi, offset);
            force_zi += __shfl_down_sync(FULL_MASK, force_zi, offset);
        }
        if(j%32==0)
        {
            fx[j] = force_xi;
            fy[j] = force_yi;
            fz[j] = force_zi;
        }
        __syncthreads();
        if (j==0)
        {
            pforce[i] += fx[0]+fx[32]+fx[64]+fx[96];
            pforce[i+N] += fy[0]+fy[32]+fy[64]+fy[96];
            pforce[i+N*2] += fz[0]+fz[32]+fz[64]+fz[96];
            //if (i==0) printf("%e \n", pforce[i]);
        }
    }
}


void force_state_gpu(int GPUS, real E, real pratio, real sc, int Dim, int MN, real thick, State_Mech_PD &pd, IHP_SIZE &ihpSize,
               Stream &st, Grid &grid, int **exchange_flag)
{
    real G = 0.5*E/(1.0+pratio);
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        if (Dim==2)
        {
            kernel_cal_theta_2D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
            ihpSize.i_size[i], ihpSize.t_size[i], MN, 0, pd.NN[i], pd.m[i], pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
            pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);
            for (int i = 0; i < GPUS; i++) {
                CHECK(cudaSetDevice(i));
                for (int j = 0; j < GPUS; j++) {
                    if (exchange_flag[i][j] == 1) {
                        kernel_cal_theta_2D<<<grid.b_h[i][j], block_size, 0, st.halo[i][j]>>>(
                                ihpSize.h_size[i][j], ihpSize.t_size[i], MN, ihpSize.h_begin[i][j], pd.NN[i], pd.m[i],
                                pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
                                pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);

                        CHECK(cudaMemcpyAsync(&(pd.theta[j])[ihpSize.p_begin[j][i]],
                                              &(pd.theta[i])[ihpSize.h_begin[i][j]],
                                              ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                              st.halo[i][j]));
                    }
                }
            }
        } else if (Dim==3)
        {
            kernel_cal_theta_3D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
                    ihpSize.i_size[i], ihpSize.t_size[i], MN, 0, pd.NN[i], pd.m[i], pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
                    pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);
            for (int i = 0; i < GPUS; i++) {
                CHECK(cudaSetDevice(i));
                for (int j = 0; j < GPUS; j++) {
                    if (exchange_flag[i][j] == 1) {
                        kernel_cal_theta_3D<<<grid.b_h[i][j], block_size, 0, st.halo[i][j]>>>(
                                ihpSize.h_size[i][j], ihpSize.t_size[i], MN, ihpSize.h_begin[i][j], pd.NN[i], pd.m[i],
                                pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
                                pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);

                        CHECK(cudaMemcpyAsync(&(pd.theta[j])[ihpSize.p_begin[j][i]],
                                              &(pd.theta[i])[ihpSize.h_begin[i][j]],
                                              ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                              st.halo[i][j]));
                    }
                }
            }
        }
    }
    device_sync(GPUS);
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.pforce[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        if (Dim==2)
        {
            real K = 0.5*E/(1.0-pratio);
            kernel_state_force_2D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], K, G, MN, horizon, thick, pi, sc, pd.NN[i], pd.m[i], pd.theta[i],
                    pd.vol[i], pd.fail[i], pd.NL[i], pd.w[i], pd.idist[i], pd.fac[i], pd.x[i],
                    pd.disp[i], pd.pforce[i], pd.dx[i]
            );
        } else if (Dim==3) {
            real K = E/3.0/(1.0-2.0*pratio);
            kernel_state_force_3D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], K, G, MN, horizon, thick, pi, sc, pd.NN[i], pd.m[i], pd.theta[i],
                    pd.vol[i], pd.fail[i], pd.NL[i], pd.w[i], pd.idist[i], pd.fac[i], pd.x[i],
                    pd.disp[i], pd.pforce[i], pd.dx[i]
            );

//            kernel_state_force_3D_atom2<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
//                    ihpSize.t_size[i], K, G, MN, horizon, thick, pi, sc, pd.NN[i], pd.m[i], pd.theta[i],
//                    pd.vol[i], pd.fail[i], pd.NL[i], pd.w[i], pd.idist[i], pd.fac[i], pd.x[i],
//                    pd.disp[i], pd.pforce[i], pd.dx[i]
//            );
        }

    }
    device_sync(GPUS);
}

void force_state_gpu1(int GPUS, real E, real pratio, real sc, int Dim, int MN, real thick, State_Mech_PD &pd, IHP_SIZE &ihpSize,
                     Stream &st, Grid &grid, int **exchange_flag)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        if (Dim==2)
        {
            kernel_cal_theta_2D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
                    ihpSize.i_size[i], ihpSize.t_size[i], MN, 0, pd.NN[i], pd.m[i], pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
                    pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);
            for (int i = 0; i < GPUS; i++) {
                CHECK(cudaSetDevice(i));
                for (int j = 0; j < GPUS; j++) {
                    if (exchange_flag[i][j] == 1) {
                        kernel_cal_theta_2D<<<grid.b_h[i][j], block_size, 0, st.halo[i][j]>>>(
                                ihpSize.h_size[i][j], ihpSize.t_size[i], MN, ihpSize.h_begin[i][j], pd.NN[i], pd.m[i],
                                pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
                                pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);

                        CHECK(cudaMemcpyAsync(&(pd.theta[j])[ihpSize.p_begin[j][i]],
                                              &(pd.theta[i])[ihpSize.h_begin[i][j]],
                                              ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                              st.halo[i][j]));
                    }
                }
            }
        } else if (Dim==3)
        {
            kernel_cal_theta_3D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
                    ihpSize.i_size[i], ihpSize.t_size[i], MN, 0, pd.NN[i], pd.m[i], pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
                    pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);
            for (int i = 0; i < GPUS; i++) {
                CHECK(cudaSetDevice(i));
                for (int j = 0; j < GPUS; j++) {
                    if (exchange_flag[i][j] == 1) {
                        kernel_cal_theta_3D<<<grid.b_h[i][j], block_size, 0, st.halo[i][j]>>>(
                                ihpSize.h_size[i][j], ihpSize.t_size[i], MN, ihpSize.h_begin[i][j], pd.NN[i], pd.m[i],
                                pd.vol[i], pd.theta[i], pd.NL[i], pd.w[i],
                                pd.fail[i], pd.idist[i], pd.fac[i], pd.x[i], pd.disp[i]);

                        CHECK(cudaMemcpyAsync(&(pd.theta[j])[ihpSize.p_begin[j][i]],
                                              &(pd.theta[i])[ihpSize.h_begin[i][j]],
                                              ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                              st.halo[i][j]));
                    }
                }
            }
        }
    }
    device_sync(GPUS);
}

void force_state_gpu2(int GPUS, real E, real pratio, real sc, int Dim, int MN, real thick, State_Mech_PD &pd, IHP_SIZE &ihpSize,
                     Stream &st, Grid &grid, int **exchange_flag)
{
    real G = 0.5*E/(1.0+pratio);
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.pforce[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        if (Dim==2)
        {
            real K = 0.5*E/(1.0-pratio);
            kernel_state_force_2D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], K, G, MN, horizon, thick, pi, sc, pd.NN[i], pd.m[i], pd.theta[i],
                    pd.vol[i], pd.fail[i], pd.NL[i], pd.w[i], pd.idist[i], pd.fac[i], pd.x[i],
                    pd.disp[i], pd.pforce[i], pd.dx[i]
            );
        } else if (Dim==3) {
            real K = E/3.0/(1.0-2.0*pratio);
            kernel_state_force_3D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], K, G, MN, horizon, thick, pi, sc, pd.NN[i], pd.m[i], pd.theta[i],
                    pd.vol[i], pd.fail[i], pd.NL[i], pd.w[i], pd.idist[i], pd.fac[i], pd.x[i],
                    pd.disp[i], pd.pforce[i], pd.dx[i]
            );

//            kernel_state_force_3D_atom<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
//                    ihpSize.t_size[i], K, G, MN, horizon, thick, pi, sc, pd.NN[i], pd.m[i], pd.theta[i],
//                    pd.vol[i], pd.fail[i], pd.NL[i], pd.w[i], pd.idist[i], pd.fac[i], pd.x[i],
//                    pd.disp[i], pd.pforce[i], pd.dx[i]
//            );
        }

    }
    device_sync(GPUS);
}