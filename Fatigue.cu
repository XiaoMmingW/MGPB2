//
// Created by wxm on 2023/8/11.
//

#include "Fatigue.cuh"

__global__ void kernel_update_phase1(int N, int MN, real dmg_limit, int *NN, int *NL, int *phase, real *dmg)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    if (i<N)
    {
        if (j<NN[i])
        {
            int cnode = NL[idx];
            if (j==0 and dmg[i]>=dmg_limit)
            {
                phase[i] = 1;
            }
            if (dmg[cnode]>=dmg_limit)
            {
                phase[i] = 1;
            }
        }
    }
}

__global__ void kernel_update_phase2(int N, int MN, int *NN, int *NL, int *phase, int *flag_state)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;

    if (i<N)
    {
        if (j<NN[i])
        {
            int cnode = NL[idx];
            if ((phase[i]==1 or phase[cnode]==1))
            {
                if (flag_state[idx]==0) flag_state[idx] = 1;
            }
        }
    }
}


__global__ void kernel_cal_life
        (int N, int MN,  real ratio, real A, real M, int *NN, real *life, int *fail, int *flag_state,
         real *lambda, real *smax)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    life[idx] = 1.0e18;
    if (i<N)
    {
        if (j<NN[i]) {
            real strain = 0.0;
            if (fail[idx] == 1)
            {
                if (flag_state[idx]==1) {
                    smax[idx] = strain = smax[idx] * (1.0 - ratio);
                    life[idx] = (1.0+lambda[idx]) / (A * pow(strain, M));
                    //printf("life %e %e\n", strain, life[idx]);
                }

            }
        }
    }
}

__global__ void kernel_update_lambda
        (int N, int MN, real A, real M, real life_cal, real *strain, int *NN, int *fail, real *lambda, int *NL, real *x,
         real *c_tip, int *flag_state)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    int cnode = 0;
    if(i<N)
    {
        if (j<NN[i])
        {
            if (fail[idx]==1) {
                if (flag_state[idx]==1)
                {
                    lambda[idx] -= life_cal * A * pow(strain[idx], M);
                    if (lambda[idx]<=-0.999) {
                        fail[idx] = 0;
                        cnode = NL[idx];
                        real temp_tip =  (x[i]+x[cnode])/2.0;
                        if (temp_tip<c_tip[0]) {
                            c_tip[0]= temp_tip;
                            //printf("temp_tip %e %e\n", c_tip[0], temp_tip);
                        }

                    }
                }
            }
        }
    }
}

void fatigue(const int GPUS, int NT, int MN, int break_num, real dmg_limit, real load_ratio, real A, real M, real &life_total, real &life_n,
             Bond_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid, real *life, real &ctip)
{

    int idx=0;
    //#pragma omp parallel for
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_update_phase1<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, dmg_limit, pd.NN[i], pd.NL[i], pd.phase[i], pd.dmg[i]);
        kernel_update_phase2<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, pd.NN[i], pd.NL[i], pd.phase[i], pd.state[i]);
        kernel_cal_life<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, load_ratio, A, M, pd.NN[i], pd.life[i], pd.fail[i], pd.state[i],
                pd.lambda[i], pd.smax[i]);
        CHECK(cudaMemcpy(&life[idx], pd.life[i], ihpSize.ih_size[i]*MN* sizeof(real), cudaMemcpyDeviceToDevice));
        idx += ihpSize.ih_size[i]*MN;
    }
    CHECK(cudaSetDevice(0));
    thrust::sort(thrust::device, life, life+NT*MN);
    real life_cal;
    CHECK(cudaMemcpy(&life_cal, &life[break_num-1], sizeof(real), cudaMemcpyDeviceToHost));
    life_total += life_cal;
    life_n = life_cal;
    real ttctip = ctip;
    real hctip[GPUS];
    //#pragma omp parallel for
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        hctip[i] = ctip;
        CHECK(cudaMemset(pd.c_tip[i], 0, sizeof(real)));
        CHECK(cudaMemcpy(pd.c_tip[i], &ttctip,  sizeof(real), cudaMemcpyHostToDevice));
        kernel_update_lambda<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, A, M, life_cal, pd.smax[i], pd.NN[i], pd.fail[i],
                pd.lambda[i], pd.NL[i], pd.x[i], pd.c_tip[i], pd.state[i]);

        CHECK(cudaMemcpyAsync(&hctip[i],pd.c_tip[i],sizeof(real), cudaMemcpyDeviceToHost,
                              st.body[i]));
    }
    device_sync(GPUS);
    ctip = hctip[0];
    for (int i=0; i<GPUS; i++) {
        if (ctip>hctip[i]) ctip = hctip[i];

    }
    cout<<"ctip: "<<ctip<<endl;

}

int cmp(const void *a, const void *b)
{
    return *((real*)a)>*((real*)b) ? 1:-1;
}


void fatigue_state(const int GPUS, int NT, int MN, int break_num, real dmg_limit, real load_ratio, real A, real M, real &life_total, real &life_n,
                   State_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid, real *life, real &ctip, real *h_life)
{

    //int idx=0;
    //#pragma omp parallel for
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_update_phase1<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, dmg_limit, pd.NN[i], pd.NL[i], pd.phase[i], pd.dmg[i]);
        kernel_update_phase2<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, pd.NN[i], pd.NL[i], pd.phase[i], pd.state[i]);
        kernel_cal_life<<<grid.b_ih[i], block_size, 0, st.body[i]>>>(
                ihpSize.ih_size[i], MN, load_ratio, A, M, pd.NN[i], pd.life[i], pd.fail[i], pd.state[i],
                pd.lambda[i], pd.smax[i]);
        thrust::sort(thrust::device, pd.life[i], pd.life[i]+ihpSize.ih_size[i]*MN);
        CHECK(cudaMemcpy(&h_life[i*break_num], pd.life[i], break_num * sizeof(real), cudaMemcpyDeviceToHost));
        //CHECK(cudaMemcpy(&life[idx], pd.life[i], ihpSize.ih_size[i]*MN* sizeof(real), cudaMemcpyDeviceToDevice));
        //idx += ihpSize.ih_size[i]*MN;
    }

    if (GPUS>1)
    {
        qsort(h_life, break_num*GPUS, sizeof(real), cmp);
    }

    //CHECK(cudaSetDevice(0));
    //thrust::sort(thrust::device, life, life+NT*MN);
    real life_cal = h_life[break_num-1];
    //CHECK(cudaMemcpy(&life_cal, &life[break_num-1], sizeof(real), cudaMemcpyDeviceToHost));
    life_total += life_cal;
    life_n = life_cal;
    real ttctip = ctip;
    real hctip[GPUS];
    //#pragma omp parallel for
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        hctip[i] = ctip;
        CHECK(cudaMemset(pd.c_tip[i], 0, sizeof(real)));
        CHECK(cudaMemcpy(pd.c_tip[i], &ttctip,  sizeof(real), cudaMemcpyHostToDevice));
        kernel_update_lambda<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, A, M, life_cal, pd.smax[i], pd.NN[i], pd.fail[i],
                pd.lambda[i], pd.NL[i], pd.x[i], pd.c_tip[i], pd.state[i]);

        CHECK(cudaMemcpyAsync(&hctip[i],pd.c_tip[i],sizeof(real), cudaMemcpyDeviceToHost,
                              st.body[i]));
    }
    device_sync(GPUS);
    ctip = hctip[0];
    for (int i=0; i<GPUS; i++) {
        if (ctip>hctip[i]) ctip = hctip[i];

    }
    //cout<<"ctip: "<<ctip<<endl;

}

__global__  void kernel_bond_force_2D_fatigue(
        int N, int MN, real horizon,  real pi, int *NN, real *vol, int *fail, int *NL, real *idist, real *fac,
        real *x, real *disp, real *pforce, real *dx, real *scr, real *bc, real *smax
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
                if (s>smax[idx])  smax[idx] = s;
                //real t =  bc[i]*s*fac[idx];
                real t =  bc[i]*scr[idx]*s*fac[idx];
                force_xi = (t)*nx*vol[cnode];
                force_yi = (t)*ny*vol[cnode];
                force_xj = -t*nx*vol[i];
                force_yj = -t*ny*vol[i];
                atomicAdd(&pforce[cnode], force_xj);
                atomicAdd(&pforce[N+cnode], force_yj);
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

__global__  void kernel_bond_force_2D_fatigue_noatom(
        int N, int MN, real horizon,  real pi, int *NN, real *vol, int *fail, int *NL, real *idist, real *fac,
        real *x, real *disp, real *pforce, real *dx, real *scr, real *bc, real *smax
)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    unsigned int cnode = 0;

    real force_xi = 0.0;
    real force_yi = 0.0;

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
                if (s>smax[idx])  smax[idx] = s;
                //real t =  bc[i]*s*fac[idx];
                real t =  2.0*bc[i]*scr[idx]*s*fac[idx]*vol[cnode];
                force_xi = (t)*nx;
                force_yi = (t)*ny;


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
            pforce[i] = force_xi;
            pforce[i+N] = force_yi;
            //if (i==0) printf("%e \n", pforce[i]);
        }
    }
}

void force_fatigue(int GPUS, int Dim, int MN, Bond_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid) {

    //#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.pforce[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));

//            kernel_bond_force_2D_fatigue<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
//                    ihpSize.t_size[i], MN, horizon, pi, pd.NN[i],pd.vol[i],  pd.fail[i],  pd.NL[i],
//                    pd.idist[i],  pd.fac[i],  pd.x[i],  pd.disp[i],  pd.pforce[i],  pd.dx[i],
//                    pd.scr[i],  pd.bc[i], pd.smax[i]
//            );

        kernel_bond_force_2D_fatigue_noatom<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], MN, horizon, pi, pd.NN[i],pd.vol[i],  pd.fail[i],  pd.NL[i],
                pd.idist[i],  pd.fac[i],  pd.x[i],  pd.disp[i],  pd.pforce[i],  pd.dx[i],
                pd.scr[i],  pd.bc[i], pd.smax[i]);

    }
    device_sync(GPUS);
}

__global__  void kernel_state_force_2D_fatigue(
        int N, real K, real G, int MN, real horizon, real thick, real pi,  int *NN, real *m, real *theta,
        real *vol, int *fail, int *NL, real *w, real *idist, real *fac, real *x, real *disp, real *pforce, real *dx,
        real *smax
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
                if (s>smax[idx])  smax[idx] = s;
                //if (s>=sc and fabs(x[i])>0.02)  fail[idx] = 0;
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

void force_fatigue_state(int GPUS, int Dim, int MN, real K, real G, real thick, State_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st,
                         Grid &grid, int **exchange_flag) {
    //#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_cal_theta_2D<<<grid.b_i[i], block_size, 0, st.body[i]>>>(
                ihpSize.i_size[i], ihpSize.t_size[i], MN, 0, pd.NN[i], pd.m[i], pd.vol[i], pd.theta[i], pd.NL[i],
                pd.w[i],
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
    }
    device_sync(GPUS);
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.pforce[i], 0, ihpSize.t_size[i] * Dim * sizeof(real)));
        kernel_state_force_2D_fatigue<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], K, G, MN, horizon, thick, pi, pd.NN[i], pd.m[i], pd.theta[i],
                pd.vol[i], pd.fail[i], pd.NL[i], pd.w[i], pd.idist[i], pd.fac[i], pd.x[i],
                pd.disp[i], pd.pforce[i], pd.dx[i], pd.smax[i]
        );
    }
    device_sync(GPUS);
}
void initial_fatigue(int GPUS, int Dim, int MN, Bond_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid) {

#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.disp[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        CHECK(cudaMemset(pd.velhalfold[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        CHECK(cudaMemset(pd.pforceold[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        CHECK(cudaMemset(pd.smax[i], 0, ihpSize.t_size[i]*MN*sizeof(real)));
    }
}

void initial_fatigue_state(int GPUS, int Dim, int MN, State_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid) {

#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(pd.disp[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        CHECK(cudaMemset(pd.velhalfold[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        CHECK(cudaMemset(pd.pforceold[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        CHECK(cudaMemset(pd.smax[i], 0, ihpSize.t_size[i]*MN*sizeof(real)));
    }
}