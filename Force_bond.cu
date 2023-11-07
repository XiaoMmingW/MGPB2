//
// Created by wxm on 2023/8/7.
//

#include "Force_bond.cuh"
__global__  void cal_bc(int N, int Dim, real E, real pratio, real pi, real horizon, real thick, real *bc,real *dx)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        real delta = horizon*dx[i];
        if (Dim==2)
        {
            bc[i] = 3.0 * E / (pi*thick*cubic(delta)*(1.0 - pratio));
        } else if (Dim==3)
        {
            bc[i] =  3.0 * E / (pi*delta*cubic(delta)*(1.0 - 2.0*pratio));
        }
    }
}

static __global__ void kernel_Disp(int N, int select, real *x, real *disp)
{
    int i =  blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        disp[select*N+i] = 0.001*x[select*N+i];
    }
}


static __global__ void kernel_surface_F_2D
        (int N, int MN, real sedload_Cal, int select, int *NN, int *NL, real *x, real *disp,real *fncst, real *idist,
         real *fac, real *bc, real *vol)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    int  cnode = 0;
    real nlength =0.0;
    real stendens = 0.0;
    real stendens_i = 0.0;
    real stendens_j = 0.0;
    if(i<N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            nlength = sqrt(square(x[cnode]-x[i]+disp[cnode]-disp[i])+
                           square(x[cnode+N]-x[i+N]+disp[cnode+N]-disp[i+N]));
            stendens = 0.25*bc[i]*square(nlength-idist[idx])/idist[idx]*fac[idx];
            stendens_i = stendens*vol[cnode];
            stendens_j = stendens*vol[i];
            atomicAdd(&fncst[cnode+select*N], stendens_j);
           //if (i==0) printf("i %d cnode %d ff %e %e %e %e \n", i, cnode, stendens_i, stendens_j, bc[i], idist[idx]);
        }
        __syncwarp();

        for (int offset = 16; offset>0; offset>>=1)
        {
            stendens_i += __shfl_down_sync(FULL_MASK, stendens_i, offset);

        }
        if (j==0)
        {
            atomicAdd(&fncst[i+select*N],stendens_i);
            //if (i==0) printf("i %d cnode %d ff %e \n", i, cnode, fncst[i+select*N]);
        }
    }
}

static __global__ void kernel_surface_F_3D
        (int N, int MN, real sedload_Cal, int select, int *NN, int *NL, real *x, real *disp,real *fncst,  real *idist,
         real *fac, real *bc, real *vol)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    int  cnode = 0;
    real nlength =0.0;
    real stendens = 0.0;
    real stendens_i = 0.0;
    real stendens_j = 0.0;
    __shared__ real f[128];
    if(i<N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            nlength = sqrt(square(x[cnode]-x[i]+disp[cnode]-disp[i])+
                           square(x[cnode+N]-x[i+N]+disp[cnode+N]-disp[i+N]) +
                           square(x[cnode+N*2]-x[i+N*2]+disp[cnode+N*2]-disp[i+N*2]));
            stendens = 0.25*bc[i]*square(nlength-idist[idx])/idist[idx];

            stendens_i = stendens*vol[cnode]*fac[idx];
            stendens_j = stendens*vol[i]*fac[idx];
            //if (i==1 or cnode==1) printf("i %d cnode %d ff %e %e %e\n", i, cnode, stendens_i, stendens_j, idist[idx]);
            atomicAdd(&fncst[cnode+select*N], stendens_j);

        }
        __syncwarp();

        for (int offset = 16; offset>0; offset>>=1)
        {
            stendens_i += __shfl_down_sync(FULL_MASK, stendens_i, offset);

        }
        if(j%32==0)
        {
            f[j] = stendens_i;
        }
        __syncthreads();
        if (j==0)
        {
            atomicAdd(&fncst[i+select*N], f[0]+f[32]+f[64]+f[96]);
            //if (i==0) printf("i %d Dim %d ff %e \n", i, j, fncst[i+select*N]);
        }
    }
}

static __global__ void kernel_surface_F_3D_atom
        (int N, int MN, real sedload_Cal, int select, int *NN, int *NL, real *x, real *disp,real *fncst,  real *idist,
         real *fac, real *bc, real *vol)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    int  cnode = 0;
    real nlength =0.0;
    real stendens = 0.0;
    __shared__ real f[128];
    if(i<N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            nlength = sqrt(square(x[cnode]-x[i]+disp[cnode]-disp[i])+
                           square(x[cnode+N]-x[i+N]+disp[cnode+N]-disp[i+N]) +
                           square(x[cnode+N*2]-x[i+N*2]+disp[cnode+N*2]-disp[i+N*2]));
            stendens = 0.5*bc[i]*square(nlength-idist[idx])/idist[idx]*vol[cnode]*fac[idx];
        }
        __syncwarp();

        for (int offset = 16; offset>0; offset>>=1)
        {
            stendens += __shfl_down_sync(FULL_MASK, stendens, offset);

        }
        if(j%32==0)
        {
            f[j] = stendens;
        }
        __syncthreads();
        if (j==0)
        {
           fncst[i+select*N] = f[0]+f[32]+f[64]+f[96];
            //if (i==0) printf("i %d Dim %d ff %e \n", i, j, fncst[i+select*N]);
        }
    }
}

static __global__ void cal_fncst(int N, int NT, real sedload, int Dim, int begin, real* fncst)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        i += begin;
        for (int j=0; j<Dim; j++) {
            fncst[i + j * NT] = sedload / fncst[i + j * NT];
        //if (i==begin) printf("i %d Dim %d %d ff %e \n", i, j, N, fncst[i + j * NT]);
        }
    }
}

static __global__ void kernel_cal_surf_coff_F
        (int N, int MN, int Dim, real pi, int *NN, int *NL, real *scr, real *x, real *fncst, real *idist)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int i = idx/MN;
    int j = idx%MN;
    int  cnode = 0;

    if (i<N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];

            if(Dim == 1)
            {
                scr[idx] = (fncst[i] + fncst[cnode]) / 2.0;

            }
            if (Dim == 2)
            {
                real theta = 0.0;
                real scx = 0.0;
                real scy = 0.0;
                if (fabs(x[cnode+N] - x[i+N]) <= 1.0e-10)
                    theta = 0.0;
                else if (fabs(x[cnode] - x[i]) <= 1.0e-10)
                    theta = 90.0*pi/180.0;
                else
                    theta = atan(fabs(x[cnode+N] - x[i+N])/fabs(x[cnode] - x[i]));
                scx = (fncst[i] + fncst[cnode]) / 2.0;
                scy = (fncst[i+N] + fncst[cnode+N]) / 2.0;
                scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy)));
            }
            if (Dim==3)
            {
                real theta = 0.0;
                real scx = 0.0;
                real scy = 0.0;
                real scz = 0.0;

                if(fabs(x[cnode+2*N]-x[i+2*N])<1.0e-10)
                {
                    if(fabs(x[cnode+N]-x[i+N])<1.0e-10)
                        theta = 0.0;
                    else if (fabs(x[cnode]-x[i])<1.0e-10)
                        theta = 90.0*pi/180.0;
                    else
                        theta = atan(fabs(x[cnode+N] - x[i+N])/fabs(x[cnode] - x[i]));
                    real phi = 90.0*pi/180.0;
                    scx = (fncst[i] + fncst[cnode]) / 2.0;
                    scy = (fncst[i+N] + fncst[cnode+N]) / 2.0;
                    scz = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                    //scr[idx] = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                    scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy) +
                                         cos(phi)*cos(phi)/(scz*scz)));
                }
                else if (fabs(x[cnode]-x[i])<1.0e-10 && fabs(x[cnode+N]-x[i+N])<1.0e-10)
                    scr[idx] = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                else
                {
                    theta = atan(fabs(x[cnode+N] - x[i+N])/fabs(x[cnode] - x[i]));
                    real phi = acos(fabs(x[cnode+2*N]-x[i+2*N])/idist[idx]);
                    scx = (fncst[i] + fncst[cnode]) / 2.0;
                    scy = (fncst[i+N] + fncst[cnode+N]) / 2.0;
                    scz = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                    //scr[idx] = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                    scr[idx] = sqrt(1.0/(cos(theta)*cos(theta) / (scx*scx)  + sin(theta)*sin(theta) / (scy*scy) +
                                         cos(phi)*cos(phi)/(scz*scz)));
                }
            }
           // if (scr[idx]==NAN) printf("i %d Dim %d ff %e \n", i, j, scr[idx]);
        }
    }
}



void surfac_correct(int GPUS , real E, real pratio, int Dim, int MN, real thick, real sedload, int **exchange_flag, IHP_SIZE &ihpSize,
                    Grid &grid, Bond_Mech_PD &pd, Stream &st)
{


    real **fncst = new real* [GPUS];
    long double strat = cpuSecond();
    //#pragma omp parallel for
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void**) &fncst[i], ihpSize.t_size[i]*Dim*sizeof(real)));
        CHECK(cudaMemset(fncst[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        cal_bc<<<grid.p_t[i], block_size, 0, st.body[i]>>>(
                ihpSize.t_size[i], Dim, E, pratio, pi, horizon, thick, pd.bc[i],pd.dx[i]);
        for (int j=0; j<Dim; j++)
        {
            kernel_Disp<<<grid.p_t[i], block_size, 0, st.body[i]>>>(
                    ihpSize.t_size[i], j, pd.x[i], pd.disp[i]);
            if (Dim==2)
            {
                kernel_surface_F_2D<<<grid.b_t[i], block_size, 0, st.body[i]>>>
                        (ihpSize.t_size[i], MN, sedload, j, pd.NN[i], pd.NL[i], pd.x[i],
                         pd.disp[i],fncst[i],pd.idist[i],pd.fac[i], pd.bc[i], pd.vol[i]);
            } else if (Dim==3)
            {
//                kernel_surface_F_3D_atom<<<grid.b_t[i], block_size, 0, st.body[i]>>>
//                        (ihpSize.t_size[i], MN, sedload, j, pd.NN[i], pd.NL[i], pd.x[i],
//                         pd.disp[i],fncst[i],pd.idist[i],pd.fac[i], pd.bc[i], pd.vol[i]);
                kernel_surface_F_3D<<<grid.b_t[i], block_size, 0, st.body[i]>>>
                        (ihpSize.t_size[i], MN, sedload, j, pd.NN[i], pd.NL[i], pd.x[i],
                         pd.disp[i],fncst[i],pd.idist[i],pd.fac[i], pd.bc[i], pd.vol[i]);

            }
            CHECK(cudaMemset(pd.disp[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
        }
    }
    device_sync(GPUS);
    cout<<"hh "<<ihpSize.i_size[0]<<" "<<ihpSize.h_size[0][1]<<endl;
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        cal_fncst<<<grid.p_i[i], block_size, 0, st.body[i]>>>(
                ihpSize.i_size[i], ihpSize.t_size[i], sedload, Dim, 0, fncst[i]);
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                cal_fncst<<<grid.p_h[i][j], block_size, 0, st.halo[i][j]>>>
                        ( ihpSize.h_size[i][j], ihpSize.t_size[i], sedload, Dim, ihpSize.h_begin[i][j], fncst[i]);
                for (int k=0; k<Dim; k++) {
                    CHECK(cudaMemcpyAsync(&(fncst[j])[ihpSize.p_begin[j][i]+k*ihpSize.t_size[j]],
                                          &(fncst[i])[ihpSize.h_begin[i][j]+k*ihpSize.t_size[i]],
                                          ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                          st.halo[i][j]));
                }
            }
        }
    }
    device_sync(GPUS);
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_cal_surf_coff_F<<<grid.b_t[i], block_size, 0, st.body[i]>>>
        (ihpSize.t_size[i], MN, Dim, pi, pd.NN[i], pd.NL[i], pd.scr[i], pd.x[i], fncst[i], pd.idist[i]);
        CHECK(cudaFree(fncst[i]));
    }
    device_sync(GPUS);
    //cout<<"corr time: "<<cpuSecond()-strat<<endl;
    delete []fncst;
}

__global__  void kernel_bond_force_2D(
        int N, int MN, real horizon,  real pi, real sc, int *NN,
        real *vol, int *fail, int *NL, real *idist, real *fac, real *x, real *disp, real *pforce, real *dx, real *scr, real *bc
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
                if (s>=sc)  fail[idx] = 0;
                real t =  bc[i]*scr[idx]*s*fac[idx];
                force_xi = (t)*nx*vol[cnode];
                force_yi = (t)*ny*vol[cnode];
                force_xj = -t*nx*vol[i];
                force_yj = -t*ny*vol[i];
                atomicAdd(&pforce[cnode], force_xj);
                atomicAdd(&pforce[N+cnode], force_yj);
                //if (i==452) printf("%e %e %e %e %e %e\n",  ed, nlength[idx], idist[idx], theta[i], disp[cnode], disp[i]);
                //if (fabs(t)>0.1 | t== NAN) printf("i %d cnode %d %e\n", i, cnode, t);

            }
//            else {
//                real rdx = 2.0*dx[i]*dx[cnode]/(dx[i]+dx[cnode]);
//                real delta = horizon * rdx;
//                if (nlength[idx]<rdx)
//                {
//                    real t = 0.5*27.0*K/pi/ gcubic(delta)/thick*(nlength[idx]-rdx)/delta;
//                    force_xi = t*nx*vol[cnode];
//                    force_yi = t*ny*vol[cnode];
//                }
//            }
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

__global__  void kernel_bond_force_3D(
        int N, int MN, real horizon,  real pi, real sc, int *NN, real *vol, int *fail, int *NL, real *idist, real *fac,
        real *x, real *disp, real *pforce, real *dx, real *scr, real *bc
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
                if ((s)>=sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real t =  bc[i]*scr[idx]*s*fac[idx];
                force_xi = (t)*nx*vol[cnode];
                force_yi = (t)*ny*vol[cnode];
                force_zi = (t)*nz*vol[cnode];
                force_xj = -t*nx*vol[i];
                force_yj = -t*ny*vol[i];
                force_zj = -t*nz*vol[i];

                atomicAdd(&pforce[cnode], force_xj);
                atomicAdd(&pforce[N+cnode], force_yj);
                atomicAdd(&pforce[N*2+cnode], force_zj);
                //if (i==452) printf("%e %e %e %e %e %e\n",  ed, nlength[idx], idist[idx], theta[i], disp[cnode], disp[i]);
                //if (fabs(t)>0.1 | t== NAN) printf("i %d cnode %d %e\n", i, cnode, t);

            }
//            else {
//                real rdx = 2.0*dx[i]*dx[cnode]/(dx[i]+dx[cnode]);
//                real delta = horizon * rdx;
//                if (nlength[idx]<rdx)
//                {
//                    real t = 0.5*27.0*K/pi/ gcubic(delta)/thick*(nlength[idx]-rdx)/delta;
//                    force_xi = t*nx*vol[cnode];
//                    force_yi = t*ny*vol[cnode];
//                }
//            }
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

__global__  void kernel_bond_force_3D_atom(
        int N, int MN, real horizon,  real pi, real sc, int *NN, real *vol, int *fail, int *NL, real *idist, real *fac,
        real *x, real *disp, real *pforce, real *dx, real *scr, real *bc
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
                if ((s)>=sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real t =  2.0*bc[i]*scr[idx]*s*fac[idx]*vol[cnode];
                force_xi = (t)*nx;
                force_yi = (t)*ny;
                force_zi = (t)*nz;

                //if (i==452) printf("%e %e %e %e %e %e\n",  ed, nlength[idx], idist[idx], theta[i], disp[cnode], disp[i]);
                //if (fabs(t)>0.1 | t== NAN) printf("i %d cnode %d %e\n", i, cnode, t);

            }
//            else {
//                real rdx = 2.0*dx[i]*dx[cnode]/(dx[i]+dx[cnode]);
//                real delta = horizon * rdx;
//                if (nlength[idx]<rdx)
//                {
//                    real t = 0.5*27.0*K/pi/ gcubic(delta)/thick*(nlength[idx]-rdx)/delta;
//                    force_xi = t*nx*vol[cnode];
//                    force_yi = t*ny*vol[cnode];
//                }
//            }
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

__global__  void kernel_bond_force_3D_atom2(
        int N, int MN, real horizon,  real pi, real sc, int *NN, real *vol, int *fail, int *NL, real *idist, real *fac,
        real *x, real *disp, real *pforce, real *dx, real *scr, real *bc
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
                //if ((s)>=sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real t =  bc[i]*scr[idx]*s*fac[idx]*vol[cnode];
                force_xi = (t)*nx;
                force_yi = (t)*ny;
                force_zi = (t)*nz;
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
                if ((s)>=sc and fabs(x[i])>0.02 and fabs(x[cnode])>0.02)  fail[idx] = 0;
                real t =  bc[i]*scr[idx]*s*fac[idx]*vol[cnode];
                force_xi = (t)*nx;
                force_yi = (t)*ny;
                force_zi = (t)*nz;
            }  else
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


 void force_gpu(int GPUS, real sc, int Dim, int MN, Bond_Mech_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid) {


    for (int i = 0; i < GPUS; i++) {
         CHECK(cudaSetDevice(i));
         CHECK(cudaMemset(pd.pforce[i], 0, ihpSize.t_size[i]*Dim*sizeof(real)));
         if (Dim==2)
         {
             kernel_bond_force_2D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                     ihpSize.t_size[i], MN, horizon, pi, sc,  pd.NN[i],pd.vol[i],  pd.fail[i],  pd.NL[i],
                     pd.idist[i],  pd.fac[i],  pd.x[i],  pd.disp[i],  pd.pforce[i],  pd.dx[i],
                     pd.scr[i],  pd.bc[i]
             );
         } else if (Dim==3)
         {
//             kernel_bond_force_3D_atom2<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
//             ihpSize.t_size[i], MN, horizon, pi, sc,  pd.NN[i],pd.vol[i],  pd.fail[i],  pd.NL[i],
//             pd.idist[i],  pd.fac[i],  pd.x[i],  pd.disp[i],  pd.pforce[i],  pd.dx[i],
//             pd.scr[i],  pd.bc[i]
//             );
             kernel_bond_force_3D<<<grid.b_t[i], block_size, 0, st.body[i]>>>(
                     ihpSize.t_size[i], MN, horizon, pi, sc,  pd.NN[i],pd.vol[i],  pd.fail[i],  pd.NL[i],
                     pd.idist[i],  pd.fac[i],  pd.x[i],  pd.disp[i],  pd.pforce[i],  pd.dx[i],
                     pd.scr[i],  pd.bc[i]
             );
         }
     }
     device_sync(GPUS);

 }