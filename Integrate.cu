//
// Created by wxm on 2023/6/20.
//

#include "Integrate.cuh"


__global__ void kernel_integrate_CD(
        int N, int NT, real dens, real dt, int begin, int Dim, real *acc, real *vel, real *disp,  real *pforce, real *bforce)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<N)
    {
        i += begin;
        real t_acc = 0.0;
        for (int j=0; j<Dim; j++)
        {
//            t_acc = (pforce[NT*j+i]+bforce[NT*j+i])/dens;
//            vel[NT*j+i] += (t_acc)*dt;
//            disp[NT*j+i] += dt*vel[NT*j+i];
            t_acc = (pforce[NT*j+i]+bforce[NT*j+i])/dens;
            vel[NT*j+i] += (acc[NT*j+i] + t_acc)*dt/2.0;
            acc[NT*j+i] = t_acc;
            disp[NT*j+i] += (dt*vel[NT*j+i]+ t_acc*dt*dt/2);
        }
    }
}

__global__ void kernel_integrate_CD_contact(
        int N, int NT, real dens, real dt, int begin, int Dim, real *acc, real *vel, real *disp,  real *pforce, real *bforce,
     real *olddisp)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<N)
    {
        i += begin;
        real t_acc = 0.0;
        for (int j=0; j<Dim; j++)
        {

            olddisp[NT*j+i] = disp[NT*j+i];

//            t_acc = (pforce[NT*j+i]+bforce[NT*j+i])/dens;
//            vel[NT*j+i] += (t_acc)*dt;
//            disp[NT*j+i] += dt*vel[NT*j+i];

            t_acc = (pforce[NT*j+i]+bforce[NT*j+i])/dens;
            vel[NT*j+i] += (acc[NT*j+i] + t_acc)*dt/2.0;
            acc[NT*j+i] = t_acc;
            disp[NT*j+i] += (dt*vel[NT*j+i]+ t_acc*dt*dt/2);
            //if (i==0) printf("pforce %e\n", pforce[NT*j+i]);
        }
    }
}

void integrate_CD(int GPUS , real dens, real dt, int Dim, int **exchange_flag, IHP_SIZE &ihpSize,
                  Grid &grid, Mech_PD &pd, Stream &st)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_integrate_CD<<<grid.p_i[i], block_size, 0, st.body[i]>>>(
                ihpSize.i_size[i], ihpSize.t_size[i], dens, dt, 0, Dim, pd.acc[i], pd.vel[i],
                pd.disp[i],  pd.pforce[i], pd.bforce[i]);
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                kernel_integrate_CD<<<grid.p_h[i][j], block_size, 0, st.halo[i][j]>>>
                        ( ihpSize.h_size[i][j], ihpSize.t_size[i], dens, dt, ihpSize.h_begin[i][j], Dim, pd.acc[i], pd.vel[i],
                          pd.disp[i],  pd.pforce[i], pd.bforce[i]);
                for (int k=0; k<Dim; k++) {
                    CHECK(cudaMemcpyAsync(&(pd.disp[j])[ihpSize.p_begin[j][i]+k*ihpSize.t_size[j]],
                                          &(pd.disp[i])[ihpSize.h_begin[i][j]+k*ihpSize.t_size[i]],
                                          ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                          st.halo[i][j]));
                }
            }
        }
    }
    device_sync(GPUS);
}

void integrate_CD_contact(int GPUS , real dens, real dt, int Dim, int **exchange_flag, IHP_SIZE &ihpSize,
                  Grid &grid, Mech_PD &pd, Stream &st)
{
    long double start = cpuSecond();
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_integrate_CD_contact<<<grid.p_i[i], block_size, 0, st.body[i]>>>(
                ihpSize.i_size[i], ihpSize.t_size[i], dens, dt, 0, Dim, pd.acc[i], pd.vel[i],
                pd.disp[i],  pd.pforce[i], pd.bforce[i],  pd.olddisp[i]);
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                kernel_integrate_CD_contact<<<grid.p_h[i][j], block_size, 0, st.halo[i][j]>>>
                        ( ihpSize.h_size[i][j], ihpSize.t_size[i], dens, dt, ihpSize.h_begin[i][j], Dim, pd.acc[i], pd.vel[i],
                          pd.disp[i],  pd.pforce[i], pd.bforce[i],  pd.olddisp[i]);
                for (int k=0; k<Dim; k++) {
                    CHECK(cudaMemcpyAsync(&(pd.disp[j])[ihpSize.p_begin[j][i]+k*ihpSize.t_size[j]],
                                          &(pd.disp[i])[ihpSize.h_begin[i][j]+k*ihpSize.t_size[i]],
                                          ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                          st.halo[i][j]));
                }
            }
        }
    }
    device_sync(GPUS);
    //cout<<"int time "<<(cpuSecond()-start)*1000<<endl;
}



static __global__ void kernel_cnn(
        int N, int NT, int Dim, real *mass, real dt, real *cn_xy, real *pforce, real *pforceold, real *velhalfold,
        real *disp, real *disp_xy)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i<N)
    {
        cn_xy[i] = 0.0;
        disp_xy[i] = 0.0;
        for (int j=0; j<Dim; j++) {
            if (fabs(velhalfold[i+j*NT]) > 0.0) {
                cn_xy[i] -=
                        1.0 * disp[i+j*NT] * disp[i+j*NT] * (pforce[i+j*NT] / mass[i] - pforceold[i+j*NT] / mass[i]) / (dt * velhalfold[i+j*NT]);
            }
            disp_xy[i] += disp[i+j*NT] * disp[i+j*NT];
        }
    }
}

__global__ void kernel_static_integrate(
        int N, int NT, int Dim, int ct, real cn, real dt, int begin, real *mass, real *pforce,  real *bforce,
        real *vel,  real *velhalfold,  real *disp, real *pforceold
)
{
    int	i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        i += begin;
        real velhalf=0.0;
        for (int j=0; j<Dim; j++) {
            if (ct == 0) {
                velhalf = 1.0 * dt / mass[i] * (pforce[i+j*NT] + bforce[i+j*NT]) / 2.0;
            } else {
                velhalf = ((2.0 - cn * dt) * velhalfold[i + j * NT] +
                           2.0 * dt / mass[i] * (pforce[i + j * NT] + bforce[i + j * NT])) /
                          (2.0 + cn * dt);

                vel[i+j*NT] = 0.5 * (velhalfold[i+j*NT] + velhalf);
                disp[i+j*NT] += velhalf * dt;
                velhalfold[i+j*NT] = velhalf;
                pforceold[i+j*NT] = pforce[i+j*NT];
                //if (fabs(bforce[i+j*NT])>1.0) printf("disp %e %e\n", velhalf, disp[i+j*NT]);

            }
        }
        //if (i==0) printf("%e %e %e\n", vel[i],bforce[i], disp[i]);
    }
}


real cal_cn(int GPUS, IHP_SIZE &ihpSize, Static_PD &pd)
{
    real cn = 0.0;
    real cn1 = 0.0;
    real cn2 = 0.0;
    for (int i=0; i<GPUS; i++) {
        CHECK(cudaSetDevice(i));
        cn1 += thrust::reduce(thrust::device, pd.cn_xy[i], pd.cn_xy[i] + ihpSize.ih_size[i]);
        cn2 += thrust::reduce(thrust::device, pd.disp_xy[i], pd.disp_xy[i] + ihpSize.ih_size[i]);
    }
    if(cn2 != 0.0)
    {
        if (cn1/cn2 > 0.0)
            cn = 2.0 * sqrt(cn1/cn2);
        else
            cn = 0.0;
    }
    else
        cn = 0.0;
    if (cn>2.0)
        cn = 1.9;
    return cn;
}

void static_integrate(int GPUS, int Dim, real dt, int ct, int **exchange_flag, State_Fatigue_PD &pd, IHP_SIZE &ihpSize, Stream &st, Grid &grid)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_cnn<<<grid.p_ih[i], block_size, 0, st.body[i]>>>(
                ihpSize.ih_size[i], ihpSize.t_size[i], Dim, pd.mass[i], dt, pd.cn_xy[i], pd.pforce[i],
                pd.pforceold[i], pd.velhalfold[i], pd.disp[i], pd.disp_xy[i]);
    }
    device_sync(GPUS);
    real cn = cal_cn(GPUS, ihpSize, pd);

    device_sync(GPUS);
    //cout<<"cn "<<cn<<endl;
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_static_integrate<<<grid.p_i[i], block_size, 0, st.body[i]>>>(
                ihpSize.i_size[i], ihpSize.t_size[i], Dim, ct, cn, dt, 0, pd.mass[i], pd.pforce[i], pd.bforce[i],
                pd.vel[i], pd.velhalfold[i], pd.disp[i], pd.pforceold[i]);
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                kernel_static_integrate<<<grid.p_h[i][j], block_size, 0, st.halo[i][j]>>>
                        (ihpSize.h_size[i][j], ihpSize.t_size[i], Dim, ct, cn, dt, ihpSize.h_begin[i][j], pd.mass[i], pd.pforce[i],
                         pd.bforce[i],
                         pd.vel[i], pd.velhalfold[i], pd.disp[i], pd.pforceold[i]);
                for (int k = 0; k < Dim; k++) {
                    CHECK(cudaMemcpyAsync(&(pd.disp[j])[ihpSize.p_begin[j][i] + k * ihpSize.t_size[j]],
                                          &(pd.disp[i])[ihpSize.h_begin[i][j] + k * ihpSize.t_size[i]],
                                          ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                          st.halo[i][j]));
                }
            }

        }
    }
    device_sync(GPUS);
}

void __global__ kernel_integrate_T(int N, real dt, real dens, real cv, int begin, real *energy, real *T, real *tbforce)
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<N)
    {
        i += begin;
        T[i] +=  (energy[i]+tbforce[i])*dt/dens/cv;
        // if (i==begin) printf("cnk %e %e %e %e\n",energy[i], tbforce[i], ccv[i], T[i]);
    }
}

void integrate_T_GPU(int GPUS , real dens, real cv, real dt, int **exchange_flag, IHP_SIZE &ihpSize,
                     Grid &grid, State_Thermal_Diffusion_PD2 &pd, Stream &st)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_integrate_T<<<grid.p_i[i], block_size, 0, st.body[i]>>>(
                ihpSize.i_size[i], dt, dens, cv, 0,  pd.energy[i], pd.T[i], pd.tbforce[i]);
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                kernel_integrate_T<<<grid.p_h[i][j], block_size, 0, st.halo[i][j]>>>
                        ( ihpSize.h_size[i][j], dt, dens, cv, ihpSize.h_begin[i][j], pd.energy[i], pd.T[i], pd.tbforce[i]);
                CHECK(cudaMemcpyAsync(&(pd.T[j])[ihpSize.p_begin[j][i]],
                                      &(pd.T[i])[ihpSize.h_begin[i][j]],
                                      ihpSize.h_size[i][j] * sizeof(real), cudaMemcpyDeviceToDevice,
                                      st.halo[i][j]));
            }
        }
    }
    device_sync(GPUS);
}


void integrate_CD_cpu(int N, real dens, real dt, int Dim, Bond_PD_CPU &pd, int omp)
{
    real t_acc = 0.0;
    if (omp==0)
    {
        for (int i=0; i<N; i++)
        {
            for (int j=0; j<Dim; j++)
            {
                pd.olddisp[N*j+i] = pd.disp[N*j+i];
                t_acc = (pd.pforce[N*j+i]+pd.bforce[N*j+i])/dens;
                pd.vel[N*j+i] += (pd.acc[N*j+i] + t_acc)*dt/2.0;
                pd.acc[N*j+i] = t_acc;
                pd.disp[N*j+i] += (dt*pd.vel[N*j+i]+t_acc*dt*dt/2);
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (t_acc)
        for (int i=0; i<N; i++)
        {
            for (int j=0; j<Dim; j++)
            {
                pd.olddisp[N*j+i] = pd.disp[N*j+i];
                t_acc = (pd.pforce[N*j+i]+pd.bforce[N*j+i])/dens;
                pd.vel[N*j+i] += (pd.acc[N*j+i] + t_acc)*dt/2.0;
                pd.acc[N*j+i] = t_acc;
                pd.disp[N*j+i] += (dt*pd.vel[N*j+i]+ t_acc*dt*dt/2);
            }
        }
    }
}

void integrate_CD_cpu(int N, real dens, real dt, int Dim, State_PD_CPU &pd, int omp)
{
    real t_acc = 0.0;
    if (omp==0)
    {
        for (int i=0; i<N; i++)
        {
            for (int j=0; j<Dim; j++)
            {
                pd.olddisp[N*j+i] = pd.disp[N*j+i];
                t_acc = (pd.pforce[N*j+i]+pd.bforce[N*j+i])/dens;
                pd.vel[N*j+i] += (pd.acc[N*j+i] + t_acc)*dt/2.0;
                pd.acc[N*j+i] = t_acc;
                pd.disp[N*j+i] += (dt*pd.vel[N*j+i]+t_acc*dt*dt/2);
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (t_acc)
        for (int i=0; i<N; i++)
        {
            for (int j=0; j<Dim; j++)
            {
                pd.olddisp[N*j+i] = pd.disp[N*j+i];
                t_acc = (pd.pforce[N*j+i]+pd.bforce[N*j+i])/dens;
                pd.vel[N*j+i] += (pd.acc[N*j+i] + t_acc)*dt/2.0;
                pd.acc[N*j+i] = t_acc;
                pd.disp[N*j+i] += (dt*pd.vel[N*j+i]+ t_acc*dt*dt/2);
            }
        }
    }
}