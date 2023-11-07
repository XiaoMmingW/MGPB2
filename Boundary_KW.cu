//
// Created by wxm on 2023/8/8.
//

#include "Boundary_KW.cuh"



__global__ void kernel_set_crack_KW(int N, int MN, real rad, int *NN, int *NL, int *fail, real *x)
{
    unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;
    unsigned int i = idx/MN;
    unsigned int j = idx%MN;
    unsigned int cnode = 0;
    if (i<N)
    {
        if(j<NN[i])
        {
            cnode = NL[idx];
            if((x[cnode]>-rad && x[i]< -rad) | (x[i]>-rad && x[cnode]< -rad))
            {
                if (x[cnode+N]>=-1.0e-10 | x[i+N]>=-1.0e-10)
                    fail[idx] = 0;
            }
            if( (x[cnode]>rad && x[i]< rad) | (x[i]>rad && x[cnode]< rad))
            {
                if (x[cnode+N]>=-1.0e-10 | x[i+N]>=-1.0e-10)
                    fail[idx] = 0;
            }
        }
    }
}


void set_crack_KW(int GPUS, int MN, real rad,  IHP_SIZE &ihpSize, Grid &grid, Base_PD &pd, cudaStream_t *st_body)
{
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        kernel_set_crack_KW<<<grid.b_t[i], block_size, 0, st_body[i]>>>
                (ihpSize.t_size[i], MN, rad, pd.NN[i], pd.NL[i], pd.fail[i], pd.x[i]);
    }
}


void cy_integrate(real dt, real force, Cylinder &cy)
{
    real t_acc = 0.0;
    t_acc = force/cy.mass;
    cy.vel += (t_acc+cy.acc)*dt;
    cy.acc = t_acc;
    cy.disp += dt*cy.vel+t_acc*dt*dt/2.0;
//    t_acc = force/cy.mass;
//    cy.vel += (t_acc)*dt;
//    cy.disp += dt*cy.vel;
}



__global__ void kernel_cy_contact(int N, int NT, real rad, real dt, real dens, real cy_x, real cy_disp, real *cy_force,
                                  real *x, real *disp, real *vel, real *vol, real *olddisp)
{
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i<N)
    {
        real dpeny = (cy_x + cy_disp) - (x[i+NT] + disp[i+NT]);
        real dpend = sqrt(square(x[i]+disp[i])+ square(x[i+NT*2]+disp[i+NT*2]));

        if (dpeny<=0.0 and dpend<rad)
        {
            disp[i+NT] += dpeny;
            real oldvel_y = vel[i+NT];
            vel[i+NT] = (disp[i+NT]-olddisp[i+NT])/dt;
            //printf("vel %e %e acc %e\n", vel[i+NT], oldvel[i+NT], vel[i+NT]-oldvel[i+NT]);
            //printf("vel %e %e acc %e\n", disp[i+NT], olddisp[i+NT], disp[i+NT]-olddisp[i+NT]);
            atomicAdd2(&cy_force[0], -dens*(vel[i+NT]-oldvel_y)/dt*vol[i]);

        }
    }
}

void cy_contact_and_integrate(int GPUS, real rad, real dt, real dens, IHP_SIZE &ihpSize, Grid &grid, Mech_PD &pd,
                              cudaStream_t *st_body, Cylinder &cy)
{
    real force[GPUS];

    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMemset(cy.force[i], 0, sizeof(real)));
        kernel_cy_contact<<<grid.p_ih[i], block_size, 0, st_body[i]>>>(
                ihpSize.ih_size[i], ihpSize.t_size[i], rad, dt, dens, cy.x, cy.disp,cy.force[i],
                pd.x[i],pd.disp[i], pd.vel[i], pd.vol[i], pd.olddisp[i]);
        CHECK(cudaMemcpy(&force[i], cy.force[i], sizeof(real), cudaMemcpyDeviceToHost));
    }
    real force_sum = 0.0;
    for (int i = 0; i < GPUS; i++) {
        force_sum += force[i];
    }
    //cout<<"force "<<force_sum<<endl;
    cy_integrate(dt, force_sum, cy);
    device_sync(GPUS);
}

void set_crack_KW_cpu(int N, int MN, real rad, Bond_PD_CPU &pd, int omp)
{
    int idx = 0;
    unsigned int cnode = 0;
    if (omp == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                if ((pd.x[cnode] > -rad && pd.x[i] < -rad) | (pd.x[i] > -rad && pd.x[cnode] < -rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
                if ((pd.x[cnode] > rad && pd.x[i] < rad) | (pd.x[i] > rad && pd.x[cnode] < rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
            }
        }
    }
    else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                if ((pd.x[cnode] > -rad && pd.x[i] < -rad) | (pd.x[i] > -rad && pd.x[cnode] < -rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
                if ((pd.x[cnode] > rad && pd.x[i] < rad) | (pd.x[i] > rad && pd.x[cnode] < rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
            }
        }
    }
}

void cy_contact_cpu(int N, real rad, real dt, real dens, real &cy_x, real &cy_disp, real &cy_force, Bond_PD_CPU &pd, int omp)
{
    real dpeny, dpend, oldvel_y;
    if (omp == 0) {
        for (int i = 0; i < N; i++) {
            dpeny = (cy_x + cy_disp) - (pd.x[i + N] + pd.disp[i + N]);
            dpend = sqrt(square(pd.x[i] + pd.disp[i]) + square(pd.x[i + N * 2] + pd.disp[i + N * 2]));
            if (dpeny < 0.0 and dpend < rad) {
                oldvel_y = pd.vel[i + N];
                pd.disp[i + N] += dpeny;
                pd.vel[i + N] = (pd.disp[i+N]-pd.olddisp[i+N]) / dt;
                cy_force -= dens * (pd.vel[i + N] - oldvel_y) / dt * pd.vol[i];
            }
        }
    } else if (omp==1){
#pragma omp parallel for private (dpeny, dpend, oldvel_y)
        for (int i = 0; i < N; i++) {
            dpeny = (cy_x + cy_disp) - (pd.x[i + N] + pd.disp[i + N]);
            dpend = sqrt(square(pd.x[i] + pd.disp[i]) + square(pd.x[i + N * 2] + pd.disp[i + N * 2]));
            if (dpeny < 0.0 and dpend < rad) {
                oldvel_y = pd.vel[i + N];
                pd.disp[i + N] += dpeny;
                pd.vel[i + N] = (pd.disp[i+N]-pd.olddisp[i+N]) / dt;
#pragma omp atomic update
                cy_force -= dens * (pd.vel[i + N] - oldvel_y) / dt * pd.vol[i];
            }
        }
    }
}

void cy_contact_and_integrate_cpu(int N, real rad, real dt, real dens, Cylinder &cy, Bond_PD_CPU &pd, int omp)
{
    real force = 0.0;
    cy_contact_cpu(N, rad, dt, dens, cy.x, cy.disp, force, pd, omp);
    cout<<"force "<<force<<endl;
    cy_integrate(dt, force, cy);
}

void set_crack_KW_cpu(int N, int MN, real rad, State_PD_CPU &pd, int omp)
{
    int idx = 0;
    unsigned int cnode = 0;
    if (omp == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                if ((pd.x[cnode] > -rad && pd.x[i] < -rad) | (pd.x[i] > -rad && pd.x[cnode] < -rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
                if ((pd.x[cnode] > rad && pd.x[i] < rad) | (pd.x[i] > rad && pd.x[cnode] < rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
            }
        }
    }
    else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                if ((pd.x[cnode] > -rad && pd.x[i] < -rad) | (pd.x[i] > -rad && pd.x[cnode] < -rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
                if ((pd.x[cnode] > rad && pd.x[i] < rad) | (pd.x[i] > rad && pd.x[cnode] < rad)) {
                    if (pd.x[cnode + N] >= -1.0e-10 | pd.x[i + N] >= -1.0e-10)
                        pd.fail[idx] = 0;
                }
            }
        }
    }
}

void cy_contact_cpu(int N, real rad, real dt, real dens, real &cy_x, real &cy_disp, real &cy_force, State_PD_CPU &pd, int omp)
{
    real dpeny, dpend, oldvel_y;
    if (omp == 0) {
        for (int i = 0; i < N; i++) {
            dpeny = (cy_x + cy_disp) - (pd.x[i + N] + pd.disp[i + N]);
            dpend = sqrt(square(pd.x[i] + pd.disp[i]) + square(pd.x[i + N * 2] + pd.disp[i + N * 2]));
            if (dpeny < 0.0 and dpend < rad) {
                oldvel_y = pd.vel[i + N];
                pd.disp[i + N] += dpeny;
                pd.vel[i + N] = (pd.disp[i+N]-pd.olddisp[i+N]) / dt;
                cy_force -= dens * (pd.vel[i + N] - oldvel_y) / dt * pd.vol[i];
            }
        }
    } else if (omp==1){
#pragma omp parallel for private (dpeny, dpend, oldvel_y)
        for (int i = 0; i < N; i++) {
            dpeny = (cy_x + cy_disp) - (pd.x[i + N] + pd.disp[i + N]);
            dpend = sqrt(square(pd.x[i] + pd.disp[i]) + square(pd.x[i + N * 2] + pd.disp[i + N * 2]));
            if (dpeny < 0.0 and dpend < rad) {
                oldvel_y = pd.vel[i + N];
                pd.disp[i + N] += dpeny;
                pd.vel[i + N] = (pd.disp[i+N]-pd.olddisp[i+N]) / dt;
#pragma omp atomic update
                cy_force -= dens * (pd.vel[i + N] - oldvel_y) / dt * pd.vol[i];
            }
        }
    }
}

void cy_contact_and_integrate_cpu(int N, real rad, real dt, real dens, Cylinder &cy, State_PD_CPU &pd, int omp)
{
    real force = 0.0;
    cy_contact_cpu(N, rad, dt, dens, cy.x, cy.disp, force, pd, omp);
    cout<<"force "<<force<<endl;
    cy_integrate(dt, force, cy);
}

