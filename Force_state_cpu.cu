//
// Created by wxm on 2023/9/1.
//

#include "Force_state_cpu.cuh"

void vol_corr_cpu(int N, int MN, int Dim, State_PD_CPU &pd, int omp)
{
    int cnode = 0;
    real delta = 0.0;
    int idx = 0;
    if (omp==0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                delta = horizon*pd.dx[i];
                if (Dim==2)
                {
                    pd.idist[idx]=sqrt(square(pd.x[cnode]-pd.x[i])+
                                       square(pd.x[cnode+N]-pd.x[i+N]));
                } else if (Dim==3)
                {
                    pd.idist[idx]=sqrt(square(pd.x[cnode]-pd.x[i])+
                                       square(pd.x[cnode+N]-pd.x[i+N]) +
                                       square(pd.x[cnode+N*2]-pd.x[i+N*2]));
                }

                if (pd.idist[idx] <= delta-pd.dx[cnode]/2.0)
                    pd.fac[idx] = 1.0;
                else if (pd.idist[idx] <= delta+pd.dx[cnode]/2.0)
                    pd.fac[idx] = (delta+pd.dx[cnode]/2.0-pd.idist[idx]) / pd.dx[cnode];
                else
                    pd.fac[idx] = 0.0;
            }

        }
    } else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx, delta)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                delta = horizon*pd.dx[i];
                if (Dim==2)
                {
                    pd.idist[idx]=sqrt(square(pd.x[cnode]-pd.x[i])+
                                       square(pd.x[cnode+N]-pd.x[i+N]));
                } else if (Dim==3)
                {
                    pd.idist[idx]=sqrt(square(pd.x[cnode]-pd.x[i])+
                                       square(pd.x[cnode+N]-pd.x[i+N]) +
                                       square(pd.x[cnode+N*2]-pd.x[i+N*2]));
                }

                if (pd.idist[idx] <= delta-pd.dx[cnode]/2.0)
                    pd.fac[idx] = 1.0;
                else if (pd.idist[idx] <= delta+pd.dx[cnode]/2.0)
                    pd.fac[idx] = (delta+pd.dx[cnode]/2.0-pd.idist[idx]) / pd.dx[cnode];
                else
                    pd.fac[idx] = 0.0;
                //if (i==0) cout<<"fac "<<pd.fac[idx]<<endl;
            }
        }
    }
}

void cal_weight(
        int N, int MN, real horizon, State_PD_CPU &pd, int omp)
{
    int cnode = 0;
    int idx = 0;
    real delta = 0.0;
    if (omp==0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                delta = horizon * pd.dx[cnode];
                pd.w[idx] = exp(-square(pd.idist[idx]) / square(delta));
                pd.m[i] += pd.w[idx] * square(pd.idist[idx]) * pd.vol[cnode] * pd.fac[idx];
                //if (cnode==0 and i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode);
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx, delta)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                delta = horizon * pd.dx[cnode];
                pd.w[idx] = exp(-square(pd.idist[idx]) / square(delta));
                pd.m[i] += pd.w[idx] * square(pd.idist[idx]) * pd.vol[cnode] * pd.fac[idx];
                //if (cnode==0 and i==0) printf("i %d cnode %d ff %e %e %e\n", i, cnode);
            }
        }
    }
}

void cal_theta_3D(int N, int NT, int MN, State_PD_CPU &pd, int omp)
{
    int cnode = 0;
    int idx = 0;
    real nlength = 0.0;
    if (omp==0) {
        for (int i = 0; i < N; i++) {
            pd.theta[i] = 0.0;
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                nlength = sqrt(square(pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i]) +
                                    square(pd.x[NT + cnode] - pd.x[NT + i] + pd.disp[NT + cnode] -
                                           pd.disp[NT + i]) +
                                    square(pd.x[NT * 2 + cnode] - pd.x[NT * 2 + i] + pd.disp[NT * 2 + cnode] -
                                           pd.disp[NT * 2 + i]));
                if (pd.fail[idx] == 1) {
                    pd.theta[i] +=
                            3.0 / pd.m[i] * pd.w[idx] * pd.idist[idx] * (nlength - pd.idist[idx]) * pd.fac[idx] *
                            pd.vol[cnode];
                }
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx, nlength)
        for (int i = 0; i < N; i++) {
            pd.theta[i] = 0.0;
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                nlength = sqrt(square(pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i]) +
                                    square(pd.x[NT + cnode] - pd.x[NT + i] + pd.disp[NT + cnode] -
                                           pd.disp[NT + i]) +
                                    square(pd.x[NT * 2 + cnode] - pd.x[NT * 2 + i] + pd.disp[NT * 2 + cnode] -
                                           pd.disp[NT * 2 + i]));
                if (pd.fail[idx] == 1) {
                    pd.theta[i] +=
                            3.0 / pd.m[i] * pd.w[idx] * pd.idist[idx] * (nlength - pd.idist[idx]) * pd.fac[idx] *
                            pd.vol[cnode];
                }
            }
        }
    }
}



void state_force_3D(int N, real K, real G, int MN, real sc, State_PD_CPU &pd, int omp)
{
    unsigned int cnode = 0;
    int idx = 0;
    real ex, ey, ez, s, nx, ny, nz;
    real nlength, t;
    if (omp == 0){
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                ex = pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i];
                ey = pd.x[N + cnode] - pd.x[N + i] + pd.disp[N + cnode] - pd.disp[N + i];
                ez = pd.x[N * 2 + cnode] - pd.x[N * 2 + i] + pd.disp[N * 2 + cnode] - pd.disp[N * 2 + i];
                nlength = sqrt(ex * ex + ey * ey + ez * ez);
                nx = ex / nlength;
                ny = ey / nlength;
                nz = ez / nlength;
                if (pd.fail[idx]) {
                    real s = (nlength - pd.idist[idx]) / pd.idist[idx];
                    if (s > sc and fabs(pd.x[i]) > 0.02 and fabs(pd.x[cnode]) > 0.02) pd.fail[idx] = 0;
                    real ed = nlength - pd.idist[idx] - pd.theta[i] * pd.idist[idx] / 3.0;
                    real t = ((3.0 * K * pd.theta[i]) / pd.m[i] * pd.w[idx] * pd.idist[idx] +
                              15.0 * G / pd.m[i] * ed * pd.w[idx]);
                    pd.pforce[i] += t * nx * pd.vol[cnode];
                    pd.pforce[cnode] -= t * nx * pd.vol[i];
                    pd.pforce[i + N] += t * ny * pd.vol[cnode];
                    pd.pforce[cnode + N] -= t * ny * pd.vol[i];
                    pd.pforce[i + N * 2] += t * nz * pd.vol[cnode];
                    pd.pforce[cnode + N * 2] -= t * nz * pd.vol[i];
                }
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx, nlength, ex, ey, ez, s, nx, ny, nz, t)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                ex = pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i];
                ey = pd.x[N + cnode] - pd.x[N + i] + pd.disp[N + cnode] - pd.disp[N + i];
                ez = pd.x[N * 2 + cnode] - pd.x[N * 2 + i] + pd.disp[N * 2 + cnode] - pd.disp[N * 2 + i];
                nlength = sqrt(ex * ex + ey * ey + ez * ez);
                nx = ex / nlength;
                ny = ey / nlength;
                nz = ez / nlength;
                if (pd.fail[idx]) {
                    real s = (nlength - pd.idist[idx]) / pd.idist[idx];
                    if (s > sc and fabs(pd.x[i]) > 0.02 and fabs(pd.x[cnode]) > 0.02) pd.fail[idx] = 0;
                    real ed = nlength - pd.idist[idx] - pd.theta[i] * pd.idist[idx] / 3.0;
                    real t = ((3.0 * K * pd.theta[i]) / pd.m[i] * pd.w[idx] * pd.idist[idx] +
                              15.0 * G / pd.m[i] * ed * pd.w[idx]);
#pragma omp atomic update
                    pd.pforce[i] += t * nx * pd.vol[cnode];
#pragma omp atomic update
                    pd.pforce[cnode] -= t * nx * pd.vol[i];
#pragma omp atomic update
                    pd.pforce[i + N] += t * ny * pd.vol[cnode];
#pragma omp atomic update
                    pd.pforce[cnode + N] -= t * ny * pd.vol[i];
#pragma omp atomic update
                    pd.pforce[i + N * 2] += t * nz * pd.vol[cnode];
#pragma omp atomic update
                    pd.pforce[cnode + N * 2] -= t * nz * pd.vol[i];
                }
            }
        }
    }
}