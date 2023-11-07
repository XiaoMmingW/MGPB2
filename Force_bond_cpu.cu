//
// Created by wxm on 2023/8/14.
//

#include "Force_bond_cpu.cuh"


void vol_corr_cpu(int N, int MN, int Dim, Bond_PD_CPU &pd, int omp)
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

void cal_bc_cpu(int N, int Dim, real E, real pratio, real pi, real horizon, real thick, Bond_PD_CPU &pd, int omp)
{

    if (omp==0) {
        for (int i = 0; i < N; i++) {
            real delta = horizon * pd.dx[i];
            if (Dim == 2) {
                pd.bc[i] = 3.0 * E / (pi * thick * cubic(delta) * (1.0 - pratio));
            } else if (Dim == 3) {
                pd.bc[i] = 3.0 * E / (pi * delta * cubic(delta) * (1.0 - 2.0 * pratio));
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            real delta = horizon * pd.dx[i];
            if (Dim == 2) {
                pd.bc[i] = 3.0 * E / (pi * thick * cubic(delta) * (1.0 - pratio));
            } else if (Dim == 3) {
                pd.bc[i] = 3.0 * E / (pi * delta * cubic(delta) * (1.0 - 2.0 * pratio));
            }
        }
    }
}


void Disp_cpu(int N, int select, Bond_PD_CPU &pd, int omp)
{
    if (omp==0) {
        for (int i = 0; i < N; i++) {
            pd.disp[select * N + i] = 0.001 * pd.x[select * N + i];
        }
    } else if (omp==1) {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            pd.disp[select * N + i] = 0.001 * pd.x[select * N + i];
        }
    }
}

void surface_F_cpu(int N, int MN, int select, int Dim, Bond_PD_CPU &pd, real *fncst, int omp)
{

    int cnode = 0;
    int idx = 0;
    real nlength =0.0;
    real stendens = 0.0;
    if (omp==0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                if (Dim == 2) {
                    nlength = sqrt(square(pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i]) +
                                   square(pd.x[cnode + N] - pd.x[i + N] + pd.disp[cnode + N] - pd.disp[i + N]));
                } else if (Dim == 3) {
                    nlength = sqrt(square(pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i]) +
                                   square(pd.x[cnode + N] - pd.x[i + N] + pd.disp[cnode + N] - pd.disp[i + N]) +
                                   square(pd.x[cnode + N * 2] - pd.x[i + N * 2] + pd.disp[cnode + N * 2] -
                                          pd.disp[i + N * 2]));
                }
                stendens = 0.25 * pd.bc[i] * square(nlength - pd.idist[idx]) / pd.idist[idx] * pd.fac[idx];
                fncst[i + select * N] += stendens * pd.vol[cnode];
                fncst[cnode + select * N] += stendens * pd.vol[i];
                //if (i==0) printf("i %d cnode %d ff %e %e %e %e \n", i, cnode, stendens_i, stendens_j, bc[i], idist[idx]);
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for private (cnode, idx, nlength, stendens)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i * MN + j;
                cnode = pd.NL[idx];
                if (Dim == 2) {
                    nlength = sqrt(square(pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i]) +
                                   square(pd.x[cnode + N] - pd.x[i + N] + pd.disp[cnode + N] - pd.disp[i + N]));
                } else if (Dim == 3) {
                    nlength = sqrt(square(pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i]) +
                                   square(pd.x[cnode + N] - pd.x[i + N] + pd.disp[cnode + N] - pd.disp[i + N]) +
                                   square(pd.x[cnode + N * 2] - pd.x[i + N * 2] + pd.disp[cnode + N * 2] -
                                          pd.disp[i + N * 2]));
                }
                stendens = 0.25 * pd.bc[i] * square(nlength - pd.idist[idx]) / pd.idist[idx] * pd.fac[idx];
#pragma omp atomic update
                fncst[i + select * N] += stendens * pd.vol[cnode];
#pragma omp atomic update
                fncst[cnode + select * N] += stendens * pd.vol[i];
                //if (i==0) printf("i %d cnode %d ff %e %e %e %e \n", i, cnode, stendens_i, stendens_j, bc[i], idist[idx]);
            }
        }
    }
}

void cal_fncst_cpu(int N, real sedload, int Dim, real* fncst, int omp)
{
    if (omp==0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < Dim; j++) {
                fncst[i + j * N] = sedload / fncst[i + j * N];
                if (i==0) printf("i %d Dim %d %d ff %e \n", i, j, N, fncst[i + j * N]);
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < Dim; j++) {
                fncst[i + j * N] = sedload / fncst[i + j * N];
                if (i==0) printf("i %d Dim %d %d ff %e \n", i, j, N, fncst[i + j * N]);
            }
        }
    }
}

void cal_surf_coff_F_cpu(int N, int MN, int Dim, Bond_PD_CPU &pd, real *fncst, int omp)
{
    int cnode = 0;
    int idx = 0;
    if (omp==0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                if (Dim == 1) {
                    pd.scr[idx] = (fncst[i] + fncst[cnode]) / 2.0;
                }
                if (Dim == 2) {
                    real theta = 0.0;
                    real scx = 0.0;
                    real scy = 0.0;
                    if (fabs(pd.x[cnode + N] - pd.x[i + N]) <= 1.0e-10)
                        theta = 0.0;
                    else if (fabs(pd.x[cnode] - pd.x[i]) <= 1.0e-10)
                        theta = 90.0 * pi / 180.0;
                    else
                        theta = atan(fabs(pd.x[cnode + N] - pd.x[i + N]) / fabs(pd.x[cnode] - pd.x[i]));
                    scx = (fncst[i] + fncst[cnode]) / 2.0;
                    scy = (fncst[i + N] + fncst[cnode + N]) / 2.0;
                    pd.scr[idx] = sqrt(
                            1.0 / (cos(theta) * cos(theta) / (scx * scx) + sin(theta) * sin(theta) / (scy * scy)));
                }
                if (Dim == 3) {
                    real theta = 0.0;
                    real scx = 0.0;
                    real scy = 0.0;
                    real scz = 0.0;

                    if (fabs(pd.x[cnode + 2 * N] - pd.x[i + 2 * N]) < 1.0e-10) {
                        if (fabs(pd.x[cnode + N] - pd.x[i + N]) < 1.0e-10)
                            theta = 0.0;
                        else if (fabs(pd.x[cnode] - pd.x[i]) < 1.0e-10)
                            theta = 90.0 * pi / 180.0;
                        else
                            theta = atan(fabs(pd.x[cnode + N] - pd.x[i + N]) / fabs(pd.x[cnode] - pd.x[i]));
                        real phi = 90.0 * pi / 180.0;
                        scx = (fncst[i] + fncst[cnode]) / 2.0;
                        scy = (fncst[i + N] + fncst[cnode + N]) / 2.0;
                        scz = (fncst[i + 2 * N] + fncst[cnode + 2 * N]) / 2.0;
                        //scr[idx] = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                        pd.scr[idx] = sqrt(
                                1.0 / (cos(theta) * cos(theta) / (scx * scx) + sin(theta) * sin(theta) / (scy * scy) +
                                       cos(phi) * cos(phi) / (scz * scz)));
                    } else if (fabs(pd.x[cnode] - pd.x[i]) < 1.0e-10 && fabs(pd.x[cnode + N] - pd.x[i + N]) < 1.0e-10)
                        pd.scr[idx] = (fncst[i + 2 * N] + fncst[cnode + 2 * N]) / 2.0;
                    else {
                        theta = atan(fabs(pd.x[cnode + N] - pd.x[i + N]) / fabs(pd.x[cnode] - pd.x[i]));
                        real phi = acos(fabs(pd.x[cnode + 2 * N] - pd.x[i + 2 * N]) / pd.idist[idx]);
                        scx = (fncst[i] + fncst[cnode]) / 2.0;
                        scy = (fncst[i + N] + fncst[cnode + N]) / 2.0;
                        scz = (fncst[i + 2 * N] + fncst[cnode + 2 * N]) / 2.0;
                        //scr[idx] = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                        pd.scr[idx] = sqrt(
                                1.0 / (cos(theta) * cos(theta) / (scx * scx) + sin(theta) * sin(theta) / (scy * scy) +
                                       cos(phi) * cos(phi) / (scz * scz)));
                    }
                }
                // if (scr[idx]==NAN) printf("i %d Dim %d ff %e \n", i, j, scr[idx]);
            }
        }
    } else if (omp==1)
    {
#pragma omp parallel for private(cnode, idx)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                if (Dim == 1) {
                    pd.scr[idx] = (fncst[i] + fncst[cnode]) / 2.0;
                }
                if (Dim == 2) {
                    real theta = 0.0;
                    real scx = 0.0;
                    real scy = 0.0;
                    if (fabs(pd.x[cnode + N] - pd.x[i + N]) <= 1.0e-10)
                        theta = 0.0;
                    else if (fabs(pd.x[cnode] - pd.x[i]) <= 1.0e-10)
                        theta = 90.0 * pi / 180.0;
                    else
                        theta = atan(fabs(pd.x[cnode + N] - pd.x[i + N]) / fabs(pd.x[cnode] - pd.x[i]));
                    scx = (fncst[i] + fncst[cnode]) / 2.0;
                    scy = (fncst[i + N] + fncst[cnode + N]) / 2.0;
                    pd.scr[idx] = sqrt(
                            1.0 / (cos(theta) * cos(theta) / (scx * scx) + sin(theta) * sin(theta) / (scy * scy)));
                }
                if (Dim == 3) {
                    real theta = 0.0;
                    real scx = 0.0;
                    real scy = 0.0;
                    real scz = 0.0;

                    if (fabs(pd.x[cnode + 2 * N] - pd.x[i + 2 * N]) < 1.0e-10) {
                        if (fabs(pd.x[cnode + N] - pd.x[i + N]) < 1.0e-10)
                            theta = 0.0;
                        else if (fabs(pd.x[cnode] - pd.x[i]) < 1.0e-10)
                            theta = 90.0 * pi / 180.0;
                        else
                            theta = atan(fabs(pd.x[cnode + N] - pd.x[i + N]) / fabs(pd.x[cnode] - pd.x[i]));
                        real phi = 90.0 * pi / 180.0;
                        scx = (fncst[i] + fncst[cnode]) / 2.0;
                        scy = (fncst[i + N] + fncst[cnode + N]) / 2.0;
                        scz = (fncst[i + 2 * N] + fncst[cnode + 2 * N]) / 2.0;
                        //scr[idx] = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                        pd.scr[idx] = sqrt(
                                1.0 / (cos(theta) * cos(theta) / (scx * scx) + sin(theta) * sin(theta) / (scy * scy) +
                                       cos(phi) * cos(phi) / (scz * scz)));
                    } else if (fabs(pd.x[cnode] - pd.x[i]) < 1.0e-10 && fabs(pd.x[cnode + N] - pd.x[i + N]) < 1.0e-10)
                        pd.scr[idx] = (fncst[i + 2 * N] + fncst[cnode + 2 * N]) / 2.0;
                    else {
                        theta = atan(fabs(pd.x[cnode + N] - pd.x[i + N]) / fabs(pd.x[cnode] - pd.x[i]));
                        real phi = acos(fabs(pd.x[cnode + 2 * N] - pd.x[i + 2 * N]) / pd.idist[idx]);
                        scx = (fncst[i] + fncst[cnode]) / 2.0;
                        scy = (fncst[i + N] + fncst[cnode + N]) / 2.0;
                        scz = (fncst[i + 2 * N] + fncst[cnode + 2 * N]) / 2.0;
                        //scr[idx] = (fncst[i+2*N] + fncst[cnode+2*N]) / 2.0;
                        pd.scr[idx] = sqrt(
                                1.0 / (cos(theta) * cos(theta) / (scx * scx) + sin(theta) * sin(theta) / (scy * scy) +
                                       cos(phi) * cos(phi) / (scz * scz)));
                    }
                }
                //if (i==0) printf("i %d Dim %d ff %e \n", i, j, pd.scr[idx]);
            }
        }
    }
}

void surface_correct_cpu(int N, int MN, int Dim, real E, real pratio, real sedload, real thick, Bond_PD_CPU &pd, int omp)
{
    cal_bc_cpu(N, Dim, E, pratio, pi,  horizon, thick, pd, omp);
    real *fncst = new real [N*Dim];
    memset(fncst, 0, N*Dim*sizeof(real));
    for (int i=0; i<Dim; i++)
    {
        Disp_cpu(N, i, pd, omp);
        surface_F_cpu(N, MN, i, Dim, pd,fncst, omp);
        memset(pd.disp, 0, N*Dim*sizeof(real));
    }
    cal_fncst_cpu(N, sedload, Dim, fncst, omp);
    cal_surf_coff_F_cpu(N, MN,  Dim, pd, fncst, omp);
}


void bond_force_cpu(int N, int MN, const int Dim, real horizon, real pi, real sc, Bond_PD_CPU &pd, int omp) {
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
                    s = (nlength - pd.idist[idx]) / pd.idist[idx];
                    if ((s) >= sc) pd.fail[idx] = 0;
                    t = pd.bc[i] * pd.scr[idx] * s * pd.fac[idx];
                    pd.pforce[i] += t * nx * pd.vol[cnode];
                    pd.pforce[cnode] -= t * nx * pd.vol[i];
                    pd.pforce[i + N] += t * ny * pd.vol[cnode];
                    pd.pforce[cnode + N] -= t * ny * pd.vol[i];
                    if (Dim == 3) {
                        pd.pforce[i + N * 2] += t * nz * pd.vol[cnode];
                        pd.pforce[cnode + N * 2] -= t * nz * pd.vol[i];
                    }
                }
            }
        }
    } else if (omp == 1){
#pragma omp parallel for private (cnode, idx, nlength, ex, ey, ez, s, nx, ny, nz, t)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < pd.NN[i]; j++) {
                idx = i*MN+j;
                cnode = pd.NL[idx];
                ex = pd.x[cnode] - pd.x[i] + pd.disp[cnode] - pd.disp[i];
                ey = pd.x[N + cnode] - pd.x[N + i] + pd.disp[N + cnode] - pd.disp[N + i];
                if (Dim == 3)
                ez = pd.x[N * 2 + cnode] - pd.x[N * 2 + i] + pd.disp[N * 2 + cnode] - pd.disp[N * 2 + i];
                nlength = sqrt(ex * ex + ey * ey + ez * ez);
                nx = ex / nlength;
                ny = ey / nlength;
                nz = ez / nlength;
                if (pd.fail[idx]) {
                    s = (nlength - pd.idist[idx]) / pd.idist[idx];
                    if ((s) >= sc) pd.fail[idx] = 0;
                    //t = pd.bc[i] * 1 * s * pd.fac[idx];
                    t = pd.bc[i] * pd.scr[idx] * s * pd.fac[idx];
#pragma omp atomic update
                    pd.pforce[i] += t * nx * pd.vol[cnode];
#pragma omp atomic update
                    pd.pforce[cnode] -= t * nx * pd.vol[i];
#pragma omp atomic update
                    pd.pforce[i + N] += t * ny * pd.vol[cnode];
#pragma omp atomic update
                    pd.pforce[cnode + N] -= t * ny * pd.vol[i];
                    if (Dim == 3) {
#pragma omp atomic update
                        pd.pforce[i + N * 2] += t * nz * pd.vol[cnode];
#pragma omp atomic update
                        pd.pforce[cnode + N * 2] -= t * nz * pd.vol[i];
                    }
                }
            }

        }
    }
}

