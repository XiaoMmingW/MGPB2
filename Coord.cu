//
// Created by wxm on 2023/6/19.
//

#include "Coord.cuh"

__global__ void kernel_coord_plate_crack(real *x, real *dxx, real *voll, int N, int nx, int ny, real dx, real vol) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        x[idx] = (-nx / 2.0 + 0.5 + idx % nx) * dx;
        x[idx + N] = (-ny / 2.0 + 0.5 + idx / nx) * dx;
        voll[idx] = vol;
        dxx[idx] = dx;
    }

}

void coord_plate(real *x, real *dxx, real *voll, int N, int nx, int ny, real dx, real vol)
{
    int idx = 0;
    for (int i=0; i<ny; i++) {
        for (int j = 0; j < nx; j++) {
            idx = i*nx + j;
            x[idx] = (-nx/2.0+0.5+j)*dx;
            x[idx+N] = (-ny/2.0+0.5+i)*dx;
            dxx[idx] = dx;
            voll[idx] = vol;
        }
    }
}

void data_segm_2D(int N, int GPUS, int segm_num, real *k, real *b, real *x, real *dx, POINT_ID pid[], IHP_SIZE &ihpSize,
                  int **exchange_flag)
{
    real delta = 0.0;
    real dist = 0.0, dist2 = 0.0;
    real kk=0.0, bb=0.0, kk2=0.0, bb2=0.0;
    real y = 0.0;
    for (int i=0; i<N; i++)   //遍历点
    {
        delta = horizon * dx[i];
        y = x[i + N];
        //cout<<"x"<<x[i]<<" "<<y<<endl;
        if (GPUS == 1) {
            pid[0].internal[ihpSize.i_size[0]] = i;
            ihpSize.i_size[0]++;
        } else {
            for (int j = 0; j < GPUS; j++) //遍历GPUS
            {

                if (j == 0) {
                    kk = k[j];
                    bb = b[j];
                    if (kk * x[i] + bb <= y) {
                        dist = fabs(kk * x[i] - y + bb) / sqrt(1.0 + kk * kk);
                        if (dist <= delta) {
                            pid[j].halo[j + 1][ihpSize.h_size[j][j + 1]] = i;
                            ihpSize.h_size[j][j + 1]++;

                        } else {
                            pid[j].internal[ihpSize.i_size[j]] = i;
                            ihpSize.i_size[j]++;

                        }
                    }


                } else if (j == GPUS - 1) {
                    kk = k[j - 1];
                    bb = b[j - 1];
                    if (kk * x[i] + bb >= y) {
                        dist = fabs(kk * x[i] - y + bb) / sqrt(1.0 + kk * kk);
                        if (dist <= delta) {
                            pid[j].halo[j - 1][ihpSize.h_size[j][j - 1]] = i;
                            ihpSize.h_size[j][j - 1]++;

                        } else {
                            pid[j].internal[ihpSize.i_size[j]] = i;
                            ihpSize.i_size[j]++;
                        }
                    }
                } else {

                    kk = k[j - 1];
                    bb = b[j - 1];
                    kk2 = k[j];
                    bb2 = b[j];
                    if (kk * x[i] + bb > y and kk2 * x[i] + bb2 < y) {
                        dist = fabs(kk * x[i] - y + bb) / sqrt(1.0 + kk * kk);
                        dist2 = fabs(kk2 * x[i] - y + bb2) / sqrt(1.0 + kk2 * kk2);
                        if (dist <= delta) {
                            pid[j].halo[j - 1][ihpSize.h_size[j][j - 1]] = i;
                            ihpSize.h_size[j][j - 1]++;

                        } else if (dist2 <= delta) {
                            pid[j].halo[j + 1][ihpSize.h_size[j][j + 1]] = i;
                            ihpSize.h_size[j][j + 1]++;
                        } else {
                            pid[j].internal[ihpSize.i_size[j]] = i;
                            ihpSize.i_size[j]++;
                        }
                    }
                }
            }
        }


        for (int i = 0; i < GPUS; i++) {
            for (int j = 0; j < GPUS; j++) {
                if (exchange_flag[i][j] == 1) {
                    ihpSize.p_size[j][i] = ihpSize.h_size[i][j];
                    memcpy(pid[j].padding[i], pid[i].halo[j], ihpSize.h_size[i][j] * sizeof(int));
                }
            }
        }
    }


//
    for (int i=0; i<GPUS; i++)
    {
        ihpSize.ih_size[i] = ihpSize.i_size[i];
        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[i][j]==1) {
                ihpSize.ih_size[i] += ihpSize.h_size[i][j];
                cout<<"hsize "<<ihpSize.h_size[i][j]<<endl;
                ihpSize.h_begin[i][j] = ihpSize.i_size[i];
                for (int k = 0; k < j; k++) {
                    if (exchange_flag[i][k] == 1) {
                        ihpSize.h_begin[i][j] += ihpSize.h_size[i][k];
                    }
                }
            }
        }
    }
//
    for (int i=0; i<GPUS; i++)
    {
        ihpSize.t_size[i] = ihpSize.ih_size[i];
        cout<<"tsize "<<ihpSize.t_size[i]<<endl;
        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[i][j]==1) {
                ihpSize.t_size[i] += ihpSize.p_size[i][j];
                ihpSize.p_begin[i][j] = ihpSize.ih_size[i];
                for (int k = 0; k < j; k++) {
                    if (exchange_flag[i][k] == 1) {
                        ihpSize.p_begin[i][j] += ihpSize.p_size[i][k];
                    }
                }
            }
        }
    }
}
//
//



void coord_transfer(int N, int GPUS, int Dim, real *x, real *dx, real *vol, POINT_ID pid1[],
                    IHP_SIZE &ihpSize, int **exchange_flag, Base_PD &pd)
{
    for (int k=0; k<GPUS; k++)
    {
        real *tx = new real [ihpSize.t_size[k]*Dim];
        real *tdx = new real [ihpSize.t_size[k]];
        real *tvol = new real [ihpSize.t_size[k]];
        int idx = 0;
        for (idx=0; idx<ihpSize.i_size[k]; idx++)
        {
            tx[idx] = x[pid1[k].internal[idx]];
            tx[idx+ihpSize.t_size[k]] = x[pid1[k].internal[idx]+N];
            tdx[idx] = dx[pid1[k].internal[idx]];
            tvol[idx] = vol[pid1[k].internal[idx]];
            //if (k==0) cout<<"x "<<tx[idx]<<" "<<tx[idx+ihpSize.t_size[k]]<<endl;
        }

        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[k][j]==1)
            {
                for (int i=0; i<ihpSize.h_size[k][j]; i++)
                {

                    tx[idx] = x[pid1[k].halo[j][i]];
                    tx[idx+ihpSize.t_size[k]] = x[pid1[k].halo[j][i]+N];
                    tdx[idx] = dx[pid1[k].halo[j][i]];
                    tvol[idx] = vol[pid1[k].halo[j][i]];
                    idx++;
                }
            }
        }
        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[k][j]==1)
            {
                for (int i=0; i<ihpSize.p_size[k][j]; i++)
                {
                    tx[idx] = x[pid1[k].padding[j][i]];
                    tx[idx+ihpSize.t_size[k]] = x[pid1[k].padding[j][i]+N];
                    tdx[idx] = dx[pid1[k].padding[j][i]];
                    tvol[idx] = vol[pid1[k].padding[j][i]];
                    idx++;
                }
            }
        }

        //传输CPU数据到GPU
        CHECK(cudaMemcpy(pd.x[k], tx, Dim*ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(pd.dx[k], tdx, ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(pd.vol[k], tvol, ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        delete []tx;
        delete []tdx;
        delete []tvol;
    }
}

void coord_transfer2(int N, int GPUS, int Dim, real *x, real *dx, real *vol, POINT_ID pid1[],
                    IHP_SIZE &ihpSize, int **exchange_flag, Base_PD &pd, node_data** data_set)
{
    for (int k=0; k<GPUS; k++)
    {
        real *tx = new real [ihpSize.t_size[k]*Dim];
        real *tdx = new real [ihpSize.t_size[k]];
        real *tvol = new real [ihpSize.t_size[k]];
        int idx = 0;

        for (idx=0; idx<ihpSize.i_size[k]; idx++)
        {
            data_set[k][idx].id = idx;
            data_set[k][idx].x = tx[idx] = x[pid1[k].internal[idx]];
            data_set[k][idx].y = tx[idx+ihpSize.t_size[k]] = x[pid1[k].internal[idx]+N];
            if (Dim==3) data_set[k][idx].z = tx[idx+ihpSize.t_size[k]*2] = x[pid1[k].internal[idx]+N*2];
            tdx[idx] = dx[pid1[k].internal[idx]];
            tvol[idx] = vol[pid1[k].internal[idx]];
        }

        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[k][j]==1)
            {
                for (int i=0; i<ihpSize.h_size[k][j]; i++)
                {
                    data_set[k][idx].id = idx;
                    data_set[k][idx].x = tx[idx] = x[pid1[k].halo[j][i]];
                    data_set[k][idx].y = tx[idx+ihpSize.t_size[k]] = x[pid1[k].halo[j][i]+N];
                    if (Dim==3) data_set[k][idx].z = tx[idx+ihpSize.t_size[k]*2] = x[pid1[k].halo[j][i]+N*2];
                    tdx[idx] = dx[pid1[k].halo[j][i]];
                    tvol[idx] = vol[pid1[k].halo[j][i]];
                    idx++;
                }
            }
        }

        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[k][j]==1)
            {
                for (int i=0; i<ihpSize.p_size[k][j]; i++)
                {
                    data_set[k][idx].id = idx;
                    data_set[k][idx].x = tx[idx] = x[pid1[k].padding[j][i]];
                    data_set[k][idx].y = tx[idx+ihpSize.t_size[k]] = x[pid1[k].padding[j][i]+N];
                    if (Dim==3) data_set[k][idx].z = tx[idx+ihpSize.t_size[k]*2] = x[pid1[k].padding[j][i]+N*2];
                    tdx[idx] = dx[pid1[k].padding[j][i]];
                    tvol[idx] = vol[pid1[k].padding[j][i]];
                    idx++;
                }
            }
        }

        //传输CPU数据到GPU
        CHECK(cudaMemcpy(pd.x[k], tx, Dim*ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(pd.dx[k], tdx, ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(pd.vol[k], tvol, ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        delete []tx;
        delete []tdx;
        delete []tvol;
    }
}
void coord_transfer3(int N, int GPUS, int Dim, real *x, real *dx, real *vol, POINT_ID pid1[],
                     IHP_SIZE &ihpSize, int **exchange_flag, State_Thermal_Diffusion_PD2 &pd, node_data** data_set)
{
    for (int k=0; k<GPUS; k++)
    {
        real *tx = new real [ihpSize.t_size[k]*Dim];
        real *tdx = new real [ihpSize.t_size[k]];
        real *tvol = new real [ihpSize.t_size[k]];
        int idx = 0;

        for (idx=0; idx<ihpSize.i_size[k]; idx++)
        {
            data_set[k][idx].id = idx;
            data_set[k][idx].x = tx[idx] = x[pid1[k].internal[idx]];
            data_set[k][idx].y = tx[idx+ihpSize.t_size[k]] = x[pid1[k].internal[idx]+N];
            if (Dim==3) data_set[k][idx].z = tx[idx+ihpSize.t_size[k]*2] = x[pid1[k].internal[idx]+N*2];
            tdx[idx] = dx[pid1[k].internal[idx]];
            tvol[idx] = vol[pid1[k].internal[idx]];
        }

        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[k][j]==1)
            {
                for (int i=0; i<ihpSize.h_size[k][j]; i++)
                {
                    data_set[k][idx].id = idx;
                    data_set[k][idx].x = tx[idx] = x[pid1[k].halo[j][i]];
                    data_set[k][idx].y = tx[idx+ihpSize.t_size[k]] = x[pid1[k].halo[j][i]+N];
                    if (Dim==3) data_set[k][idx].z = tx[idx+ihpSize.t_size[k]*2] = x[pid1[k].halo[j][i]+N*2];
                    tdx[idx] = dx[pid1[k].halo[j][i]];
                    tvol[idx] = vol[pid1[k].halo[j][i]];
                    idx++;
                }
            }
        }

        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[k][j]==1)
            {
                for (int i=0; i<ihpSize.p_size[k][j]; i++)
                {
                    data_set[k][idx].id = idx;
                    data_set[k][idx].x = tx[idx] = x[pid1[k].padding[j][i]];
                    data_set[k][idx].y = tx[idx+ihpSize.t_size[k]] = x[pid1[k].padding[j][i]+N];
                    if (Dim==3) data_set[k][idx].z = tx[idx+ihpSize.t_size[k]*2] = x[pid1[k].padding[j][i]+N*2];
                    tdx[idx] = dx[pid1[k].padding[j][i]];
                    tvol[idx] = vol[pid1[k].padding[j][i]];
                    idx++;
                }
            }
        }

        //传输CPU数据到GPU
        CHECK(cudaMemcpy(pd.x[k], tx, Dim*ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(pd.dx[k], tdx, ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(pd.vol[k], tvol, ihpSize.t_size[k]* sizeof(real), cudaMemcpyHostToDevice));
        delete []tx;
        delete []tdx;
        delete []tvol;
    }
}



void data_segm(int N, int GPUS, real horizon, real *sx, real *x, real *dx, POINT_ID pid[], IHP_SIZE &ihpSize,
                  int **exchange_flag)
{
    real delta = 0.0;
    real dist = 0.0, dist2 = 0.0;
    real y = 0.0;
//#pragma omp parallel for private (delta, dist, dist2, y)
    for (int i=0; i<N; i++)   //遍历点
    {
        delta = (horizon) * dx[i];
        if (GPUS == 1) {
//#pragma omp atomic update
            ihpSize.i_size[0]++;
            pid[0].internal[ihpSize.i_size[0]-1] = i;
        } else {
            for (int j = 0; j < GPUS; j++) //遍历GPUS
            {
                if (j == 0) {
                    if (x[i] <= sx[j]) {
                        dist = fabs(x[i] - sx[j]);
                        if (dist <= delta) {
//#pragma omp atomic update
                            ihpSize.h_size[j][j + 1]++;
                            pid[j].halo[j + 1][ihpSize.h_size[j][j + 1]-1] = i;
                        } else {
//#pragma omp atomic update
                            ihpSize.i_size[j]++;
                            pid[j].internal[ihpSize.i_size[j]-1] = i;
                        }
                    }
                } else if (j == GPUS - 1) {
                    if (x[i] > sx[j-1]) {
                        dist = fabs(x[i] - sx[j-1]);
                        if (dist <= delta) {
//#pragma omp atomic update
                            ihpSize.h_size[j][j - 1]++;
                            pid[j].halo[j - 1][ihpSize.h_size[j][j - 1]-1] = i;

                        } else {
//#pragma omp atomic update
                            ihpSize.i_size[j]++;
                            pid[j].internal[ihpSize.i_size[j]-1] = i;
                        }
                    }
                } else {
                    if (x[i]>sx[j-1] and x[i]<=sx[j]) {
                        dist = fabs(x[i]-sx[j-1]);
                        dist2 = fabs(x[i]-sx[j]);
                        if (dist <= delta) {
//#pragma omp atomic update
                            ihpSize.h_size[j][j - 1]++;
                            pid[j].halo[j - 1][ihpSize.h_size[j][j - 1]-1] = i;

                        } else if (dist2 <= delta) {
//#pragma omp atomic update
                            ihpSize.h_size[j][j + 1]++;
                            pid[j].halo[j + 1][ihpSize.h_size[j][j + 1]-1] = i;

                        } else {
//#pragma omp atomic update
                            ihpSize.i_size[j]++;
                            pid[j].internal[ihpSize.i_size[j]-1] = i;

                        }
                    }
                }
            }
        }

    }

    for (int i = 0; i < GPUS; i++) {
        for (int j = 0; j < GPUS; j++) {
            if (exchange_flag[i][j] == 1) {
                ihpSize.p_size[j][i] = ihpSize.h_size[i][j];
                memcpy(pid[j].padding[i], pid[i].halo[j], ihpSize.h_size[i][j] * sizeof(int));
            }
        }
    }
//
    for (int i=0; i<GPUS; i++)
    {
        ihpSize.ih_size[i] = ihpSize.i_size[i];
        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[i][j]==1) {
                ihpSize.ih_size[i] += ihpSize.h_size[i][j];
                //cout<<"hsize "<<ihpSize.h_size[i][j]<<endl;
                ihpSize.h_begin[i][j] = ihpSize.i_size[i];
                for (int k = 0; k < j; k++) {
                    if (exchange_flag[i][k] == 1) {
                        ihpSize.h_begin[i][j] += ihpSize.h_size[i][k];
                    }
                }
            }
        }
    }
//
    for (int i=0; i<GPUS; i++)
    {
        ihpSize.t_size[i] = ihpSize.ih_size[i];
        //cout<<"tsize "<<ihpSize.t_size[i]<<endl;
        for (int j=0; j<GPUS; j++)
        {
            if (exchange_flag[i][j]==1) {
                ihpSize.t_size[i] += ihpSize.p_size[i][j];
                ihpSize.p_begin[i][j] = ihpSize.ih_size[i];
                for (int k = 0; k < j; k++) {
                    if (exchange_flag[i][k] == 1) {
                        ihpSize.p_begin[i][j] += ihpSize.p_size[i][k];
                    }
                }
            }
        }
    }
}
