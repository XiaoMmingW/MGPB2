//
// Created by wxm on 2023/6/19.
//

#include "ex1.cuh"


void ex1()
{
    /*
    real length = 0.05;
    real height = 0.05;
    real dx = 0.1e-3;
    int nx = (length-dx/2.0)/dx+1;
    int ny = (height-dx/2.0)/dx+1+6;
    int MN = 32;
    int N = nx*ny;
    int Dim = 2;
    int gpus = 1;
    int nt = 1250;
    real dt = 1.3367e-8;
    real sc = 0.04472;
    real E = 192.0e9;
    real pratio = 0.3;
    real K = E/3.0/(1.0-2.0*pratio);
    real G = 0.5*E/(1.0+pratio);
    real dens = 8000.0;
    real vel_load = 20.0;
    size_t iexchange = 3*nx* sizeof(real);
    int exchange_size = 3*nx;
    int gridsize_coord = (N-1)/block_size + 1;
    isCapableP2P(gpus);
    enableP2P (gpus);
    real *x, *voll, *dxx;
    CHECK(cudaMalloc((void**)&x, Dim * sizeof(real) * N));
    CHECK(cudaMalloc((void**)&voll, sizeof(real) * N));
    CHECK(cudaMalloc((void**)&dxx, sizeof(real) * N));
    real *hx = new real [N*Dim];
    real *hvol = new real [N];
    real *hdx = new real [N];
    //kernel_coord_plate_crack<<<gridsize_coord, block_size>>>(x, dxx, voll, N, nx, ny, dx, cubic(dx));
    coord_plate(hx, hdx, hvol, N, nx, ny, dx, cubic(dx));
    int *point_num = new int [gpus];
    point_num[0] = N;//(N/2+3*nx);
    point_num[1] = (N/2+3*nx);
    Base_Atom a[gpus];
    Base_Bond b[gpus];

    for (int i=0;i<gpus;i++)
    {
        a[i].init(point_num[i],Dim, i);
        b[i].init(point_num[i], MN, i);
    }

    CHECK(cudaMemcpy(a[0].x, hx, point_num[0] * sizeof(real), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(&a[0].x[point_num[0]], &hx[N], point_num[0]* sizeof(real), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a[0].vol, hvol, point_num[0]* sizeof(real), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(a[0].dx, hdx, point_num[0]* sizeof(real), cudaMemcpyHostToDevice));
    if (gpus>1) {
        CHECK(cudaMemcpy(a[1].x, &hx[point_num[0] - nx * 6], point_num[1] * sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(&a[1].x[point_num[1]], &hx[N + point_num[0] - nx * 6], point_num[1] * sizeof(real),
                         cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(a[1].vol, &hvol[point_num[0] - nx * 6], point_num[1] * sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(a[1].dx, &hdx[point_num[0] - nx * 6], point_num[1] * sizeof(real), cudaMemcpyHostToDevice));
    }
    //流分配
    cudaStream_t st_halo[gpus], st_body[gpus];
    for (int i=0;i<gpus;i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaStreamCreate(&st_halo[i]));
        CHECK(cudaStreamCreate( &st_body[i]));
    }
    long double strat = cpuSecond();
    for (int i=0;i<gpus;i++)
    {
        CHECK(cudaSetDevice(i));
        int grid_size1 = (point_num[i]-1)/block_size+1;
        int grid_size2 = (point_num[i]*MN-1)/block_size+1;
        initial_fail<<<grid_size2,block_size,0, st_body[i]>>>(point_num[i], MN, b[i].fail);
        kernel_find_neighbor_2D<<<grid_size1,block_size,0, st_body[i]>>>
        (point_num[i], MN, horizon, a[i].x, a[i].dx, a[i].NN, b[i].NL);
        kernel_vol_Corr<<<grid_size2,block_size,0, st_body[i]>>>
        (point_num[i], MN, horizon, a[i].NN, b[i].NL, a[i].x, a[i].dx, b[i].idist, b[i].fac);
        kernel_initial_weight_2D<<<grid_size2,block_size,0, st_body[i]>>>(
                point_num[i], MN, horizon, a[i].NN, b[i].NL, a[i].m, a[i].dx, a[i].vol, b[i].w, b[i].idist, b[i].fac);
        gpu_set_crack<<<grid_size2,block_size,0, st_body[i]>>>(
                point_num[i], MN, pi, 0.01, 0.0, 0.0, 0.0, point_num[i], a[i].NN, b[i].fail,
                b[i].NL, a[i].x);
    }
for (int i=0;i<gpus;i++) {
    CHECK(cudaSetDevice(i));
    CHECK(cudaDeviceSynchronize());
    //CHECK(cudaStreamSynchronize(st_body[0]));
    // CHECK(cudaStreamSynchronize(st_body[1]));
}
    if (gpus>1) {
//        CHECK(cudaMemcpy(&a[0].m[point_num[0] - exchange_size], &a[1].m[exchange_size], iexchange, cudaMemcpyDeviceToDevice));
//        CHECK(cudaMemcpy(&a[1].m[0], &a[0].m[point_num[0]-2*exchange_size],iexchange, cudaMemcpyDeviceToDevice));
        CHECK(cudaMemcpyAsync(&a[0].m[point_num[0] - exchange_size], &a[1].m[exchange_size],
                                  iexchange, cudaMemcpyDeviceToDevice,  st_halo[0]));
//        CHECK(cudaMemcpyPeerAsync(&a[0].m[point_num[0] - exchange_size], 0, &a[1].m[exchange_size], 1,
//                              iexchange,  st_halo[0]));
        CHECK(cudaMemcpyAsync(&a[1].m[0], &a[0].m[point_num[0]-2*exchange_size],
                              iexchange, cudaMemcpyDeviceToDevice, st_halo[0]));

    }
    CHECK(cudaDeviceSynchronize());
    for (int j=0; j<1250; j++) {
        for (int i=0;i<gpus;i++) {
            CHECK(cudaSetDevice(i));
            int grid_sizep = (point_num[i]-1)/block_size+1;
            int grid_sizeb = (point_num[i]*MN-1)/block_size+1;
            //load_vel<<<grid_sizep,block_size,0, st_body[i]>>>(a[i].disp, a[i].x, (j+1)*dt, point_num[i], vel_load, height/2.0);
            kernel_cal_theta_2D<<<grid_sizeb,block_size,0, st_body[i]>>>(
                    point_num[i], MN, a[i].NN, a[i].m, a[i].vol, a[i].theta, b[i].NL,b[i].w, b[i].fail,
                    b[i].nlength, b[i].idist, b[i].fac, a[i].x, a[i].disp);
//cout<<"hh"<<endl;
        }
        CHECK(cudaDeviceSynchronize());
        if (gpus>1) {
            CHECK(cudaMemcpy(&a[0].theta[point_num[0] - exchange_size], &a[1].theta[exchange_size], iexchange, cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(&a[1].theta[0], &a[0].theta[point_num[0]-2*exchange_size],iexchange, cudaMemcpyDeviceToDevice));
//            CHECK(cudaMemcpyAsync(&a[0].theta[point_num[0] - exchange_size], &a[1].theta[exchange_size],
//                                  iexchange, cudaMemcpyDefault, st_body[0]));
//            CHECK(cudaMemcpyAsync(&a[1].theta[0], &a[0].theta[point_num[0]-2*exchange_size],
//                                  iexchange, cudaMemcpyDefault, st_body[1]));
        }
        CHECK(cudaDeviceSynchronize());
        for (int i=0;i<gpus;i++) {
            CHECK(cudaSetDevice(i));
            int grid_sizep = (point_num[i]-1)/block_size+1;
            int grid_sizeb = (point_num[i]*MN-1)/block_size+1;
            CHECK(cudaMemset(a[i].pforce,0, Dim*point_num[i]* sizeof(real)));
            kernel_state_force_2D<<<grid_sizeb,block_size,0, st_body[i]>>>(
                    point_num[i], K, G, MN, horizon, dx, pi, sc, a[i].NN, a[i].m, a[i].theta, a[i].vol, b[i].fail, b[i].NL,
                    b[i].w, b[i].idist, b[i].nlength, b[i].fac, a[i].x, a[i].disp, a[i].pforce, a[i].dx
            );
            integrate_2D<<<grid_sizep,block_size,0, st_body[i]>>>(
                    point_num[i], dens, dt, a[i].vel, a[i].disp,  a[i].pforce, a[i].bforce, a[i].acc);
        }
        CHECK(cudaDeviceSynchronize());
        if (gpus>1) {
            CHECK(cudaMemcpy(&a[0].disp[point_num[0] - exchange_size], &a[1].disp[exchange_size], iexchange, cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(&a[1].disp[0], &a[0].disp[point_num[0]-2*exchange_size],iexchange, cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(&a[0].disp[point_num[0]*2 - exchange_size], &a[1].disp[point_num[1]+exchange_size], iexchange, cudaMemcpyDeviceToDevice));
            CHECK(cudaMemcpy(&a[1].disp[point_num[1]+0], &a[0].disp[point_num[0]+point_num[0]-2*exchange_size],iexchange, cudaMemcpyDeviceToDevice));
            //            CHECK(cudaMemcpyAsync(&a[0].disp[point_num[0] - exchange_size], &a[1].disp[exchange_size],
//                                  iexchange, cudaMemcpyDefault, st_body[0]));
//            CHECK(cudaMemcpyAsync(&a[1].disp[0], &a[0].disp[point_num[0]-2*exchange_size],
//                                  iexchange, cudaMemcpyDefault, st_body[1]));
        }
        CHECK(cudaDeviceSynchronize());
            for (int i=0;i<gpus;i++) {
                CHECK(cudaSetDevice(i));
                int grid_sizep = (point_num[i]-1)/block_size+1;
                int grid_sizeb = (point_num[i]*MN-1)/block_size+1;
                load_vel<<<grid_sizep,block_size,0, st_body[i]>>>(a[i].disp, a[i].x, (j+1)*dt, point_num[i], vel_load, height/2.0);
            }
        CHECK(cudaDeviceSynchronize());
    }
    cout<<"cost time: "<<cpuSecond()-strat<<endl;
    for (int i=0;i<gpus;i++) {
        CHECK(cudaSetDevice(i));
        int grid_sizep = (point_num[i] - 1) / block_size + 1;
        int grid_sizeb = (point_num[i] * MN - 1) / block_size + 1;
        kernel_cal_dmg_2D<<<grid_sizeb,block_size,0, st_body[i]>>>(point_num[i], MN, a[i].NN, a[i].dmg, a[i].vol, b[i].NL, b[i].fail, b[i].fac);
    }

    save_disp_gpu(point_num[0],  a[0], "disp0.txt");
    //save_disp_gpu(point_num[1],  a[1], "disp1.txt");

    for (int i=0;i<N;i++)
    {
                Base_Atom(const int N, const int Dim, const int gpu_id);
    }
     */

}
