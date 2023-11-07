//
// Created by wxm on 2023/6/19.
//

#include "Neighbor.cuh"

__global__ void  kernel_find_neighbor_2D
        (int N, int MN, real horizon, real *x, real *dx, int *NN, int *NL)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        for (int j=i+1; j<N; ++j)
        {
            if ( square(x[j]-x[i])+square(x[N+j]-x[N+i]) < square(horizon*dx[i]))
            {
                NL[i * MN + atomicAdd(&NN[i],1)] = j;
                NL[j * MN + atomicAdd(&NN[j],1)] = i;
            }
        }
    }
}

__global__ void  kernel_find_neighbor_2D_new
        (int N, int NT, int MN, real horizon, int begin, real *x, real *dx, int *NN, int *NL)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        i += begin;
        for (int j=0; j<NT; ++j)
        {
            if (j!=i) {
                if (square(x[j] - x[i]) + square(x[NT + j] - x[NT + i]) < square(horizon * dx[i])) {
                    NL[i * MN + atomicAdd(&NN[i], 1)] = j;
                }
            }
        }
    }
}

__global__ void  kernel_find_neighbor_3D_new
        (int N, int NT, int MN, real horizon, int begin, real *x, real *dx, int *NN, int *NL)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N)
    {
        i += begin;
        for (int j=0; j<NT; ++j)
        {
            if (j!=i) {
                if (square(x[j] - x[i]) + square(x[NT + j] - x[NT + i]) + square(x[NT*2 + j] - x[NT*2 + i]) < square(horizon * dx[i])) {
                    NL[i * MN + atomicAdd(&NN[i], 1)] = j;
                }
            }
        }
    }
}

int cmp_x(const void *a, const void *b)
{
    return ((node_data*)a)->x>((node_data*)b)->x ? 1:-1;
}

int cmp_y(const void *a, const void *b)
{
    return ((node_data*)a)->y>((node_data*)b)->y ? 1:-1;
}

int cmp_z(const void *a, const void *b)
{
    return ((node_data*)a)->z>((node_data*)b)->z ? 1:-1;
}

int cal_split(node_data *data_set, int size, int Dim)
{

    real mean_x=0.0, mean_y = 0.0, mean_z = 0.0;
    real var_x=0.0, var_y = 0.0, var_z = 0.0;
    for (int i=0; i<size; i++)
    {
        mean_x += data_set[i].x;
        mean_y += data_set[i].y;
        if (Dim==3) mean_z += data_set[i].z;
    }
    mean_x /= size;
    mean_y /= size;
    mean_z /= size;
    for (int i=0; i<size; i++)
    {
        var_x += square(data_set[i].x - mean_x);
        var_y += square(data_set[i].y - mean_y);
        if (Dim==3) var_z += square(data_set[i].z - mean_z);
    }
//    cout<<"mx "<<mean_x<<"my "<<mean_y<<" mz "<<mean_z<<endl;
//    cout<<"vx "<<var_x<<" vy "<<var_y<<" vz "<<var_z<<endl;
    if (Dim==2)
    {
        return var_x>=var_y ? 0:1;
    } else
    {

        if (var_x>=var_y and var_x>=var_z) return 0;
        if (var_y>=var_x and var_y>=var_z) return 1;
        if (var_z>=var_x and var_z>=var_y) return 2;
    }
}

int choose_split(node_data *data_set, int size, int split)
{

    if (split==0) qsort(data_set, size, sizeof(data_set[0]), cmp_x);
    else if (split==1) qsort(data_set, size, sizeof(data_set[0]), cmp_y);
    else if (split==2) qsort(data_set, size, sizeof(data_set[0]), cmp_z);
    return (size-1)/2;
}



int bulid_kdtree(node_data *data_set, int N, int size, int Dim, int &root, kd_node *kdtree)
{

    if(size==0)
    {
        return -1;
    }
    else if (size==1)
    {
        int id = data_set[0].id;
        kdtree[id].left = -1;
        kdtree[id].right = -1;
        return id;
    }
    else{
        int split = cal_split(data_set, size, Dim);
        int split_size = choose_split(data_set, size, split);
        int split_id = data_set[split_size].id;
        //cout<<"split_id "<<split_id<<" size "<<size<<" "<<split<<endl;
        if (split_size>N/3) {
            root = split_id;
            //return 0;
        }

        //cout<<"split_id "<<split_id<<" size "<<size<<endl;
        kdtree[split_id].split = split;
        //更新split

        kdtree[split_id].left =  bulid_kdtree(data_set, N, split_size, Dim, root, kdtree);
        kdtree[split_id].right =  bulid_kdtree(&data_set[split_size+1], N, size-split_size-1, Dim, root, kdtree);
        return split_id;
    }
}


__device__ __host__ void range_search(int i, int root, int MN, real delta, int NT, int size, int Dim, real *x, int *NN, int *NL, kd_node *kdtree)
{
    int s = 0;
    int size_begin = 0;
    const int sizek = size;
    int length = 1;
    int *id_list = new int [sizek];
    id_list[0] = root;
    int point;
    int length2 = 0;
    while (length>0)
    {
        point = id_list[size_begin];
        if (cal_dist(NT, i, point, x, Dim) < delta*delta)
        {
            if (i != point)  NL[i*MN+(NN[i]++)] = point;
        }
        size_begin++;
        length--;
        s = kdtree[point].split;

        if (s==0)
        {
            if (x[point]>=x[i]-delta and kdtree[point].left != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin] = kdtree[point].left;
                } else
                {
                    id_list[length] = kdtree[point].left;
                }
                length++;
            }
            if (x[point]<=x[i]+delta and kdtree[point].right != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin] = kdtree[point].right;

                } else
                {
                    id_list[length] = kdtree[point].right;
                }
                length++;
            }
        }
        else if (s==1)
        {
            if (x[point+NT]>=x[i+NT]-delta and kdtree[point].left != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin] = kdtree[point].left;
                } else
                {
                    id_list[length] = kdtree[point].left;
                }
                length++;
            }
            if (x[point+NT]<=x[i+NT]+delta and kdtree[point].right != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin] = kdtree[point].right;

                } else
                {
                    id_list[length] = kdtree[point].right;
                }
                length++;
            }

        }
        else if (s==2)
        {
            if (x[point+2*NT]>=x[i+2*NT]-delta and kdtree[point].left != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin] = kdtree[point].left;
                } else
                {
                    id_list[length] = kdtree[point].left;
                }
                length++;
            }
            if (x[point+2*NT]<=x[i+2*NT]+delta and kdtree[point].right != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin] = kdtree[point].right;

                } else
                {
                    id_list[length] = kdtree[point].right;
                }
                length++;
            }

        }
        if (length>length2) length2=length;
        if (length>=sizek-1) {
            printf("length\n");
            break;
        }
    }
    if (i==root) {
        printf("length %d\n", length2);
    }
    delete []id_list;
}

__device__ __host__ void range_search2(int i, int root, int MN, real delta, int NT, int size, int Dim, int begin, real *x,
                                       int *NN, int *NL, int *id_list, kd_node *kdtree)
{
    int s = 0;
    int size_begin = 0;
    const int sizek = size;
    int length = 1;
    //int step_begin = (i)*size;
    int step_begin = (i-begin)*size;
    id_list[step_begin] = root;
    int point;
    int length2 = 0;
    while (length>0)
    {
        point = id_list[step_begin+size_begin];
        if (cal_dist(NT, i, point, x, Dim) < delta*delta)
        {
            if (i != point) {
                long int idx = (long int)i*MN+(NN[i]++);
                NL[idx] = point;
                //if (i==root) printf("nn %d %d %d\n", NN[i], NL[idx], idx);
            }
        }
        size_begin++;
        length--;
        s = kdtree[point].split;
        if (s==0)
        {
            if (x[point]>=x[i]-delta and kdtree[point].left != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin+step_begin] = kdtree[point].left;
                } else
                {
                    id_list[length+step_begin] = kdtree[point].left;
                }
                length++;
            }
            if (x[point]<=x[i]+delta and kdtree[point].right != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin+step_begin] = kdtree[point].right;

                } else
                {
                    id_list[length+step_begin] = kdtree[point].right;
                }
                length++;
            }
        }
        else if (s==1)
        {
            if (x[point+NT]>=x[i+NT]-delta and kdtree[point].left != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin+step_begin] = kdtree[point].left;
                } else
                {
                    id_list[length+step_begin] = kdtree[point].left;
                }
                length++;
            }
            if (x[point+NT]<=x[i+NT]+delta and kdtree[point].right != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin+step_begin] = kdtree[point].right;

                } else
                {
                    id_list[length+step_begin] = kdtree[point].right;
                }
                length++;
            }

        }
        else if (s==2)
        {
            if (x[point+2*NT]>=x[i+2*NT]-delta and kdtree[point].left != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin+step_begin] = kdtree[point].left;
                } else
                {
                    id_list[length+step_begin] = kdtree[point].left;
                }
                length++;
            }
            if (x[point+2*NT]<=x[i+2*NT]+delta and kdtree[point].right != -1)
            {
                if (size_begin>0)
                {
                    id_list[--size_begin+step_begin] = kdtree[point].right;

                } else
                {
                    id_list[length+step_begin] = kdtree[point].right;
                }
                length++;
            }

        }
        if (length>length2) length2=length;
        if (length>=sizek-1) {
            printf("length\n");
            break;
        }
    }
    if (i==root) {
        printf("length %d\n", length2);
    }
}

void find_neighbor_cpu(int N, int MN, int *NN, int *NL, real *x, real delta)
{
    for (int i=0; i<N; i++)
    {
        NN[i] = 0;
        for (int j=0; j<N; j++)
        {
            if (square(x[i]-x[j])+square(x[i+N]-x[j+N])<=delta*delta)
            {
                if (j!=i) NL[i*MN+NN[i]++] = j;
            }
        }
    }
}

__global__ void  kernel_find_neighbor_kd
        (int N, int NT, int MN, real horizon, int begin, int root, int size, int Dim, real *x, real *dx, int *NN, int *NL, kd_node* kdtree)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i<N)
    {
        i += begin;
        if (i<NT) {
            range_search(i, root, MN, horizon * dx[i], NT, size, Dim, x, NN, NL, kdtree);
        }
    }
}

__global__ void  kernel_find_neighbor_kd2
        (int N, int NT, int MN, real horizon, int begin, int root, int size, int Dim, int step, int *id_list, real *x,
         real *dx, int *NN, int *NL, kd_node* kdtree)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i<N) {
        i += begin;
        if (i < NT) {
            range_search2(i, root, MN, horizon * dx[i], NT, size, Dim, begin, x, NN, NL, id_list, kdtree);
        }
    }
}

void find_neighbor_kd(int GPUS, int N, int N2, int MN, int Dim,  IHP_SIZE &ihpSize, POINT_ID pid[], int **exchange_flag, Grid &grid,
                      real *hx, real *hdx, real *hvol, Base_PD &pd, cudaStream_t *st_body)
{
    node_data **data_set = new node_data * [GPUS];
    kd_node **kdtree = new kd_node * [GPUS];
    kd_node **kdtree_gpu = new kd_node * [GPUS];
    real **x = new real *[GPUS];
    int root[GPUS];
    for (int i = 0; i < GPUS; i++) {
        data_set[i] = new node_data [ihpSize.t_size[i]];
        kdtree[i] = new kd_node [ihpSize.t_size[i]];
        x[i] = new real [ihpSize.t_size[i]*Dim];
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void**)&kdtree_gpu[i], ihpSize.t_size[i]*sizeof(kd_node)));
    }

    coord_transfer2(N2, GPUS, Dim, hx, hdx, hvol, pid,ihpSize, exchange_flag,pd, data_set);
    long double strat = cpuSecond();
//#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        CHECK(cudaSetDevice(i));

        CHECK(cudaMemcpy(x[i], pd.x[i], Dim*ihpSize.t_size[i]* sizeof(real), cudaMemcpyDeviceToHost));
        bulid_kdtree(data_set[i], ihpSize.t_size[i], ihpSize.t_size[i], Dim, root[i], kdtree[i]);
        delete []data_set[i];
        //cout<<"root "<<x[i][kdtree[i][root[i]].left]<<endl;

        CHECK(cudaMemcpy(kdtree_gpu[i], kdtree[i], ihpSize.t_size[i]* sizeof(kd_node), cudaMemcpyHostToDevice));
        kernel_find_neighbor_kd<<<grid.p_t[i],block_size, 0, st_body[i]>>>
                (ihpSize.t_size[i], ihpSize.t_size[i], MN, horizon, 0, root[i], 55, Dim, pd.x[i],
                 pd.dx[i], pd.NN[i], pd.NL[i], kdtree_gpu[i]);
//        kernel_find_neighbor_3D_new<<<grid_point_t[i],block_size, 0, st_body[i]>>>
//        (ihpSize.t_size[i], ihpSize.t_size[i], MN, horizon, 0,pd.x[i], pd.dx[i], pd.NN[i], pd.NL[i]);
        CHECK(cudaFree(kdtree_gpu[i]));
    }
    device_sync(GPUS);
    cout<<"cost time: "<<cpuSecond()-strat<<endl;
    delete []data_set;
    delete []kdtree;
    delete []kdtree_gpu;
}

void find_neighbor_kd2(int GPUS, int N, int MN, int Dim,  IHP_SIZE &ihpSize, POINT_ID pid[], int **exchange_flag, Grid &grid,
                       real *hx, real *hdx, real *hvol, Base_PD &pd, cudaStream_t *st_body)
{
    node_data **data_set = new node_data * [GPUS];
    kd_node **kdtree = new kd_node * [GPUS];
    kd_node **kdtree_gpu = new kd_node * [GPUS];
    real **x = new real *[GPUS];
    int root[GPUS];
    int **id_list = new int *[GPUS];
    int size[GPUS];
    int step[GPUS];
    int upper[GPUS];
    int grid_size[GPUS];
    for (int i = 0; i < GPUS; i++) {
        data_set[i] = new node_data [ihpSize.t_size[i]];
        kdtree[i] = new kd_node [ihpSize.t_size[i]];
        x[i] = new real [ihpSize.t_size[i]*Dim];
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void**)&kdtree_gpu[i], ihpSize.t_size[i]*sizeof(kd_node)));
        size[i] = sqrt(ihpSize.t_size[i])*2;
        if (size[i]<200) size[i] = 200;
        step[i] = (1024*1024*1024)/ sizeof(int)/size[i];
        cout<<step[i]<<" "<< size[i]<<endl;
        if (step[i]>ihpSize.t_size[i]) step[i] = ihpSize.t_size[i];
        upper[i] = (ihpSize.t_size[i] - 1)/step[i] + 1;
        CHECK(cudaMalloc((void**)&id_list[i], step[i]*size[i]*sizeof(int)));
        grid_size[i] = (step[i]-1)/block_size + 1;
    }
    coord_transfer2(N, GPUS, Dim, hx, hdx, hvol, pid,ihpSize, exchange_flag,pd, data_set);
    long double strat = cpuSecond();
#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        //if (i==1){
        CHECK(cudaSetDevice(i));

        CHECK(cudaMemcpy(x[i], pd.x[i], Dim * ihpSize.t_size[i] * sizeof(real), cudaMemcpyDeviceToHost));
        bulid_kdtree(data_set[i], ihpSize.t_size[i], ihpSize.t_size[i], Dim, root[i], kdtree[i]);
        delete[]data_set[i];
        cout << "root " << root[i] << endl;

        CHECK(cudaMemcpy(kdtree_gpu[i], kdtree[i], ihpSize.t_size[i] * sizeof(kd_node), cudaMemcpyHostToDevice));
    }
    long double neighbor_time = cpuSecond();
    #pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        //if (i==1)
        CHECK(cudaSetDevice(i));
        for (int j = 0; j < upper[i]; j++) {

            kernel_find_neighbor_kd2<<<grid_size[i], block_size, 0, st_body[i]>>>
                    (step[i], ihpSize.t_size[i], MN, horizon, j * step[i], root[i],
                     size[i], Dim, step[i], id_list[i], pd.x[i], pd.dx[i], pd.NN[i], pd.NL[i], kdtree_gpu[i]);
            //cout << "upper " << upper[i] << " hh " << i << endl;
        }
    }
    device_sync(GPUS);

    cout<<"neighbor time: "<<(cpuSecond()-neighbor_time)*1000<<endl;
    for (int i = 0; i < GPUS; i++) {
        //if (i==1)
        CHECK(cudaSetDevice(i));
        kernel_initial_fail<<<grid.b_t[i], block_size, 0, st_body[i]>>>(ihpSize.t_size[i], MN, pd.fail[i]);
        CHECK(cudaFree(kdtree_gpu[i]));
        CHECK(cudaFree(id_list[i]));
    }

    device_sync(GPUS);

    cout<<"cost time: "<<cpuSecond()-strat<<endl;

    delete []data_set;
    delete []kdtree;
    delete []kdtree_gpu;
    delete []id_list;
}
__global__ void kernel_initial_fail2(int N, int MN, bool *fail)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx<N*MN) fail[idx] = 1;
}
void find_neighbor_kd3(int GPUS, int N, int MN, int Dim,  IHP_SIZE &ihpSize, POINT_ID pid[], int **exchange_flag, Grid &grid,
                       real *hx, real *hdx, real *hvol, State_Thermal_Diffusion_PD2 &pd, cudaStream_t *st_body)
{
    node_data **data_set = new node_data * [GPUS];
    kd_node **kdtree = new kd_node * [GPUS];
    kd_node **kdtree_gpu = new kd_node * [GPUS];
    real **x = new real *[GPUS];
    int root[GPUS];
    int **id_list = new int *[GPUS];
    long int size[GPUS];
    long int step[GPUS];
    int upper[GPUS];
    int grid_size[GPUS];
    long int t_size[GPUS];

    for (int i = 0; i < GPUS; i++) {
        data_set[i] = new node_data [ihpSize.t_size[i]];
        kdtree[i] = new kd_node [ihpSize.t_size[i]];
        x[i] = new real [ihpSize.t_size[i]*Dim];
        t_size[i] = ihpSize.t_size[i];
        cout<<"Dim: "<<ihpSize.t_size[i]*Dim<<endl;
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc((void**)&kdtree_gpu[i], t_size[i]*sizeof(kd_node)));
        size[i] = sqrt(t_size[i]);
        if (size[i]<200) size[i] = 200;
        step[i] = (1024*1024*1024)/ sizeof(int)/size[i];
        cout<<step[i]<<" "<< size[i]<<endl;
        if (step[i]>t_size[i]) step[i] = t_size[i];
        upper[i] = (t_size[i] - 1)/step[i] + 1;
        CHECK(cudaMalloc((void**)&id_list[i], step[i]*size[i]*sizeof(int)));
        grid_size[i] = (step[i]-1)/block_size + 1;
    }
    coord_transfer3(N, GPUS, Dim, hx, hdx, hvol, pid,ihpSize, exchange_flag, pd, data_set);
    long double strat = cpuSecond();
#pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        //if (i==1){
        CHECK(cudaSetDevice(i));

        CHECK(cudaMemcpy(x[i], pd.x[i], Dim * t_size[i] * sizeof(real), cudaMemcpyDeviceToHost));
        bulid_kdtree(data_set[i], t_size[i], t_size[i], Dim, root[i], kdtree[i]);
        delete[]data_set[i];
        cout << "root " << root[i] << endl;

        CHECK(cudaMemcpy(kdtree_gpu[i], kdtree[i], t_size[i] * sizeof(kd_node), cudaMemcpyHostToDevice));
    }
    cout<<step[0]<<" "<<grid_size[0]<<endl;
    long double neighbor_time = cpuSecond();
    #pragma omp parallel for
    for (int i = 0; i < GPUS; i++) {
        //if (i==1)
        CHECK(cudaSetDevice(i));
        for (int j = 0; j < upper[i]; j++) {
            //cout<<"begin "<<j * step[i]<<endl;
            kernel_find_neighbor_kd2<<<grid_size[i], block_size, 0, st_body[i]>>>
                    (step[i], t_size[i], MN, horizon, j * step[i], root[i],
                     size[i], Dim, step[i], id_list[i], pd.x[i], pd.dx[i], pd.NN[i], pd.NL[i], kdtree_gpu[i]);
            //cout << "upper " << upper[i] << " hh " << i << endl;
        }
    }
    device_sync(GPUS);

    cout<<"neighbor time: "<<(cpuSecond()-neighbor_time)*1000<<endl;
//    for (int i = 0; i < GPUS; i++) {
//        //if (i==1)
//        CHECK(cudaSetDevice(i));
//        kernel_initial_fail2<<<grid.b_t[i], block_size, 0, st_body[i]>>>(ihpSize.t_size[i], MN, pd.fail[i]);
//        CHECK(cudaFree(kdtree_gpu[i]));
//        CHECK(cudaFree(id_list[i]));
//    }

    //device_sync(GPUS);

    //cout<<"cost time: "<<cpuSecond()-strat<<endl;

    delete []data_set;
    delete []kdtree;
    delete []kdtree_gpu;
    delete []id_list;
}

void find_neighbor_kd_cpu(int N, int MN, int Dim, Bond_PD_CPU &pd, int omp)
{
    node_data* data_set = new node_data [N];
    kd_node *kdtree = new kd_node [N];
    int root = 0;
    int size = sqrt(N);
    if (size<200) size = 200;
    int *id_list = new int [size*N];

    if (omp==0)
    {
        for (int i=0; i<N; i++)
        {
            data_set[i].id = i;
            data_set[i].x = pd.x[i];
            data_set[i].y = pd.x[i+N];
            data_set[i].z = pd.x[i+N*2];
        }
        bulid_kdtree(data_set, N, N, Dim, root, kdtree);
        cout<<root<<endl;
        delete []data_set;
        long double start = cpuSecond();
        for (int i=0; i<N; i++)
        {
            range_search2(i, root, MN, horizon*pd.dx[i], N, size, Dim, 0, pd.x,
                          pd.NN, pd.NL, id_list, kdtree);
        }
        for (int i=0; i<N*MN; i++) pd.fail[i] = 1;
        cout<<"neighbor time: "<<(cpuSecond()-start)*1000<<endl;
    } else if (omp==1)
    {

//#pragma omp parallel for
        for (int i=0; i<N; i++)
        {
            data_set[i].id = i;
            data_set[i].x = pd.x[i];
            data_set[i].y = pd.x[i+N];
            data_set[i].z = pd.x[i+N*2];
        }
        bulid_kdtree(data_set, N, N, Dim, root, kdtree);
        cout<<root<<endl;
        delete []data_set;
        long double start = cpuSecond();
#pragma omp parallel for
        for (int i=0; i<N; i++)
        {
            range_search2(i, root, MN, horizon*pd.dx[i], N, size, Dim, 0, pd.x,
                          pd.NN, pd.NL, id_list, kdtree);
        }
        cout<<"neighbor time: "<<(cpuSecond()-start)*1000<<endl;
#pragma omp parallel for
        for (int i=0; i<N*MN; i++)
            pd.fail[i] = 1;
    }
}

void find_neighbor_kd_cpu(int N, int MN, int Dim, State_PD_CPU &pd, int omp)
{
    node_data* data_set = new node_data [N];
    kd_node *kdtree = new kd_node [N];
    int root = 0;
    int size = sqrt(N);
    if (size<200) size = 200;
    int *id_list = new int [size*N];

    if (omp==0)
    {
        for (int i=0; i<N; i++)
        {
            data_set[i].id = i;
            data_set[i].x = pd.x[i];
            data_set[i].y = pd.x[i+N];
            data_set[i].z = pd.x[i+N*2];
        }
        bulid_kdtree(data_set, N, N, Dim, root, kdtree);
        cout<<root<<endl;
        delete []data_set;
        long double start = cpuSecond();
        for (int i=0; i<N; i++)
        {
            range_search2(i, root, MN, horizon*pd.dx[i], N, size, Dim, 0, pd.x,
                          pd.NN, pd.NL, id_list, kdtree);
        }
        for (int i=0; i<N*MN; i++) pd.fail[i] = 1;
        cout<<"neighbor time: "<<(cpuSecond()-start)*1000<<endl;
    } else if (omp==1)
    {

//#pragma omp parallel for
        for (int i=0; i<N; i++)
        {
            data_set[i].id = i;
            data_set[i].x = pd.x[i];
            data_set[i].y = pd.x[i+N];
            data_set[i].z = pd.x[i+N*2];
        }
        bulid_kdtree(data_set, N, N, Dim, root, kdtree);
        cout<<root<<endl;
        delete []data_set;
        long double start = cpuSecond();
#pragma omp parallel for
        for (int i=0; i<N; i++)
        {
            range_search2(i, root, MN, horizon*pd.dx[i], N, size, Dim, 0, pd.x,
                          pd.NN, pd.NL, id_list, kdtree);
        }
        cout<<"neighbor time: "<<(cpuSecond()-start)*1000<<endl;
#pragma omp parallel for
        for (int i=0; i<N*MN; i++)
            pd.fail[i] = 1;
    }
}