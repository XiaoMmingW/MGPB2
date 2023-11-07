//
// Created by wxm on 2023/7/26.
//

#include "GPUS_Function.cuh"
void enableP2P (int ngpus)
{
    for (int i=0; i<ngpus; i++)
    {
        CHECK(cudaSetDevice(i));

        for (int j = 0; j < ngpus; j++)
        {
            if (i == j) continue;

            int peer_access_available = 0;
            CHECK(cudaDeviceCanAccessPeer(&peer_access_available, i, j));

            if (peer_access_available) CHECK(cudaDeviceEnablePeerAccess(j, 0));
            else
            {
                cout<<"can't acess "<<endl;
            }
        }
    }
}

bool isCapableP2P(int ngpus)
{
    cudaDeviceProp prop[ngpus];
    int iCount = 0;

    for (int i = 0; i < ngpus; i++)
    {
        CHECK(cudaGetDeviceProperties(&prop[i], i));

        if (prop[i].major >= 2) iCount++;

        printf("> GPU%d: %s %s capable of Peer-to-Peer access\n", i,
               prop[i].name, (prop[i].major >= 2 ? "is" : "not"));
        fflush(stdout);
    }

    if(iCount != ngpus)
    {
        printf("> no enough device to run this application\n");
        fflush(stdout);
    }

    return (iCount == ngpus);
}


//void transfer_data(const int GPU_ID, const int GPUS, int **exchange_flag, int **halo_size, IHP_SIZE &ihpSize, real **data, cudaStream_t **st_halo)
//{
//
//    for(int j=0; j<GPUS; j++)
//    {
//        if (exchange_flag[GPU_ID][j]==1)
//        {
//            CHECK(cudaMemcpyAsync(dataGpuCell[j].data_padding[GPU_ID], dataGpuCell[j].data_halo[GPU_ID],
//                                  halo_size[][j]*sizeof(real), cudaMemcpyDeviceToDevice, st_halo[j]));
//        }
//    }
//
//}

void device_sync(int GPUS)
{
    for (int i=0; i<GPUS; i++)
    {
        CHECK(cudaSetDevice(i));
        CHECK(cudaDeviceSynchronize());
    }
}


