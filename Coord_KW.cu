//
// Created by wxm on 2023/8/5.
//

#include "Coord_KW.cuh"

int coord_KW(int N, int nx, int ny, int nz, real rad, real dx, real *x, real *hdx, real *hvol)
{
    real tx;
    real ty;
    real tz;
    real vol = cubic(dx);
    unsigned int num= 0;
    for (int k=0; k<nz;k++)
    {
        for(int j=0; j<ny;j++)
        {
            for(int i=0; i<nx; i++)
            {
                tx = (-(nx-1)/2.0 + i) * dx;
                ty = (-(ny-1)/2.0+ j) * dx;
                tz = (-nz/2.0+0.5 + k) * dx;

                //if(i==1)  cout<<"x "<<tx<<" y "<<ty<<" z "<<tz<<endl;
                //if(i==0 && j==0)  cout<<"x "<<dx<<" y "<<nx<<endl;
                if ((tx>(-rad-0.0015/2.0) && tx<(-rad+0.0015/2.0)))
                {
                    if(ty<-1.0e-10)
                    {
                        x[num] = tx;
                        x[num+N] = ty;
                        x[num+2*N] = tz;
                        hdx[num] = dx;
                        hvol[num] = vol;
                        num++;
                    }
                }
                else if (tx>(rad-0.0015/2.0) && tx<(rad+0.0015/2.0))
                {
                    if(ty<-1.0e-10)
                    {
                        x[num] = tx;
                        x[num+N] = ty;
                        x[num+2*N] = tz;
                        hdx[num] = dx;
                        hvol[num] = vol;
                        num++;
                    }
                }
                else
                {
                    x[num] = tx;
                    x[num+N] = ty;
                    x[num+2*N] = tz;
                    hdx[num] = dx;
                    hvol[num] = vol;
                    num++;
                }
            }
        }
    }
    return num;
}
