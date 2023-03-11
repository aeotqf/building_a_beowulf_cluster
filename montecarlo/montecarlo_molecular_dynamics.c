#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define L 16
#define N L*L*L
#define N_mc 100000

int main(int argc, char **argv)
{
    int i,j,n;
    int p,rank;
    int i_start,i_end;
    int E,E0,E1,E2,E3,E4,E5,E6,E7,E8,E9;
    double beta = 1.0;
    double deltaE;
    int *lattice;
    int x,y,z,xp,xm,yp,ym,zp,zm;
    int dE[17] = {6,-4,-4,-4,-4,-4,-4,-4,2,2,2,2,2,2,2,2,0};
    double acc_rate;
    int n_accept = 0;
    int n_total = 0;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&p);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    lattice = (int *)malloc(N*sizeof(int));

    srand(rank);

    for (i = 0; i < N; i++) lattice[i] = 1;

    for (n = 0; n < N_mc; n++) {
        i_start = rank*N/p;
        i_end = (rank+1)*N/p;

        for (i = i_start; i < i_end; i++) {
            x = i%L;
            y = (i/L)%L;
            z = i/(L*L);
            E0 = 0;
            xp = (x+1)%L;
            xm = (x-1+L)%L;
            yp = (y+1)%L;
            ym = (y-1+L)%L;
            zp = (z+1)%L;
            zm = (z-1+L)%L;
            E1 = lattice[z*L*L+y*L+xp]+lattice[z*L*L+y*L+xm]+
                 lattice[z*L*L+yp*L+x]+lattice[z*L*L+ym*L+x]+
                 lattice[zp*L*L+y*L+x]+lattice[zm*L*L+y*L+x];
            E2 = -E1;
            E3 = E1+E0;
            E4 = E2+E0;
            E5 = E0;
            E6 = 3+E0;
            E7 = -3+E0;
            E8 = 4+E0;
            E9 = -4+E0;
            deltaE = dE[E1+E0]+dE[E2+E0]+dE[E3]+dE[E4]+dE[E5]+
                     dE[E6]+dE[E7]+dE[E8]+dE[E9];
            acc_rate = exp(-beta*deltaE);
            if (deltaE < 0 || (double)rand()/RAND_MAX < acc_rate) {
                lattice[i] *= -1;
                n_accept++;
            }
            n_total++;
        }
    }

    MPI_Reduce(&n_accept,&i,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
    MPI_Reduce(&n_total,&j,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

    if (rank == 0) {
        printf("%f\n",(double)i/j);
    }

    MPI_Finalize();
    free(lattice);
    return 0;
}

/*J.C. Eilbeck and J.A.C. Gallas, "Semi-implicit Fourier-spectral solution of the complex Ginzburg-Landau equation," 
Computer Physics Communications, vol. 77, no. 3, pp. 289-301, 1993.*/