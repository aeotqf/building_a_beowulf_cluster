#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SEED 35791246

int main(int argc, char* argv[])
{
    int n, i, count = 0, mycount = 0;
    double x, y, z, pi;
    int myid, numprocs, ierr;
    double startwtime = 0.0, endwtime;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        printf("Enter the number of iterations used to estimate pi: ");
        scanf("%d", &n);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    srand(SEED + myid);

    startwtime = MPI_Wtime();

    for (i = 0; i < n/numprocs; i++) {
        x = (double)rand()/RAND_MAX;
        y = (double)rand()/RAND_MAX;
        z = x*x + y*y;
        if (z <= 1.0) mycount++;
    }

    MPI_Reduce(&mycount, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    endwtime = MPI_Wtime();

    if (myid == 0) {
        pi = (double)count/(double)n*4.0;
        printf("The result is %lf\n", pi);
        printf("Time = %f\n", endwtime - startwtime);
    }

    MPI_Finalize();

    return 0;
}
