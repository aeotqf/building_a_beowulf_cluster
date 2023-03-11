#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000
#define M 1000

int main(int argc, char** argv)
{
    int rank, size, i, j, k;
    double A[N][M], B[M][N], C[N][N];
    double startTime, endTime;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int chunk_size = N / size;
    int start = rank * chunk_size;
    int end = (rank + 1) * chunk_size;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            A[i][j] = (double)rand() / RAND_MAX;
        }
    }

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            B[i][j] = (double)rand() / RAND_MAX;
        }
    }

    startTime = MPI_Wtime();

    for (i = start; i < end; i++) {
        for (j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (k = 0; k < M; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, C, N*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    endTime = MPI_Wtime();

    if (rank == 0) {
        printf("Matrix multiplication took %f seconds\n", endTime - startTime);
    }

    MPI_Finalize();
    return 0;
}


// in c we need to compile first:
//mpicc -o matrix_multiplication matrix_multiplication.c

//then run:
//mpirun -np 8 ./matrix_multiplication
