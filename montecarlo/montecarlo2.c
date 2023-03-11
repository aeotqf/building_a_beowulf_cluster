#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char **argv) {
    int rank, num_procs;
    double pi_approx = 0, local_sum = 0, x, y;
    int i, n = 10000000, local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    srand(time(NULL) + rank);

    local_n = n / num_procs;

    for (i = 0; i < local_n; i++) {
        x = (double) rand() / RAND_MAX;
        y = (double) rand() / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_sum += 1.0;
        }
    }

    MPI_Reduce(&local_sum, &pi_approx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi_approx = 4.0 * pi_approx / n;
        printf("Approximate value of pi: %f\n", pi_approx);
    }

    MPI_Finalize();

    return 0;
}
