#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define DIM 3

int num_particles = 1000;
double box_size = 10.0;
double cutoff_distance = 2.5;
double time_step = 0.005;
int output_frequency = 10;

void compute_forces(double positions[][DIM], double forces[][DIM], int num_particles_local, int start, int end) {
    int i, j, k;
    double distance_vector[DIM], distance, magnitude, force_vector[DIM];
    double potential_energy = 0.0;

    for (i = start; i < end - 1; i++) {
        for (j = i + 1; j < end; j++) {
            distance = 0.0;
            for (k = 0; k < DIM; k++) {
                distance_vector[k] = positions[i][k] - positions[j][k];
                distance_vector[k] -= box_size * round(distance_vector[k] / box_size);
                distance += distance_vector[k] * distance_vector[k];
            }
            distance = sqrt(distance);

            if (distance < cutoff_distance) {
                magnitude = 48 * (pow(1 / distance, 14) - 0.5 * pow(1 / distance, 8));
                for (k = 0; k < DIM; k++) {
                    force_vector[k] = magnitude * distance_vector[k];
                    forces[i - start][k] += force_vector[k];
                    forces[j - start][k] -= force_vector[k];
                }
                potential_energy += 4 * (pow(1 / distance, 12) - pow(1 / distance, 6));
            }
        }
    }
}

void integrate(double positions[][DIM], double velocities[][DIM], double forces[][DIM], int num_particles_local, int start, int end) {
    int i, j;
    double potential_energy = 0.0, kinetic_energy = 0.0;

    for (i = start; i < end; i++) {
        for (j = 0; j < DIM; j++) {
            positions[i][j] += velocities[i - start][j] * time_step + 0.5 * forces[i - start][j] * time_step * time_step;
            velocities[i - start][j] += 0.5 * forces[i - start][j] * time_step;
        }
    }

    compute_forces(positions, forces, num_particles_local, start, end);

    for (i = start; i < end; i++) {
        for (j = 0; j < DIM; j++) {
            velocities[i - start][j] += 0.5 * forces[i - start][j] * time_step;
        }
        kinetic_energy += 0.5 * (velocities[i - start][0] * velocities[i - start][0] +
                                 velocities[i - start][1] * velocities[i - start][1] +
                                 velocities[i - start][2] * velocities[i - start][2]);
    }

    MPI_Allreduce(MPI_IN_PLACE, &potential_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &kinetic_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0 && step % output_frequency == 0) {
        double total_energy = potential_energy + kinetic_energy;
        printf("Step %d, Total Energy: %.2f, Potential Energy: %.
