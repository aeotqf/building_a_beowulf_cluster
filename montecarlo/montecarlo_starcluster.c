/*
  An MPI program that simulates the evolution of a star cluster
  using a Monte Carlo algorithm.

  Based on the example code from "An Introduction to Parallel
  Programming" by Peter Pacheco.

  look for the reference
*/

/*
  An MPI program that simulates the evolution of a star cluster
  using a Monte Carlo algorithm.

  Based on the example code from "An Introduction to Parallel
  Programming" by Peter Pacheco.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define PI 3.14159265358979323846

/* function to generate random numbers */
double rand_number() {
  return (double)rand() / (double)RAND_MAX;
}

/* function to calculate the radius of the star cluster */
double cluster_radius(int nstars) {
  return pow((3.0 * nstars) / (4.0 * PI), 1.0 / 3.0);
}

/* function to calculate the initial velocities of the stars */
void init_velocities(double *v, int n) {
  int i;
  for (i = 0; i < n; i++) {
    v[i] = rand_number();
  }
}

/* function to calculate the potential energy of a star */
double potential_energy(double *r, int n, int i) {
  int j;
  double pe = 0.0;
  for (j = 0; j < n; j++) {
    if (i != j) {
      pe += 1.0 / r[j];
    }
  }
  return pe;
}

/* function to calculate the total potential energy of the star cluster */
double total_potential_energy(double *r, int n) {
  int i;
  double tpe = 0.0;
  for (i = 0; i < n; i++) {
    tpe += potential_energy(r, n, i);
  }
  return tpe / 2.0;
}

/* function to update the positions and velocities of the stars */
void update_stars(double *r, double *v, int n, double dt) {
  int i, j;
  double r_ij, f_ij, r_ij_sq, dx, dy, dz;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      if (i != j) {
        dx = r[j * 3] - r[i * 3];
        dy = r[j * 3 + 1] - r[i * 3 + 1];
        dz = r[j * 3 + 2] - r[i * 3 + 2];
        r_ij_sq = dx * dx + dy * dy + dz * dz;
        r_ij = sqrt(r_ij_sq);
        f_ij = 1.0 / (r_ij_sq * r_ij);
        v[i * 3] += dx * f_ij * dt;
        v[i * 3 + 1] += dy * f_ij * dt;
        v[i * 3 + 2] += dz * f_ij * dt;
      }
    }
  }
  for (i = 0; i < n; i++) {
    r[i * 3] += v[i * 3] * dt;
    r[i * 3 + 1] += v[i * 3 + 1] * dt;
    r[i * 3 + 2] += v[i * 3 + 2] * dt;
  }
}

/* function to simulate the evolution of the star cluster */
void simulate_star_cluster(int nstars, double time_step, int nsteps, int rank, int size) {
  int i,
