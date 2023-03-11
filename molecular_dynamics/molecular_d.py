import numpy as np
from mpi4py import MPI

def compute_forces(positions, forces):
    global box_size, cutoff_distance

    forces[:, :] = 0.0
    potential_energy = 0.0

    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            distance_vector = positions[i] - positions[j]
            distance_vector = distance_vector - box_size * np.round(distance_vector / box_size)
            distance = np.linalg.norm(distance_vector)

            if distance < cutoff_distance:
                magnitude = 48 * ((1 / distance)**14 - 0.5 * (1 / distance)**8)
                force_vector = magnitude * distance_vector
                forces[i] += force_vector
                forces[j] -= force_vector
                potential_energy += 4 * ((1 / distance)**12 - (1 / distance)**6)

    return potential_energy

def integrate(positions, velocities, forces):
    global time_step

    positions += velocities * time_step + 0.5 * forces * time_step**2

    old_forces = forces.copy()
    potential_energy = compute_forces(positions, forces)

    velocities += 0.5 * (old_forces + forces) * time_step
    kinetic_energy = 0.5 * np.sum(velocities**2)

    return potential_energy, kinetic_energy

def run_simulation(num_steps):
    global positions, velocities, forces

    for step in range(num_steps):
        potential_energy, kinetic_energy = integrate(positions, velocities, forces)

        if step % output_frequency == 0:
            total_energy = potential_energy + kinetic_energy
            if rank == 0:
                print(f"Step {step}, Total Energy: {total_energy:.2f}, "
                      f"Potential Energy: {potential_energy:.2f}, "
                      f"Kinetic Energy: {kinetic_energy:.2f}")

if __name__ == '__main__':
    # Simulation parameters
    num_particles = 1000
    box_size = 10.0
    cutoff_distance = 2.5
    time_step = 0.005
    output_frequency = 10

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divide particles among processes
    chunk_size = num_particles // size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size

    # Initialize positions, velocities, and forces
    np.random.seed(0)
    positions = np.random.uniform(0, box_size, (num_particles, 3))[start:end]
    velocities = np.zeros((chunk_size, 3))
    forces = np.zeros((chunk_size, 3))

    # Run simulation
    run_simulation(500)
