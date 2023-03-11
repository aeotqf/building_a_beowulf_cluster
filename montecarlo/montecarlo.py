import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set parameters
num_steps = 1000000
step_size = 0.1
init_pos = [0.0, 0.0, 0.0]

# Set up random number generator
seed = rank + 1
rng = np.random.default_rng(seed)

# Define the function to calculate the energy of the system
def calculate_energy(position):
    return np.sum(np.square(position))

# Define the Monte Carlo algorithm
def run_monte_carlo(init_pos):
    current_pos = np.array(init_pos)
    current_energy = calculate_energy(current_pos)

    for i in range(num_steps):
        # Generate a trial move
        trial_pos = current_pos + rng.normal(size=3) * step_size
        trial_energy = calculate_energy(trial_pos)

        # Accept or reject the move based on the Metropolis criterion
        delta_energy = trial_energy - current_energy
        if delta_energy < 0 or rng.random() < np.exp(-delta_energy):
            current_pos = trial_pos
            current_energy = trial_energy

    return current_energy

# Divide the work among the processes
num_trials_local = num_trials // size
start = rank * num_trials_local
end = start + num_trials_local

# Run the Monte Carlo simulation in parallel
energy_local = run_monte_carlo(init_pos)
energy_all = comm.gather(energy_local, root=0)

# Output the result on the root process
if rank == 0:
    energy_avg = np.mean(energy_all)
    print(f"Average energy: {energy_avg:.4f}")
