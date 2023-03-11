from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Define the size of the matrices
N = 1000
M = 1000

# Divide the work among the processes
chunk_size = N // size
start = rank * chunk_size
end = (rank + 1) * chunk_size

# Create the matrices
A = np.random.rand(N, M)
B = np.random.rand(M, N)
C = np.zeros((N, N))

# Perform the matrix multiplication in parallel
for i in range(start, end):
    for j in range(N):
        for k in range(M):
            C[i][j] += A[i][k] * B[k][j]

# Combine the results from all processes
comm.Allreduce(MPI.IN_PLACE, C, op=MPI.SUM)

# Print the result
if rank == 0:
    print(C)

"""
Since this program is written in Python and uses the MPI4Py library, 
there is no need to compile the code before running
"""
#run:
#mpirun -np 8 python matrix_multiplication.py

