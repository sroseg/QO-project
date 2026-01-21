# Created a function that helps prepare the initial Hamiltonian
# The resulting matrix is the Kronecker product of a single qubit gate
# for each qubit k and the identity matrices


import numpy as np


def nqubits(gate,k,N):
    gates = [] #create a list to access the single qubit gate and/or the
               # identity matrix for N number of qubits
    eye = np.eye(2)
    for i in range(N):
        gates.append(gate if i == k else eye)
    matrix_result = gates[0]
    for j in gates[1:]:
        matrix_result = np.kron(matrix_result,j)
    return matrix_result
