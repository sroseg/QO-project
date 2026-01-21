# This is a provisional code for developing an Adiabatic
# Quantum Computing to solve the nearest sum problem.
# Given a set of numbers and a target number 'Sum', the
# algorithm shall select among the set which would be
# feasible solution to the problem. The feasible solution
# is a subset of numbers of which sum gives the
# least amount of error compared to the target 'Sum'.

import numpy as np
import math
from scipy import linalg
from numpy.f2py.crackfortran import endifs

#Initialization
S = (13,17,19,110)
L = 30
T = 15
N = len(S)

# Prepare the initial Hamiltonian, choose sigma X operator
sigmaX = np.array([[0,1],[1,0]])

H0 = np.zeros((2**N,2**N), dtype=int)

def nqubits(gate,k,N):
    eye = np.eye(2)
    gates = []

    for i in range(N):
        gates.append(gate if i == k else eye)
    matrix_result = gates[0]
    for j in gates[1:]:
        matrix_result = np.kron(matrix_result,j)
    return matrix_result

for k in range(N):
    H0 = H0 + nqubits(sigmaX,k,N)

H0 = -H0 #change signs

# Prepare the Final Hamiltonian
matrix_A = np.array([[0,0],[0,1]])

Hf = np.zeros((2**N,2**N), dtype=float)
Smat = np.array(S)
for k in range(N):
    Hf += Smat[k]*nqubits(matrix_A,k,N)

Hf = Hf - (L * np.eye(2**N))
Hf = Hf**2

print(Hf)

# Uniform superposition state
PsiInit = 1/math.sqrt(2**N)*np.ones(2**N,dtype=float)

print(PsiInit)

# Schedule function

#S_func =

