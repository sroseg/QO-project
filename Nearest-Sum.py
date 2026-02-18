# This is a provisional code for developing an Adiabatic
# Quantum Computing to solve the nearest sum problem.
# Given a set of numbers and a target number 'Sum', the
# algorithm shall select among the set which would be
# feasible solution to the problem. The feasible solution
# is a subset of numbers of which sum gives the
# least amount of error compared to the target 'Sum'.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#Initialization
S = [13,17,19,14] #set of numbers to pick from
L = 30 #target number
n = len(S)
Tfinal = 200 #duration of the adiabatic evolution

# dimensions
N = 2**n

# Prepare the initial Hamiltonian H0, choose sigma X operator
sigmaX = np.array([[0,1],[1,0]])

H0 = np.zeros((N,N), dtype=float)

# This is a function that computes the kronecker product  of a given gate and
# the identity matrix for the nth qubit
def nqubits(gate,k,n):
    eye = np.eye(2)
    gates = []

    for i in range(n):
        gates.append(gate if i == k else eye)
    matrix_result = gates[0]
    for j in gates[1:]:
        matrix_result = np.kron(matrix_result,j)
    return matrix_result

for k in range(n):
    H0 = H0 + nqubits(sigmaX,k,n)

H0 = -H0 #change signs

# Prepare the Final Hamiltonian
matrix_A = np.array([[0,0],[0,1]])

Hf = np.zeros((N,N), dtype=float)
for k in range(n):
    Hf = Hf + (S[k]*nqubits(matrix_A,k,n))

# Encoding the problem to the final Hamiltonian Hf
Hf = Hf - (L * np.eye(N))

#Square the matrix
Hf = np.matmul(Hf,Hf)

# Preparing the solution for the TDSE
# Uniform superposition state (initial values)
PsiInit = 1/np.sqrt(N)*np.ones(N,dtype=complex)


# Schedule function
def S_func(t):
    return 0.5*(1-np.cos(np.pi*t/Tfinal))

# RHS of the TDSE
def RHS(t,y):
    Ham = (1-S_func(t)) * H0 + S_func(t)* Hf
    Yderiv = np.matmul(Ham,y)
    return -1j * Yderiv

# Solving the TDSE using solver solve_ivp with default method rk45
# set time span
t_span = (0.0,float(Tfinal))

# set absolute and relative tolerances to get the desired accuracy
sol = solve_ivp(RHS,t_span,PsiInit,atol=1e-5,rtol=1e-5)

# Storing the final values
PsiFinal = sol.y[:,-1]
PsiFinalVector = np.abs(PsiFinal)**2

max_index = round(PsiFinalVector.argmax()) #rounding off to force int values

# This is a function to convert the index of the final state into a bitstream
def convert2bitstream(bitstream,max_index):

    bitstream = []

    while max_index > 0:
        number = max_index % 2
        bitstream.append(number)
        max_index = max_index // 2 # floor division

    bitstream.reverse()
    return bitstream

num2str = ' '.join(map(str,convert2bitstream(bitstream,max_index)))
print("The bitstream is", num2str)

# Computing for the fidelity which is the maximum value of the final state
Fid = f"{PsiFinalVector.max():.4%}"
print("Fidelity is", Fid)
# Check the norm conservation
norm = f"{np.linalg.norm(PsiFinalVector): .4%}"
print("Final norm is", norm)

# plot PsiFinal into a histogram
x = np.arange(0,N)
plt.bar(x,PsiFinalVector)
plt.show()






