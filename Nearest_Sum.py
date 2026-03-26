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
import time

start = time.process_time()
print(f"CPU Time: {time.process_time() - start:.6f} seconds")

#Initialization
S = [25,19,17,14,13] #set of numbers to pick from
L = 40 #target number
n = len(S)
Tfinal = 400 #duration of the adiabatic evolution

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

step1_time = time.process_time() - start
print('Encoding the problem into the Hamiltonian was successfully executed at: %s seconds' % (round((time.process_time() - start), 4)))

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

step2_time = time.process_time()
print('The time it took to compute the solution: %s seconds' % (round(step2_time-step1_time, 4)))

max_index = round(PsiFinalVector.argmax()) #rounding off to force int values
print(PsiFinalVector.argmax())
# This is a function to convert the index of the final state into a bitstream
def convert2bitstring(max_index,n):

    bitstr = np.zeros((1,n), dtype=int)
    number = max_index
    for i in range(n):
        check = number // 2**(n-i-1) # floor division
        bitstr[0,i] = check
        number = number - check*2**(n-i-1)

    return bitstr

num2str = ' '.join(map(str,convert2bitstring(max_index,n)))
print("The bitstring is", num2str)

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

bits = convert2bitstring(max_index,n)
new_S = []
for i in range(n):
    if bits[0,i] == 1:
        new_S.append(S[i])

print("The values are:", new_S)





