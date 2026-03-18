# This is a provisional code for developing an Adiabatic
# Quantum Computation to solve the knapsack problem.
# Given a set of n valuable items with weights,
# and a knapsack with capacity C,
# the algorithm shall select among the items that could
# maximize the value of the packed knapsack.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time


start = time.time()
#Initialization
W = [3,4,6,7] #set of items to pick from
V = [19,13,10,25] # value of the items to be put in the knapsack (to be maximized)
C = 10 #capacity
yvar = 0 # term to enforce that if final knapsack weight is n, y=1 otherwise 0
A = 5 #penalty term for HA
B = 5 #penalty term for HB
n = len(W) # introducing a slack variable
Tfinal = 100 #duration of the adiabatic evolution

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

# Encoding the problem to the final Hamiltonian Hf
# HA = A * [(yvar - sumW)**2]
# HB = -C * sumV
# Hf = HA + HB

# Prepare the Final Hamiltonian

matrix_A = np.array([[0,0],[0,1]])
matrix_B = np.array([[0,0],[0,1]])

# encoding the weight constraint which is part of HA
sumW = np.zeros((N,N), dtype=float)
for k in range(n):
    sumW = sumW + (W[k]*nqubits(matrix_A,k,n))

sumW = sumW - (C * np.eye(N))
print(sumW)
# yvar term
for k in range(n):
    yvar = yvar + 2**(k) * nqubits(matrix_A,k,n)
print(yvar)

HA = (yvar - sumW)
HA = A * np.matmul(HA,HA)
print(HA)
# encoding the maximizing the items of the knapsack in HB
sumV = np.zeros((N,N), dtype=float)
for k in range(n):
    sumV = sumV + (V[k] * nqubits(matrix_B, k, n))

HB = -B * sumV
HB = np.matmul(HB,HB)
print(HB)
Hf = HA + HB

step1_time = time.time() - start
print('Encoding the problem into the Hamiltonian was successfully executed at: %s seconds' % (round((time.time() - start), 4)))

#Square the matrix
Hf = np.matmul(Hf,Hf)

print("Preparing the solution")
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

print(f"Now solving IVP with 2**{n} dimension")
# set absolute and relative tolerances to get the desired accuracy
sol = solve_ivp(RHS,t_span,PsiInit,atol=1e-7,rtol=1e-7)

# Storing the final values
PsiFinal = sol.y[:,-1]
PsiFinalVector = np.abs(PsiFinal)**2
step2_time = time.time()
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






