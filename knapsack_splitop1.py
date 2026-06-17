# This is a provisional code for developing an Adiabatic
# Quantum Computation to solve the knapsack problem.
# Given a set of n valuable items with weights,
# and a knapsack with capacity L,
# the algorithm shall select among the items that could
# maximize the value of the packed knapsack.

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time


start = time.perf_counter()
#Initialization
W = [3,4,7] #set of items with weights to pick from
V = [19,30,10] # value of the items to be put in the knapsack (to be maximized)
L = 10 #capacity
A = 1 #penalty term for HA
B = 5 #penalty term for HB

n = len(V) 
m = int(np.ceil(np.log2(L)))

# Augmented set of of W and V
aux =  2 ** np.arange(m)
W = np.append(W, aux)
W = np.array(W,dtype=int)
V_2 = np.append(V, aux*0)


#time parameters
Tfinal = 500 #duration of the adiabatic evolution
N_steps = 100000
dt = Tfinal/N_steps

# dimensions
dim = n + m
N = 2**dim
qbits = 2**n


# Prepare the initial Hamiltonian H0, choose sigma X operator
sigmaX = np.array([[0,1],[1,0]])

H0 = np.zeros((N,N), dtype=float)

# This is a function that computes the kronecker product  of a given gate and
# the identity matrix for the nth qubit
def nqubits(gate,k,dim):
    eye = np.eye(2)
    gates = []

    for i in range(dim):
        gates.append(gate if i == k else eye)
    matrix_result = gates[0]
    for j in gates[1:]:
        matrix_result = np.kron(matrix_result,j)
    return matrix_result


def nqubits_vector(gate,k,dim):
    I_vect = [1,1]
    gates = []

    for i in range(dim):
        gates.append(gate if i == k else I_vect)
    matrix_result = gates[0]
    for j in gates[1:]:
        matrix_result = np.kron(matrix_result,j)
    return matrix_result

for k in range(dim):
    H0 = H0 + nqubits(sigmaX,k,dim)

H0 = -H0 #change signs

# Prepare the Final Hamiltonian

matrix_A = np.array([[0,0],[0,1]])


    
V_sum = np.zeros((N,N), dtype=float)
W_width = np.zeros((N,N), dtype=float)
for k in range(dim):
    V_sum = V_sum + (V_2[k]*nqubits(matrix_A,k,dim))
    W_width = W_width + (W[k]*nqubits(matrix_A,k,dim))

I = np.eye(N,N)
HA = -A * V_sum
HB = B * (W_width - L * I) @ (W_width - L * I)
Hf = HA + HB 


vector_A = np.array([0,1])
vector_B = np.array([0,1])

V_sum_diag = np.zeros((1,N), dtype=float)
W_width_diag = np.zeros((1,N), dtype=float)
for k in range(dim):
    V_sum_diag = V_sum_diag + (V_2[k]*nqubits_vector(vector_A,k,dim))
    W_width_diag = W_width_diag + (W[k]*nqubits_vector(vector_A,k,dim))


# Encoding the problem to the final Hamiltonian Hf
I = np.ones((N))
HA = -A * V_sum_diag.flatten()
HB = B * (W_width_diag.flatten() - L * I) ** 2
Hfdiag = HA + HB 


#Square the matrix
#Hfdiag = Hfdiag ** 2
#Hfdiag = Hfdiag.flatten()


step1_time = time.perf_counter()
print('Encoding the problem into the Hamiltonian was successfully executed at: %s seconds' % (round((step1_time - start), 4)))


print("Preparing the solution")
# Preparing the solution for the TDSE
# Uniform superposition state (initial values)
PsiInit = 1/np.sqrt(N)*np.ones(N,dtype=complex)


# Schedule function
def S_func(t):
#    return 0.5*(1-np.cos(np.pi*t/Tfinal))
    return 1/Tfinal*t


U_A = linalg.expm(-1j*dt/2*(1-dt/(2*Tfinal))*H0)
U_APlus = linalg.expm(1j*H0*dt**2/(2*Tfinal))

t = 0
Psi = PsiInit
while t < Tfinal:
    U_A = linalg.expm(-1j*(1-S_func(t+dt/2))*H0*dt/2)
    U_B = np.exp(-1j*S_func(t+dt/2)*Hfdiag*dt)
    Psi = U_A @ Psi # update U_A term in Psi
    Psi = U_B * Psi # update U_B term in Psi
    Psi = U_A @ Psi # update 
    U_A = U_A @ U_APlus
    t += dt


# Storing the final values
PsiFinal = Psi
PsiFinal_SQ = np.abs(PsiFinal)**2


max_index = round(PsiFinal_SQ.argmax()) #rounding off to force int values

step2_time = time.perf_counter()
print('The time it took to compute the solution: %s seconds' % (round(step2_time-step1_time, 4)))


# This is a function to convert the index of the final state into a bitstream
def convert2bitstring(max_index,n):

    bitstr = np.zeros((1,n), dtype=int)
    number = max_index
    for i in range(n):
        check = number // 2**(n-i-1) # floor division
        bitstr[0,i] = check
        number = number - check*2**(n-i-1)

    return bitstr

bits_full = convert2bitstring(max_index, dim)
bits_item = bits_full[0, :n]      

num2str = ' '.join(map(str, bits_item))

print("The bitstring is", num2str)


# Computing for the fidelity which is the maximum value of the final state
Fid = f"{PsiFinal_SQ.max():.4%}"
print("Fidelity is", Fid)
# Check the norm conservation
norm = np.sum(PsiFinal_SQ)
print("Final norm is", norm)

prob_items = np.zeros(2**n)
for i, j in enumerate(PsiFinal_SQ):
    bits_full = convert2bitstring(i, dim)[0]
    b_item = bits_full[:n]

    item_index = int("".join(map(str, b_item)), 2)
    prob_items[item_index] += j

x = np.arange(0,qbits)
plt.figure(1)
plt.clf()
plt.bar(x,prob_items)
plt.show()

bits = np.array(list(map(int, bits_item)))
new_W = []
new_V = []
for i in range(n):
    if bits[i] == 1:
        new_W.append(W[i])
        new_V.append(V[i])


print("The total weight of the knapsack:", sum(new_W))
print("The optimal value of the knapsack", sum(new_V))

end_time = time.perf_counter()
print('The time it took to compute the solution: %s seconds' % (round(end_time-start, 4)))








