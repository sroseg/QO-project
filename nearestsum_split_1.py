# This is a provisional code for developing an Adiabatic
# Quantum Computing to solve the nearest sum problem.
# Given a set of numbers and a target number 'Sum', the
# algorithm shall select among the set which would be
# feasible solution to the problem. The feasible solution
# is a subset of numbers of which sum gives the
# least amount of error compared to the target 'Sum'.

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time

start = time.process_time()


#Initialization
S = [25,19,17,10] #set of numbers to pick from
L = 40 #target number
n = len(S)

#time parameters
Tfinal = 2000 #duration of the adiabatic evolution
N_steps = 50000
dt = Tfinal/N_steps

# dimensions
N = 2**n

# Prepare the initial Hamiltonian H0, choose sigma X operator
sigmaX = np.array([[0,1],[1,0]])

H0 = np.zeros((N,N), dtype=float)

# This is a function that computes the kronecker product  of a given gate and
# the identity matrix for the nth qubit
def nqubits(gate,k,n):
    I_2 = np.eye(2)
    gates = []

    for i in range(n):
        gates.append(gate if i == k else I_2)
    matrix_result = gates[0]
    for j in gates[1:]:
        matrix_result = np.kron(matrix_result,j)
    return matrix_result

def nqubits_vector(gate,k,n):
    I_vect = [1,1]
    gates = []

    for i in range(n):
        gates.append(gate if i == k else I_vect)
    matrix_result = gates[0]
    for j in gates[1:]:
        matrix_result = np.kron(matrix_result,j)
    return matrix_result

for k in range(n):
    H0 = H0 + nqubits(sigmaX,k,n)

H0 = -H0 #change signs

# Prepare the Final Hamiltonian
vector_A = np.array([0,1])

Hfdiag = np.zeros((1,N), dtype=float)
for k in range(n):
    Hfdiag = Hfdiag + (S[k]*nqubits_vector(vector_A,k,n))

# Encoding the problem to the final Hamiltonian Hf
Hfdiag = Hfdiag - L * np.ones((1,N))

#Square the matrix
Hfdiag = Hfdiag ** 2
Hfdiag = Hfdiag.flatten()

step1_time = time.process_time() - start
print('Encoding the problem into the Hamiltonian was successfully executed at: %s seconds' % (round((time.process_time() - start), 4)))

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
    U_B = np.exp(-1j*S_func(t+dt/2)*Hfdiag*dt)
    Psi = U_A @ Psi # update U_A term in Psi
    Psi = U_B * Psi # update U_B term in Psi
    Psi = U_A @ Psi # update 
    U_A = U_A @ U_APlus
    t += dt



# Storing the final values
PsiFinal = Psi
PsiFinal_sq = np.abs(PsiFinal)**2

step2_time = time.process_time()
print('The time it took to compute the solution: %s seconds' % (round(step2_time-step1_time, 4)))

max_index = round(PsiFinal_sq.argmax()) #rounding off to force int values
print(PsiFinal_sq.argmax())
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
Fid = f"{PsiFinal_sq.max():.4%}"
print("Fidelity is", Fid)
# Check the norm conservation
norm = np.sum(PsiFinal_sq)
print(f"Final norm is {norm:.4f}")

# plot PsiFinal into a histogram
x = np.arange(0,N)
plt.figure(1)
plt.clf()
plt.bar(x,PsiFinal_sq)
plt.show()

bits = convert2bitstring(max_index,n)
new_S = []
for i in range(n):
    if bits[0,i] == 1:
        new_S.append(S[i])

print("The values are:", new_S)

print(f"CPU Time: {time.process_time() - start:.6f} seconds")





