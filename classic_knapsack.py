# Brute force code for the knapsack problem
# Given a set of n valuable items with weights,
# and a knapsack with capacity L,
# the algorithm shall select among the items that could
# maximize the value of the packed knapsack.

import numpy as np

#Initialization
W = [3,4,6,7] #set of items to pick from
V = [19,13,10,25] # value of the items to be put in the knapsack (to be maximized)
L = 10 #capacity of the knapsack
A = 5 # initial acceptable value for the value of the knapsack
B = 30 # initial threshold for the number of items that can be picked

n = len(V)
m = int(np.ceil(np.log2(L)))
aux =  2 ** np.arange(m)

# Augmented set of of W and V
W = np.append(W, aux)
W = np.array(W,dtype=int)
V_2 = np.append(V, aux*0)

# dimensions
dim = n+m
N = 2**dim

best_x = None

for i in range(N):
    x = np.fromiter(f"{i:0{dim}b}", dtype=int)

    V_sum = np.matmul(x,V_2)
    W_width = np.matmul(x, W) - L
    W_sum = np.square(W_width)

    if W_sum <= B:
        if V_sum >= A:
            B = W_sum
            A = V_sum
            best_x = x.copy()


print("Best bitstring:", best_x)
print("Optimal value of the knapsack:", A)

# Print the total weight of the knapsack
sum_of_W = 0
for i in range(dim):
    if best_x.tolist()[i] == 1:
        sum_of_W = sum_of_W + W[i]

print("The total weight of the knapsack:", sum_of_W)