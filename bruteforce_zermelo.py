
# An attempt to find the optimal distance of the river-crossing problem using 
# brute force method.


import numpy as np
from itertools import product

# Initialization
m = 6 # y-axis
n = 5 # x - axis
D = 6 # distance from the left bank of the river to the right bank 


# List all possible nodes 
setA = list(range(1,m+1)) 
setB = list(range(1,n+1))

nodes = list(product(setA, setB))

# Time Costs 
# v = 
# Sfunc = 0.9 * v * np.expm(x-D/y)
# L = np.sqrt((x_f - x_i)**2 + (y_f - y_i)**2)
# v = 
# t = L/v
# u_vector

def distance(node_i, node_f):
    x_i, y_i = node_i
    x_f, y_f = node_f

    return np.sqrt((x_f - x_i)**2 + (y_f - y_i)**2)


def calculate_cost(route):
    total_cost = 0

    for i in range(len(route) - 1):
        total_cost += distance(route[i], route[i + 1])

    return total_cost
    

# Listing all possible routes in a list
routes = [list(zip(setA, steps))
    for steps in product(setB, repeat=m)]

#print(routes[0:15624])

best_route = []
best_cost = 10000 # big enough number

# Iterative process to calculate routes
for steps in product(setB, repeat=m):
    route = list(zip(setA, steps))

    # calculate time costs
    time_cost = calculate_cost(route)
    
    if time_cost < best_cost:
        best_cost = time_cost
        best_route = route

    

print(best_route)
