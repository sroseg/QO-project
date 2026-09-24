# An attempt to find the optimal distance of the river-crossing problem using 
# brute force method.


import numpy as np
from itertools import product

# Initialization
n = 6 # x-axis
m = 5 # y - axis
D = 1 # distance from the left bank of the river to the right bank 
L_y = .25
dx = D / (n+1)
dy = L_y / (m-1) 

# List all possible nodes 
setA = list(range(1,n+1)) 
setB = list(range(1,m+1))

nodes = list(product(setA, setB))

# Time Costs 
v = 1 
def Sfunc(x,y):
       return 0.9 * v * np.exp(-(x-D/np.pi)**2)
   
# L = np.sqrt((x_f - x_i)**2 + (y_f - y_i)**2)
# t = L/v

def distance(node_i, node_f):
    x_i, y_i = node_i
    x_f, y_f = node_f

    return np.sqrt((x_f - x_i)**2+ (y_f - y_i)**2)

def time_step(node_i,node_f):
    x_i= node_i[0]*dx
    x_f= node_f[0]*dx
    y_i= node_i[1]*dy - L_y/2
    y_f = node_f[1]*dy - L_y/2
    delta = (y_f - y_i) / (x_f - x_i) 
    S = Sfunc((x_i + x_f)/2, (y_i + y_f)/2)
    return ((x_f - x_i) * (1 + delta**2)) / (np.sqrt((1+ delta**2) * v**2 - S**2)- delta * S) 
    

def calculate_cost(route):
    total_cost = 0

    for i in range(len(route) - 1):
        total_cost += time_step(route[i], route[i + 1])

    total_cost += time_step((0,int((m+1)/2)), route[0]) 
    total_cost += time_step(route[-1], (n+1,int((m+1)/2)))     

    return total_cost
    

# Listing all possible routes in a list
routes = [list(zip(setA, steps))
    for steps in product(setB, repeat=n)]

#print(routes[0:15624])

best_route = []
best_cost = 10000 # big enough number

# Iterative process to calculate routes
for steps in product(setB, repeat=n):
    route = list(zip(setA, steps))

    # calculate time costs
    time_cost = calculate_cost(route)
    
    if time_cost < best_cost:
        best_cost = time_cost
        best_route = route

route_f = [j for i,j in best_route]
print(route_f)
