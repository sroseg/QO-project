# Classical solution(brute force) to solve the nearest sum
# problem.

from itertools import combinations

# Problem: find the subset/s of the set S of which its sum
# is the nearest to the target L

#initialization
S = (13,17,19,14)
L = 30
sub_S = []
n = len(S)

# list all subsets of set S

i = 1
while i < n+1:
    comb_len = list(combinations(S, i))
    sub_S.extend(comb_len)
    i = i + 1

print("These are the combinations of the set S:")
print(sub_S)

# calculate the differences between the sums of each subset vs. L
err_list = []
for i in sub_S:
    err_list.append(abs(sum(list(i))-L))

min_err = min(err_list)
get_indices = [j for j, x in enumerate(err_list) if x == min_err]

print("These are the subsets of the set S that are nearest to L:")
for i in get_indices:
    print(sub_S[i])