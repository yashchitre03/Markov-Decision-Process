import re
import numpy as np
import copy
import math
import random

# policy iteration function
def policy_iteration():
    global policy_old
    while True:
        unchanged = True
        utility_new = copy.deepcopy(utility_dash)
        for row in range(size[0]):
            for col in range(size[1]):
                if [row, col] not in avoid:
                    util_list = neighbor(row, col, utility_new[:][:])
                    utility_dash[row][col] = reward[row][col] + discount_rate * expected_utility(util_list[:], row, col, policy_old)
        utility_new = copy.deepcopy(utility_dash)
        for row in range(size[0]):
            for col in range(size[1]):
                if [row, col] not in avoid:
                    util_listo = neighbor(row, col, utility_new[:][:])
                    p_u(util_listo[:], row, col)
                    if policy_new[row][col] != policy_old[row][col]:
                        policy_new[row][col] = policy_old[row][col]
                        unchanged = False
        if unchanged:
            return policy_new

# finding neighbor utilities
def neighbor(row, col, utility_t):
    n_util = []
    if ((row - 1) < 0) or math.isnan(utility_t[row - 1][col]):         # down
        n_util.append(utility_t[row][col])
    else:
        n_util.append(utility_t[row - 1][col])
        
    if ((col - 1) < 0) or math.isnan(utility_t[row][col - 1]):         # left
        n_util.append(utility_t[row][col])
    else:
        n_util.append(utility_t[row][col - 1])
        
    if ((row + 1) >= size[0]) or math.isnan(utility_t[row + 1][col]):  # up
        n_util.append(utility_t[row][col])
    else:
        n_util.append(utility_t[row + 1][col])
        
    if ((col + 1) >= size[1]) or math.isnan(utility_t[row][col + 1]):  # right
        n_util.append(utility_t[row][col])
    else:
        n_util.append(utility_t[row][col + 1])
        
    return n_util

# expected utility calculation
def expected_utility(u_list, row, col, policy_old):
    if policy_old[row][col] == 'D':
        U = (0.8 * u_list[0]) + (0.1 * u_list[1]) + (0 * u_list[2]) + (0.1 * u_list[3])
    elif policy_old[row][col] == 'L':
        U = (0.1 * u_list[0]) + (0.8 * u_list[1]) + (0.1 * u_list[2]) + (0 * u_list[3])        
    elif policy_old[row][col] == 'U':
        U = (0 * u_list[0]) + (0.1 * u_list[1]) + (0.8 * u_list[2]) + (0.1 * u_list[3])      
    else:
        U = (0.1 * u_list[0]) + (0 * u_list[1]) + (0.1 * u_list[2]) + (0.8 * u_list[3])
    return U 

# calculating P*U
def p_u(u_listo, row, col):
    global policy_old
    down = (0.8 * u_listo[0]) + (0.1 * u_listo[1]) + (0 * u_listo[2]) + (0.1 * u_listo[3])
    left = (0.1 * u_listo[0]) + (0.8 * u_listo[1]) + (0.1 * u_listo[2]) + (0 * u_listo[3])
    up = (0 * u_listo[0]) + (0.1 * u_listo[1]) + (0.8 * u_listo[2]) + (0.1 * u_listo[3])
    right = (0.1 * u_listo[0]) + (0 * u_listo[1]) + (0.1 * u_listo[2]) + (0.8 * u_listo[3])
    m = max(down, left, up, right)
    if m == up:
        c = 'N'
    elif m == left:
        c = 'S' if row == 1 else 'W'
    elif m == down:
        c = 'N' if row == 1 else 'S'
    else:
        c = 'E'
    policy_old[row][col] = c
    return m 


# Taking the input from text file and formatting accordingly
iteration = 0
ip = []
for line in open("mdp_input.txt"):
    if not line.startswith(("#", "\n")):
        ip.append(line.rstrip())
        
size = re.findall(r'[\d.]+', ip[0])
size = [int(i) for i in size]

walls = re.findall(r'[\d.]+', ip[1])
walls = np.reshape(walls, (-1, 2))
walls = [[int(j) for j in i] for i in walls]

terminal_states = re.findall(r'[+-]?\d+(?:\.\d+)?', ip[2])
terminal_states = np.reshape(terminal_states, (-1, 3))
terminal_states = [[int(j) for j in i] for i in terminal_states]

r = re.findall(r'[+-]?\d+(?:\.\d+)?', ip[3])
r = float(r[0])

transition_probabilities = re.findall(r'[+-]?\d+(?:\.\d+)?', ip[4])
transition_probabilities = [float(i) for i in transition_probabilities]

discount_rate = re.findall(r'[+-]?\d+(?:\.\d+)?', ip[5])
discount_rate = float(discount_rate[0])

epsilon = re.findall(r'[+-]?\d+(?:\.\d+)?', ip[6])
epsilon = float(epsilon[0])

move = ['L', 'R', 'U', 'D']

policy_old = np.zeros(size)
policy_old = [[str(j) for j in i] for i in policy_old]
policy_old = np.array(policy_old)

policy_new = np.zeros(size)
policy_new = [[random.choice(move) for j in i] for i in policy_new]
policy_new = np.array(policy_new)

reward = np.zeros(size)
avoid = []
utility_dash = np.zeros(size)

for wall in walls:    
    reward[wall[0]][wall[1]] = math.nan
    utility_dash[wall[0]][wall[1]] = math.nan
    policy_old[wall[0]][wall[1]] = "-"
    policy_new[wall[0]][wall[1]] = "-"
    avoid.append([wall[0], wall[1]])
       
for each in terminal_states:
    reward[each[0]][each[1]] = each[2]
    utility_dash[each[0]][each[1]] = each[2]
    avoid.append([each[0], each[1]])
    policy_old[each[0]][each[1]] = 'T'
    policy_new[each[0]][each[1]] = 'T'

reward[reward == 0] = r
utility_dash[utility_dash == 0] = r

policy_new = policy_iteration()

print("\nFinal Policy:")
print(policy_new[::-1])