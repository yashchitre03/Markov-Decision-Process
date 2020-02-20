import re
import numpy as np
import copy
import math

# value iteration function
def value_iteration():
    global iteration
    while True:
        iteration += 1
        utility = copy.deepcopy(utility_dash)
        delta = 0
        rev_u = utility_dash[::-1]
        print("\nIteration: ", iteration)
        print(rev_u)
        for row in range(size[0]):
            for col in range(size[1]):
                if [row, col] not in avoid:
                    util_list = neighbor(row, col, utility[:][:])
                    maximum = p_u(util_list[:], row, col)
                    utility_dash[row][col] = reward[row][col] + discount_rate * maximum
                    temp = abs(utility_dash[row][col] - utility[row][col])
                    if temp > delta:
                        delta = temp
        if delta < (epsilon * (1 - discount_rate) / discount_rate):
            break
    return utility


# finding the utility of neighbors
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


# calculating P*U
def p_u(u_list, row, col):
    global moves
    down = (0.8 * u_list[0]) + (0.1 * u_list[1]) + (0.0 * u_list[2]) + (0.1 * u_list[3])
    left = (0.1 * u_list[0]) + (0.8 * u_list[1]) + (0.1 * u_list[2]) + (0.0 * u_list[3])
    up = (0.0 * u_list[0]) + (0.1 * u_list[1]) + (0.8 * u_list[2]) + (0.1 * u_list[3])
    right = (0.1 * u_list[0]) + (0.0 * u_list[1]) + (0.1 * u_list[2]) + (0.8 * u_list[3])
    m = max(down, left, up, right)
    if m == down:
        c = 'S'
    elif m == left:
        c = 'W'
    elif m == up:
        c = 'N'
    else:
        c = 'E'
    moves[row][col] = c
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

moves = np.zeros(size)
moves = [[str(j) for j in i] for i in moves]
moves = np.array(moves)

avoid = []
utility_dash = np.zeros(size)
for wall in walls:    
    utility_dash[wall[0]][wall[1]] = math.nan
    moves[wall[0]][wall[1]] = "-"
    avoid.append([wall[0], wall[1]])
    
print("Iteration: ", iteration)
rev_u = utility_dash[::-1]
print(rev_u)
   
reward = np.zeros(size)
for each in terminal_states:
    reward[each[0]][each[1]] = each[2]
    utility_dash[each[0]][each[1]] = each[2]
    avoid.append([each[0], each[1]])
    moves[each[0]][each[1]] = 'T'
    
reward[reward == 0] = r
utility_dash[utility_dash == 0] = r

utility = value_iteration()

print("\nFinal Value after Convergence:")
print(utility[::-1])
print("\nFinal Policy:")
print(moves[::-1])