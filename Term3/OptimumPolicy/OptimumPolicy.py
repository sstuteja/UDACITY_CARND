# User Instructions:
# 
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's 
# optimal path to the position specified in goal; 
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a 
# right turn.

forward = [[-1, 0],  # go up
           [0, -1],  # go left
           [1, 0],  # go down
           [0, 1]]  # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [0, 4, 3]  # given in the form [direction,row,col]
# direction = 0: up
#             1: left
#             2: down
#             3: right

goal = [2, 0]  # given in the form [row,col]

cost = [2, 1, 20]  # cost has 3 values, corresponding to making


# a right turn, no turn, and a left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------

# ----------------------------------------
# modify code below
# ----------------------------------------

def optimum_policy2D(grid, init, goal, cost, forward=forward, forward_name=forward_name, action=action, action_name=action_name, BLOCKED_VALUE=9999999):
    value = [[[BLOCKED_VALUE for col in range(len(grid[0]))] for row in range(len(grid))],
             [[BLOCKED_VALUE for col in range(len(grid[0]))] for row in range(len(grid))],
             [[BLOCKED_VALUE for col in range(len(grid[0]))] for row in range(len(grid))],
             [[BLOCKED_VALUE for col in range(len(grid[0]))] for row in range(len(grid))]]

    closed = [[[0 for col in range(len(grid[0]))] for row in range(len(grid))],
              [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
              [[0 for col in range(len(grid[0]))] for row in range(len(grid))],
              [[0 for col in range(len(grid[0]))] for row in range(len(grid))]]
    currentstate = [[0, goal[0], goal[1]], [1, goal[0], goal[1]], [2, goal[0], goal[1]], [3, goal[0], goal[1]]]

    # We don't care for the final orientation
    value[0][goal[0]][goal[1]] = 0
    value[1][goal[0]][goal[1]] = 0
    value[2][goal[0]][goal[1]] = 0
    value[3][goal[0]][goal[1]] = 0

    policy = [[[' ' for col in range(len(grid[0]))] for row in range(len(grid))],
              [[' ' for col in range(len(grid[0]))] for row in range(len(grid))],
              [[' ' for col in range(len(grid[0]))] for row in range(len(grid))],
              [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]]
    policy2D = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    value2D = [[BLOCKED_VALUE for col in range(len(grid[0]))] for row in range(len(grid))]
    orientations2D = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    policy[0][goal[0]][goal[1]] = '*'
    policy[1][goal[0]][goal[1]] = '*'
    policy[2][goal[0]][goal[1]] = '*'
    policy[3][goal[0]][goal[1]] = '*'
    policy2D[goal[0]][goal[1]] = '*'

    change = True
    while change:
        change = False
        nextstate = []
        for ctr in range(len(currentstate)):
            dir1 = currentstate[ctr][0]
            x1 = currentstate[ctr][1]
            y1 = currentstate[ctr][2]
            for actionctr in range(len(action_name)):
                dir2 = (dir1 - action[actionctr])%4
                x2 = x1 - forward[dir1][0]
                y2 = y1 - forward[dir1][1]
                if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                    newcost = value[dir1][x1][y1] + cost[actionctr]
                    if newcost < value[dir2][x2][y2]:
                        value[dir2][x2][y2] = newcost
                        nextstate.append([dir2, x2, y2])
                        policy[dir2][x2][y2] = action_name[actionctr]
                    change = True
        currentstate = nextstate[:]

    # Now use INIT to compute policy2D
    currentstate = init[:]
    change = True
    while (currentstate[1] != goal[0] or currentstate[2] != goal[1]):
        dir1 = currentstate[0]
        x1 = currentstate[1]
        y1 = currentstate[2]

        policy2D[x1][y1] = policy[dir1][x1][y1]
        value2D[x1][y1] = value[dir1][x1][y1]

        thisAction = action_name.index(policy2D[x1][y1]) - 1
        dir2 = (dir1 + thisAction)%4
        x2 = x1 + forward[dir2][0]
        y2 = y1 + forward[dir2][1]

        currentstate = [dir2, x2, y2]

    return policy2D


policy2D = optimum_policy2D(grid, init, goal, cost)

print('OPTIMUM POLICY')
for row in policy2D:
    print(row)
print('')