"""
A bare bones examples of optimizing a black-box function (f) using
Natural Evolution Strategies (NES), where the parameter distribution is a
gaussian of fixed standard deviation.
"""

import numpy as np

np.random.seed(0)

def f(w):
    # ... 1) create a neural network with weights w
    # ... 2) run the neural network on the environment for some time
    # ... 3) sum up and return the total reward

    # optimizing quadratic function
    reward = -np.sum(np.square(solution - w))
    return reward

# hyperparameters
npop = 100 # population size
sigma = 0.3 # noise standard deviation
alpha = 0.007 # learning rate

# solution
solution = np.array([-0.5, 0.2, -0.3])
w = np.random.randn(3)

for i in range(10000):
    # print current fitness of the most likely parameter setting
    if i % 100 == 0:
        print('iter %d, w: %s, solution: %s, reward: %f' % (i, str(w), str(solution), f(w)))
    N = np.random.randn(npop,3)
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma * N[j] # jitter is using gaussian
        R[j] = f(w_try) # evaluate the jittering version

    # stadardise the rewards to have gaussian distribution
    A = (R - np.mean(R) / np.std(R))

    w = w + alpha / (npop * sigma) * np.dot(N.T, A)
