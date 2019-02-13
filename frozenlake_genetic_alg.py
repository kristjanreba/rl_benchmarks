import numpy as np
import gym
import time
import random

def run_episode(env, policy, t_max=100, render=False):
    observation = env.reset() # Restart the environment to start a new episode
    total_reward = 0
    for i in range(t_max):
        if render:
            env.render()
        action = policy[observation]
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def eval_policy(env, policy, n_episodes=100):
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / n_episodes

def get_random_policy():
    return np.random.choice(4, size=16)

def crossover(policy1, policy2):
    new_policy = policy1.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand > 0.5:
            new_policy[i] = policy2[i]
    return new_policy

def mutation(policy, p=0.05):
    new_policy = policy.copy()
    for i in range(16):
        rand = np.random.uniform()
        if rand < p:
            new_policy[i] = np.random.choice(4)
    return new_policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    # Generate random policies
    n_policy = 100
    n_steps = 20
    start = time.time()
    policy_list = [get_random_policy() for _ in range(n_policy)]

    for ix in range(n_steps):
        policy_scores = [eval_policy(env, p) for p in policy_list]
        print('Generation %d : max score = %0.2f' %(ix+1, max(policy_scores)))
        policy_ranks = list(reversed(np.argsort(policy_scores)))
        elite_set = [policy_list[x] for x in policy_ranks[:5]]
        select_probs = np.array(policy_scores) / np.sum(policy_scores)
        child_set = [
            crossover(
                policy_list[np.random.choice(range(n_policy), p=select_probs)],
                policy_list[np.random.choice(range(n_policy), p=select_probs)])
            for _ in range(n_policy - 5)
        ]
        mutated_list = [mutation(p) for p in child_set]
        policy_list = elite_set
        policy_list += mutated_list
    policy_score = [eval_policy(env, p) for p in policy_list]
    best_policy = policy_list[np.argmax(policy_score)]

    end = time.time()

    print("Best policy score = %0.2f. Time =  %4.4f s" %(np.max(policy_score), end-start))
    run_episode(env, best_policy, render=True)
