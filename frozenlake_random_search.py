import gym
import numpy as np
import time

def get_random_policy():
    W = np.random.choice(4, size=16)
    return W

def eval_policy(env, policy, n_episodes=100):
    total_rewards = 0.0
    for _ in range(n_episodes):
        total_rewards += run_episode(env, policy)
        return total_rewards / n_episodes

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

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')

    # Generate random policies
    n_policy = 2000
    start = time.time()
    policy_list = [get_random_policy() for _ in range(n_policy)]

    # Evaluate each policy
    scores_list = [eval_policy(env, p) for p in policy_list]
    end = time.time()

    print("time: %4.4f s" %(end-start))

    # Select the best policy
    best_policy_ix = np.argmax(scores_list)
    best_policy = policy_list[best_policy_ix]
    print(best_policy)
    print(scores_list[best_policy_ix])

    run_episode(env, best_policy, render=True)
