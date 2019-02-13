import gym
import numpy as np

LEFT = 0
RIGHT = 1

def get_random_policy():
    W = np.random.rand(1,4)
    b = np.random.rand(1)
    return W, b

def select_action(env, policy, observation):
    W = policy[0]
    b = policy[1]
    if np.dot(W, observation) + b > 0 :
        return RIGHT
    else:
        return LEFT

def run_episode(env, policy, t_max=1000, render=False):
    observation = env.reset() # Restart the environment to start a new episode
    total_reward = 0
    for t in range(t_max):
        if render:
            env.render()
        action = select_action(env, policy, observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    print(env.action_space)
    print(env.observation_space)

    # Generate random policies
    n_policy = 500
    policy_list = [get_random_policy() for _ in range(n_policy)]

    # Evaluate each policy
    scores_list = [run_episode(env, p) for p in policy_list]

    # Select the best policy
    best_policy_ix = np.argmax(scores_list)
    best_policy = policy_list[best_policy_ix]
    print(best_policy)
    print(scores_list[best_policy_ix])

    run_episode(env, best_policy, render=True)
