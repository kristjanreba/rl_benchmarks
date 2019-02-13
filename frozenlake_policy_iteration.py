import numpy as np
import gym

def run_episode(env, policy, max_steps=1000, render=False):
    observation = env.reset()
    total_reward = 0
    observation = env.reset()
    for i in range(max_steps):
        if render:
            env.render()
        action = policy[observation]
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, num_episodes=100):
    total_rewards = 0.0
    for e in range(num_episodes):
        total_rewards += run_episode(env, policy)
    return total_rewards / num_episodes

def compute_V(env, policy):
    for s in range(env.env.nS):
        V[s] = np.sum([p*(r+gamma*Vprev[s_]) for (p,s_,r,_) in env.env.P[s][a]])
    return V

def policy_iteration():
    for s in range(env.env.nS)


def main(env_name, num_episodes=100, max_iter=10000, gamma=1.0):
    #env_name = 'FrozenLake-v0'
    env = gym.make(env_name)
    env.seed(0)

    V_optimal = policy_iteration(env, max_iter, gamma)
    policy = extract_policy(env, V_optimal, gamma)
    policy_score = evaluate_policy(env, policy, num_episodes)

    print('Policy average score: ', policy_score)

if __name__ == "__main__":
    main('FrozenLake-v0')
    '''
    FrozenLake-v0
    FrozenLake8x8-v0
    '''
