# Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from tqdm import *
from lake_envs import *
import matplotlib as mlp

mlp.use('tkagg')
import matplotlib.pyplot as plt


def epsilon_greedy_policies(e,Q:np.ndarray,s):
    if np.random.random()<e:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[s])

def do_nothing(Q):
    pass

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, learning_rate=0.1, e=0.8, decay_rate=0.99,after_one_episode_fn=do_nothing):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.
    
    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int 
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)
    
    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state, action values
    """
    Q = np.zeros((env.nS, env.nA))
    for i in tqdm(range(num_episodes)):
        s = env.reset()
        done = False
        while not done:
            action=epsilon_greedy_policies(e,Q,s)
            next_state, reward, done, _ = env.step(action)
            Q[s][action] += learning_rate * (reward + gamma * np.max(Q[next_state]) - Q[s][action])
            s = next_state
        if i % 10 == 0:
            e *= decay_rate
        after_one_episode_fn(Q)
    return Q


def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99,after_one_episode=do_nothing):
    """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
    Update Q at the end of every episode.
    
    Parameters
    ----------
    env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
    num_episodes: int 
    Number of episodes of training.
    gamma: float
    Discount factor. Number in range [0, 1)
    learning_rate: float
    Learning rate. Number in range [0, 1)
    e: float
    Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)
    
    Returns
    -------
    np.array
    An array of shape [env.nS x env.nA] representing state-action values
    """
    Q = np.zeros((env.nS, env.nA))
    for i in tqdm(range(num_episodes)):
        s = env.reset()
        done = False
        while not done:
            action=epsilon_greedy_policies(e,Q,s)
            next_state, reward, done, _ = env.step(action)
            next_action=epsilon_greedy_policies(e,Q,s)
            Q[s][action] += lr * (reward + gamma * Q[next_state][next_action] - Q[s][action])
            s = next_state
            a = next_action
        if i % 10 == 0:
            e *= decay_rate
        after_one_episode(Q)
    return Q


def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!
    
    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    step=0
    while not done and step<500:
        # env.render()
        # time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        step+=1
    #print("Episode reward: %f" % episode_reward)
    return episode_reward


# Feel free to run your own debug code in main!
def main():
    #env = gym.make("Deterministic-4x4-FrozenLake-v0")
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    score_Q_learning = []
    score_SARSA_learning = []
    num_of_episodes=10000

    def insert_score_list(env,list):
        def insert(Q):
            score=render_single_Q(env,Q)
            list.append(score)
        return insert

    Q2 = learn_Q_SARSA(env,num_episodes=num_of_episodes,after_one_episode=insert_score_list(env,score_SARSA_learning))
    Q1 = learn_Q_QLearning(env,num_episodes=num_of_episodes,after_one_episode_fn=insert_score_list(env,score_Q_learning))
    avg1=[]
    avg2=[]
    sum1=0
    sum2=0
    for i in (range(num_of_episodes)):
        sum1+=score_Q_learning[i]
        sum2+=score_SARSA_learning[i]
        avg1.append(sum1/(1+i))
        avg2.append(sum2/(1+i))
    plt.plot(np.arange(num_of_episodes), np.array(avg1))
    plt.plot(np.arange(num_of_episodes), np.array(avg2))
    plt.title('The running average score of the Q-learning agent')
    plt.xlabel('traning episodes')
    plt.ylabel('score')
    plt.legend(['q-learning', 'sarsa'], loc='upper right')
    plt.savefig('q_vs_sarsa_on_stochastic_env.png')
    #plt.savefig('q_vs_sarsa_on_deterministic_env.png')
    plt.show()


if __name__ == '__main__':
    main()
