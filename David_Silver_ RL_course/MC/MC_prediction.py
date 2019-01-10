"""
21点游戏
这一小节打算[1]书中使用21点游戏为背景，对于两个人，游戏规则是这样：

有两种角色，庄家和玩家
扑克有1-10，J,Q,K，其中J,Q,K表示数值10，卡牌1（Ace）可以表示1也可以表示11
在最开始，玩家发两张牌，庄家发两张牌，其中庄家的牌一张是公开的，玩家两张牌都是不公开的，只有玩家自己才能看到。
玩家可以选择要牌，可以决定手中的1表示的是1还是表示11，如果玩家在开始的时候是一张1一种10(10,J,Q,K)那么表示是natural，此时如果庄家也是natural那么表示的是平局，反之则玩家胜，如果在玩家要牌的过程中手中牌的总数大于21，那么就爆了，玩家输，如果玩家没要爆并停止要牌，那么庄家开始要牌，庄家在点数小于17的时候必须要牌，如果超过了17那么就要停止要牌，庄家在要牌的过程中爆了则庄家输，如果庄家停牌并没有爆，那么这个时候庄家和玩家开牌，谁的点数更靠近21点便胜利，如果相等则平局。

游戏分析：

在最开始，如果玩家手中的点数小于11那么必然会要牌直到超过11点

每个人手中如果1可以在不爆的情况下可以表示为11，那么必然会当做11 
根据以上分析，那么在游戏过程中，玩家手中的总和应该在12-21之间，而庄家公开的牌是1-10,不管庄家的1表示的11还是1，而玩家手中的1可以表示1或者11，当玩家表示为1的时候表示是no usable，当玩家表示为11的时候表示为usable，那么显然所有的state状态又200个。
游戏建模假设：

牌是无限发的，因此玩家和庄家不可以通过桌面上公开的牌进行猜测剩下的牌的概率

"""

import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

sys.path.append("../lib/envs") 
from blackjack import BlackjackEnv
sys.path.append("../lib")
import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function for a given policy using sampling.    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """
    # Keeps track of sum and count of returns for each state to calculate an average. We could use an array to save all returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # The final value function
    V = defaultdict(float)
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # Generate an episode, An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()
        for t in range(100):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        # Find all states the we've visited in this episode, We convert each state to a tuple so that we can use it as a dict key
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            # Find the first occurance of the state in the episode
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            # Sum up all rewards since the first occurance
            G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            # Calculate average return for this state over all sampled episodes
            returns_sum[state] += G
            returns_count[state] += 1.0
            V[state] = returns_sum[state] / returns_count[state]
    return V

def sample_policy(observation):
    # A policy that sticks if the player score is >= 20 and hits otherwise.
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")
