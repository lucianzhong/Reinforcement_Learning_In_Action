Reinforcement_Learning
=====

Reference:
------

 https://blog.csdn.net/u010223750/article/details/78879047

 https://github.com/dennybritz/reinforcement-learning



 

Model-Based RL: Policy and Value Iteration using Dynamic Programming
	
	Learning Goals:
	Understand the difference between Policy Evaluation and Policy Improvement and how these processes interact
	Understand the Policy Iteration Algorithm
	Understand the Value Iteration Algorithm
	Understand the Limitations of Dynamic Programming Approaches

	Summary
	1. Dynamic Programming (DP) methods assume that we have a perfect model of the environment's Markov Decision Process (MDP). That's usually not the 		case in practice, but it's important to study DP anyway
	2. Policy Evaluation: Calculates the state-value function V(s) for a given policy. In DP this is done using a "full backup". At each state, we look 	ahead one step at each possible action and next state. We can only do this because we have a perfect model of the environment
	3. Full backups are basically the Bellman equations turned into updates
	4. Policy Improvement: Given the correct state-value function for a policy we can act greedily with respect to it (i.e. pick the best action at each 	state). Then we are guaranteed to improve the policy or keep it fixed if it's already optimal
	5. Policy Iteration: Iteratively perform Policy Evaluation and Policy Improvement until we reach the optimal policy
	6. Value Iteration: Instead of doing multiple steps of Policy Evaluation to find the "correct" V(s) we only do a single step and improve the policy 	immediately. In practice, this converges faster
	7. DP methods bootstrap: They update estimates based on other estimates (one step ahead)

	策略迭代有一个缺点，就是每一步都要进行策略评估，当状态空间很大的时候是非常耗费时间的。值迭代是直接将贝尔曼最优化方程拿来迭代计算的，这一点是不同于策略迭代的.所以值迭代会直接收敛到最优值，从而我们就可以得到最优策略，因为它就是一个贪婪的选择。再反过去看一下策略迭代的过程，策略评估过程是应用贝尔曼方程来计算当前最优策略下的值函数，接着进行策略提升，即在每个状态都选择一个最优动作来最大化值函数，以改进策略。但是想一下，在策略评估过程我们一定要等到它收敛到准确的值函数吗？答案是不一定，我们可以设定一个误差，中断这个过程，用一个近似的值函数用以策略提升（格子世界的例子中就可以看出，在迭代到第三步以后，其实最优策略就已经确定了），而我们提出这个方法的时候并不是这么做的，而是等到策略评价过程收敛，这是一个极端的选择，相当于在迭代贝尔曼最优化方程！所以，换句话说，值迭代其实可以看成是策略迭代一个极端情况。
	一般来说，策略迭代的收敛速度更快一些，在状态空间较小时，最好选用策略迭代方法。当状态空间较大时，值迭代的计算量更小一些


Model-Free Prediction & Control with Monte Carlo (MC):
	基于模型的强化学习方法，对于很多现实问题，其实环境的state和状态转移概率是未知的，因此在计算Value的时候不能按照基于模型的方法进行全概率展开，这也是免模型学习的难点所在。很自然的对于很多数学问题，如果不能直接求解，采样的方法是个替代的方法，比如重要性采样.
	在蒙特卡洛(MC)的强化学习中，MC并不是特指某个具体的方法，只是单纯指的是基于随机采样的方法进行计算学习。本节主要讲的是通过MC计算Value的方法
	对于某种策略π，我们从起始状态s0出发，根据该策略获取状态的轨迹：s0,a0,r1,s1,a1,r2...,sT−1,aT−1,rT,sT

	在MC prediction中，计算Value函数有两种方法，第一种是first-visit,第二种是evert-visit，其中first-visit在计算Value(st)Value(st)值的时候是对于在每个episode中，选取该episode第一次出现状态stst以后的序列reward值来计算value(st)value(st),然后对所有的episode中的value(st)value(st)按照出现次数取平均值，就可以得到value(st)value(st),

	MC prediction，主要介绍的是如何利用采样轨迹的方法计算Value函数，但是在强化学习中，我们主要想学习的是Q函数，也就是计算出每个state对应的action以及其reward值
	这一部分将会介绍基于 greedyϵ−greedy方法，所谓ϵ−greedyϵ−greedy方法，就是对于当前策略，我们以1−epsilon1−epsilon的概率选择当前策略所要执行的动作A，以ϵϵ的概率随机执行其他的动作
