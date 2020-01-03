# Reinforcement-Learning

This repo contains my course homeworks and projects of reinforcement learning.

## Black-Box Optimization
### Goal
Using Black-Box optimization methods (like CEM, FCHC, GA, etc.) to find the optimal policies for agent.


## Policy Improvement
### Goal
Provided a policy and its histories, the goal is to safely find better policies of the problem.
### Implementation
[Link of this section](https://github.com/RiverLeeGitHub/Reinforcement-Learning/blob/master/PolicyImprovement/readme.txt)

## Temporal Difference (TD) Learning 
### Introduction
TD learning is a policy evaluation algorithm that learns from experiences similar to Monte-Carlo algorithms. It chooses actions using π and sees what happens, which is called sampling, rather than requiring knowledge about P and R. However, like the dynamic programming methods, it produces estimates based on other estimates - it bootstraps. This enables the model perform its updates before the end of an episode.
### Goal
This implementation includes two TD learning algorithms: Sarsa and Q-Learning. 
- Sarsa uses TD to estimate q^π, and simultaneously change π to be (nearly) greedy with respect to q^π. It changes the Bellman equation into an update rule.
- Q-Learning changes the Bellman optimality equation into an update rule comparing to Sarsa.

Some of the introduction is quoted from Prof. Philips Tomas’s notes.
