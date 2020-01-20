## Black-box Optimization

Black-box optimization (BBO) algorithms are generic optimization algorithms. They are called black-box because they treat the objective funcition f as a black-box. The algorithms don't look inside of the function to leverage knowledge about its structure in order to speed up the optimization process. For RL, this means that BBO algorithms will not leverage the knowledge that the environment can be modeled as an MDP.

### Agents

The agents in the folder stand for the BBO algorithms that optimize the objective function. They are Cross-Entropy Method, First-Choice Hill-Climbing, and Genetic Algorithm.

### Environments

Environment of RL is what the agents interact with during the whole process. It usually tells agents what state they are in and how many rewards they can get after taking an action in the state. There's two environments that are tested in the code: CartPole and GridWorld.

### Policies

A Policy is a decision-making rule in reinforcement learning. An agent can use a policy to select actions to take in an environment. This folder contains a tabular softmax policy selector. For continuous state, we often use linear approximation to approximate the map between states and actions.







