import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy
    """

    def __init__(self, theta: np.ndarray, sigma: float, evaluationFunction: Callable, numEpisodes: int = 10):
        self._name = "First_Choice_Hill_Climbing"
        # DONE
        self._theta, self._sigma, self._evaluationFunction, self._numEpisodes = theta, sigma, evaluationFunction, numEpisodes

        self._Sigma = sigma * np.eye(self._theta.shape[0])
        self._backup = [self._theta, self._Sigma]
        self.J = -float('inf')#np.mean(self._evaluationFunction(self._theta, self._numEpisodes))
        self.J_prime = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        # DONE
        return self._theta.flatten()

    def plot(self, plt, trial, returns):
        plt.errorbar(x=trial,
                     y=[np.mean(returns)],
                     yerr=np.std(returns),
                     fmt='bo-')

    def train(self,trial=None,plt=None) -> np.ndarray:
        # DONE
        policy = np.array(np.random.multivariate_normal(self._theta, self._Sigma))
        returns = self._evaluationFunction(policy, self._numEpisodes)
        J_prime = np.sum(returns)
        self.J_prime = J_prime
        # if trial is not None:
        #     self.plot(plt=plt,
        #               trial=trial,
        #               returns=returns)

        if J_prime > self.J:
            self._theta = policy
            self.J = J_prime
        return self._theta

    def reset(self) -> None:
        # DONE
        [self._theta, self._Sigma] = self._backup
        self.J = self.evaluationFunction(self._theta, self.numEpisodes)
