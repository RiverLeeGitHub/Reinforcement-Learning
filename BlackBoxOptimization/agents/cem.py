import numpy as np
from .bbo_agent import BBOAgent
import matplotlib.pyplot as plt

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta: np.ndarray, sigma: float, popSize: int, numElite: int, numEpisodes: int,
                 evaluationFunction: Callable, epsilon: float = 0.0001):

        self._name = "Cross_Entropy_Method"

        self._theta = theta  # TODO: set this value to the current mean parameter vector
        self._Sigma = sigma * np.eye(self._theta.shape[0])  # TODO: set this value to the current covariance matrix

        self._backup = [self._theta, self._Sigma]

        self._popSize = popSize
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._evaluationFunction = evaluationFunction
        self._epsilon = epsilon

        self._bestParameter = None
        self._estimations = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        """
        The best policy parameters the agent has found. This should be a 1D
        numpy array.
        """
        # DONE
        return self._bestParameter.flatten()

    def estimations(self):
        return self._estimations

    def train(self) -> np.ndarray:
        """
        Perform a single iteration of the BBO algorithm.
        :return: bestParameter
        """
        estimations = []  # stored (theta, J) of each policy
        for k in range(self._popSize):
            # print("populate:",k)
            policy = np.array(np.random.multivariate_normal(self._theta,self._Sigma))
            returns = self._evaluationFunction(policy, self._numEpisodes)
            J_estimated = np.sum(returns)
            estimations.append((policy, J_estimated))
        sorted_estimations = sorted(estimations, key=lambda x: x[1], reverse=True)

        # compute the average parameter of elite estimations
        theta_sum = np.zeros(self._theta.shape)
        for k in range(self._numElite):
            theta_sum += sorted_estimations[k][0]
        self._theta = theta_sum / self._numElite
        self._bestParameter = sorted_estimations[0][0]
        self._estimations = sorted_estimations

        # compute Sigma of elite estimations
        I = np.eye(self._theta.shape[0])
        summation = 0
        for k in range(self._numElite):
            tmp = sorted_estimations[k][0] - self._theta
            summation += np.dot(tmp[:,None],tmp[None,:])
        self._Sigma = np.array((self._epsilon * I + summation) / (self._epsilon + self._numElite))

        return self._theta

    def reset(self) -> None:
        [self._theta, self._Sigma] = self._backup
        self._populations = []
        pass
