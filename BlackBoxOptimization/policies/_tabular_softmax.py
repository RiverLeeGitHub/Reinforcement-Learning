import numpy as np
from .skeleton import Policy
from typing import Union
import random

class TabularSoftmax(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numStates:int, numActions: int):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        self._theta = np.zeros((numStates, numActions))
        self.numStates = numStates
        self.numActions = numActions


    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        prob = self._theta[state]
        if action is None:
            return prob
        p = prob[action]
        return p

    def samplAction(self, state:int)->int: # sampling randomly
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        act = random.choices(np.range(state), self.getActionProbabilities(state))

        return act[0]

    def getActionProbabilities(self, state:int)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        etheta = np.exp(self._theta[state])
        sum = np.sum(etheta)
        dis = etheta/sum
        return dis
