import numpy as np
from .bbo_agent import BBOAgent
import random
from typing import Callable

class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box
    optimization (BBO) algorithm.

    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of
                individuals in the population and M is the number of
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation

    """

    def __init__(self, populationSize: int, evaluationFunction: Callable,
                 initPopulationFunction: Callable, numElite: int = 2, numEpisodes: int = 20):#1,20
        self._name = "Genetic_Algorithm"

        # DONE
        self._initPopulationFunction = initPopulationFunction
        self._population = initPopulationFunction(
            populationSize)  # TODO: set this value to the most recently created generation
        self._evaluationFunction = evaluationFunction
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._populationSize = populationSize

        self._indexTruncation = min(2, len(self._population))#10
        self._learningParameter = 0.5#2.5

        self._bestParameter = None
        self._estimations = None
    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> np.ndarray:
        # DONE
        '''
        parameters should return the best parameters found overall, i.e., across multiple training steps. You will have to consider parameters when coding the reset method. In CEM, parameters should be initialized to the theta input.
        There is more than one acceptable way parameters can be initialized in the GA agent. For instance, upon initialization, a call to parameters could return None, since the agent has not trained yet and this is where the population is created.
        Upon initialization (or reset), parameters could instead return the best parameters found in the initial population (because there is enough information in the constructor to generate the initial population and find the best parameter). There will be a penalty if parameters is initialized (or reset) to a less intuitive value, i.e., a parameter that does not exist in the initial population. Note that initializing and resetting parameters for fchc is straightforward because the starting parameter is provided in the constructor.
        :return:
        '''
        # bestParameter = sorted(self._population, key=lambda x: x[1], reverse=True)
        # return np.array(bestParameter).flatten()
        return np.array(self._bestParameter)

    def _mutate(self, parent: np.ndarray) -> np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified.

        output:
            child -- a mutated copy of the parent
        """
        # # DONE
        # children = []
        # for i in range(len(parent)):
        #     child = parent[i] + self._learningParameter * np.random.normal(loc=0, scale=1, size=parent.shape)
        #     children.append(child)
        # return np.array(children)

        child = parent + self._learningParameter * np.random.normal(0, 1, parent.shape[0])
        return child

    def estimations(self):
        return self._estimations

    def plot(self, plt, trial, returns):
        plt.errorbar(x=trial,
                     y=[np.mean(returns)],
                     yerr=np.std(returns),
                     fmt='o')

    def train(self) -> np.ndarray:
        # DONE
        def _crossover(parent, numChlidren):
            children = []
            for i in range(numChlidren):
                couple = random.choices(parent, k=2)
                child = (couple[0] + couple[1]) / 2
                children.append(child)
            return np.array(children)
        #
        # def get_parents(indexTruncation, sorted_estimations):
        #     return sorted_estimations[:indexTruncation]
        #
        # def get_children(learningParameter, parents):
        #     numChildren = self._populationSize - self._numElite
        #     mutated = self._mutate(parents)
        #     crossovered = _crossover(mutated, numChildren)
        #     children = crossovered[:numChildren]
        #     return children

        estimations = []
        for k in range(self._populationSize):
            returns = self._evaluationFunction(self._population[k], self._numEpisodes)
            J_estimated = np.sum(returns)
            estimations.append((self._population[k], J_estimated))
        # print("estimation::",estimations)
        sorted_estimations = sorted(estimations, key=lambda x: x[1], reverse=True)
        # print("sorted estimation::", sorted_estimations)
        parents = [list(sorted_estimations[i][0]) for i in range(len(sorted_estimations))][:self._indexTruncation]
        self._bestParameter = parents[0]
        # print("parents::",parents)
        print("best:",sorted_estimations[0][1])

        numChildren = self._populationSize# - self._numElite

        children = []

        for i in range(numChildren):
            mutated = self._mutate(np.array(parents[i%len(parents)]))
            children.append(mutated)

        children = _crossover(children, numChildren)

        # print("children:::",(children))
        next_gen = parents+list(children)
        self._estimations = np.array(sorted_estimations)
        self._population = np.array(next_gen)
        return self._population

    def reset(self) -> None:
        # TODO
        self._population = self._initPopulationFunction(self._populationSize)  # TODO: set this value to the most recently created generation
