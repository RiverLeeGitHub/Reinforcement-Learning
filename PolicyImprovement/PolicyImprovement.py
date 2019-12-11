import numpy as np
import math
import cma
from scipy.stats import t
import csv

class fourierBasis:
    def __init__(self, inputDimension, iOrder, dOrder):
        self.inputDimension = inputDimension
        iTerms = iOrder * inputDimension  # Number of independent terms
        dTerms = int(math.pow(dOrder + 1, inputDimension))  # Number of dependent terms
        oTerms = min(iOrder, dOrder) * inputDimension  # Overlap of iTerms and dTerms
        self.nTerms = iTerms + dTerms - oTerms
        counter = np.zeros(inputDimension)
        self.c = np.zeros((self.nTerms, inputDimension))
        termCount = 0
        while termCount < dTerms:
            self.c[termCount] = counter
            self.incrementCounter(counter, dOrder)
            termCount += 1

        for i in range(inputDimension):  # Add the independent terms
            j = dOrder + 1

            while j <= iOrder:
                self.c[termCount][i] = float(j)
                termCount += 1
                j += 1

    def getNumOutputs(self):
        return self.nTerms

    def incrementCounter(self, buff, maxDigit):
        for i in range(buff.shape[0]):
            buff[i] += 1
            if buff[i] <= maxDigit:
                break
            buff[i] = 0

    def basify(self, x):
        result = np.zeros(self.nTerms)
        for i in range(self.nTerms):
            result[i] = np.cos(math.pi * np.dot(self.c[i], x))
        return result

def dataLoader(path):
    Data = []
    with open(path) as f:
        tmp = [line.split() for line in f]        # create a list of lists
        for i, x in enumerate(tmp):              #print the list items
            Data.append(x[0].split(','))

    m = int(Data[0][0])
    A = int(Data[1][0])
    k = int(Data[2][0])
    theta_b = np.array(Data[3]).astype(np.float).reshape(2,2)
    n = int(Data[4][0])
    Episodes = []
    for i in range(n):
        Episodes.append(np.array(Data[i+5]).astype(np.float))
    Episodes = np.array(Episodes)
    FirstEp = np.array(Data[200005]).astype(np.float)
    print("Number of state features: %d\nNumber of discrete actions: %d\nFourier basis order: %d\nParameters of the policy:%s\nNumber of episodes:%d"%(m,A,k,np.array2string(theta_b),n))
    # print("First Episode: ",FirstEp)
    return m, A, k, theta_b, n, Episodes

def calculate_pi(s,a,fb,theta):
    phi = fb.basify(s)
    up = np.exp(phi.dot(theta[a]))
    down = np.sum(np.exp(phi.dot(theta.T)))
    return up/down

def calculate_PDIS_H(theta_e,theta_b,epi,fb):
    pie_pib = 1.0
    ret = 0.0
    for t in range(epi.shape[0]):
        if t%3==0:
            s = np.array([epi[t]])
            a = int(epi[t+1])
            r = epi[t+2]
            up = calculate_pi(s,a,fb,np.array(theta_e))
            down = calculate_pi(s,a,fb,theta_b)
            pie_pib = pie_pib/down*up
            ret+=pie_pib*r
    return ret

def calculate_PDIS_D(episodes,theta_e,theta_b,fb):
    ret = []
    for epi in episodes:
        ret.append(calculate_PDIS_H(theta_e,theta_b,epi,fb))

    return np.mean(np.array(ret))

def sigma_hat(episodes,theta_e,theta_b,fb):
    return np.std([calculate_PDIS_H(theta_e,theta_b,epi,fb) for epi in episodes])


if __name__ == '__main__':

    '''load data'''
    m, A, k, theta_b, n, Episodes = dataLoader(path="data.csv")

    '''split dataset'''
    D_c = Episodes[:150000]
    D_s = Episodes[150000:]
    print("DATA SPLIT: ", len(D_c), len(D_s))

    delta = 0.1

    # calculate pi_b
    pi_b = np.zeros((m,A))

    # define function to calculate pi_b
    fb = fourierBasis(m, k, k)

    # start policy improvement
    count = 1
    lowerBound = 10 # c
    while count <= 100:
        print("COUNT===========================================",count)

        '''find optimal policies using CMA-ES'''
        es = cma.CMAEvolutionStrategy(4 * [0], 1)
        iter = 0
        while (not es.stop()) and (iter<=8):
            iter += 1
            solutions = es.ask()# ask for n policies
            fitness_list = []
            for x in solutions:
                theta_e = np.array(x).reshape(2,2)
                fitness = calculate_PDIS_D(D_c,theta_e,theta_b,fb)
                print("CANDIDATE SCORE========", fitness)
                fitness_list.append(-fitness)
            es.tell(solutions,fitness_list)
            es.logger.add()  # write data to disc to be plotted
            es.disp()
        es.result_pretty()
        cma.plot()


        theta_e_list = es.ask(8) # ask 8 optimal policies to do the safety test

        for theta_e in theta_e_list:
            print("CANDIDATE OF THETA_E===============================",theta_e)

            '''candidate test'''
            evaluation = calculate_PDIS_D(D_c, np.array(theta_e).reshape(2,2), theta_b, fb) - 2 * sigma_hat(D_c, np.array(theta_e).reshape(2,2), theta_b, fb) / (
                        len(D_s) ** 0.5) * (t.ppf(1 - delta, len(D_s) - 1))
            if evaluation < lowerBound:
                print("IGNORE THIS OPTIMAL POLICY FOR NOT PASSING CANDIDATE TEST")
                continue
            else:

                '''safety test'''
                def safetyEvaluation(D_s, theta_e, theta_b, fb):
                    evaluation = calculate_PDIS_D(D_s, theta_e, theta_b, fb) - sigma_hat(D_s,theta_e,theta_b,fb)/(len(D_s)**0.5)*(t.ppf(1-delta,len(D_s)-1))
                    print("SAFETY EVALUATION=========", evaluation)
                    if evaluation >= lowerBound:
                        return True
                    else:
                        return False

                if safetyEvaluation(D_s, np.array(theta_e).reshape(2,2), theta_b, fb):

                    with open(str(count) + '.csv', 'w') as f:
                        writer = csv.writer(f)
                        writer.writerow([str(theta_e[i]) for i in range(len(theta_e))])
                        print("SAVED")
                    count += 1
                else:
                    print("NSF")
