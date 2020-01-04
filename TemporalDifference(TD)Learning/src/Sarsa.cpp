#include "stdafx.h"

using namespace std;

// This is the constructor. If you added member variables, be sure to initialize them here.
Sarsa::Sarsa(const int & stateDim, const int & numActions, const double & alpha, const double & gamma, const double & epsilon, const int & iOrder, const int & dOrder) : stateDim(stateDim), numActions(numActions), alpha(alpha), gamma(gamma) {
    fb.init(stateDim, iOrder, dOrder);
	numFeatures = fb.getNumOutputs();
	w.resize(numActions);
	for (int a = 0; a < numActions; a++)
		w[a] = vector<double>(numFeatures, 0.0);
	d1 = bernoulli_distribution(epsilon);
	d2 = uniform_int_distribution<int>(0, numActions - 1);
}

// This is the train function. While the contents will differ from QLearning, you might copy the general structure (if-statements checking that terms are initialized, compute TD-error, update weights, set cur <-- new (curState, curAction, curReward?)
void Sarsa::train(std::mt19937_64 & generator, const std::vector<double> & s, const int & a, double & r, const std::vector<double> & sPrime, const bool & sPrimeTerminal) {
    // If we haven't initialized phi, initialize it and set the flag for phiInit.
    if (!phiInit) {
        phiInit = true;
        aLast = a;// Store the s,a,r for the next call.
        sLast = s;
        rLast = r;
        return;// Don't do anything in the first call.
    }

    if (!sPrimeTerminal) {
        vector<double> phiLast = fb.basify(sLast);// Compute phi of the last step
        vector<double> phi = fb.basify(s);// Compute phi of the current step

        double q = dot(w[aLast], phiLast);// Compute q based on the last step
        double qPrime = dot(w[a], phi);// Compute qPrime based on the current step
        double TDError = rLast + gamma * qPrime - q;// Compute the last step TD Error

        for (int i = 0; i < w[aLast].size(); i++) {
            w[aLast][i] = w[aLast][i] + alpha * TDError * phiLast[i];// refresh the weights
        }

        aLast = a;// Store the s,a,r for the next call.
        sLast = s;
        rLast = r;
    }
    else{
        vector<double> phiLast = fb.basify(sLast);
        vector<double> phi = fb.basify(s);

        double q = dot(w[aLast], phiLast);
        double qPrime = 0;// Set the qPrime to zero because it is the terminal state.
        double TDError = rLast + gamma * qPrime - q;

        for (int i = 0; i < w[aLast].size(); i++) {
            w[aLast][i] = w[aLast][i] + alpha * TDError * phiLast[i];
        }
    }
}

// When a new episode starts, do you need to clear any of your variables, or set any of your flags to true/false?
void Sarsa::newEpisode(mt19937_64 & generator) {
    phiInit = false;// The initial value is false
}

// This is identicaly to the getAction function in QLearning. You shouldn't have to change this.
int Sarsa::getAction(const std::vector<double> & s, std::mt19937_64 & generator) {

        if (d1(generator)) { // Explore
            return d2(generator);
        }
        vector<double> features = fb.basify(s);
        vector<int> bestActions(1, 0);
        double bestActionValue = dot(w[0],features);
        for (int a = 1; a < numActions; a++) {
            double curActionValue = dot(w[a], features);
            if (curActionValue == bestActionValue)
                bestActions.push_back(a);
            else if (curActionValue > bestActionValue) {
                bestActionValue = curActionValue;
                bestActions.resize(1);
                bestActions[0] = a;
            }
        }
        if ((int)bestActions.size() == 1)
            return bestActions[0];
        return (uniform_int_distribution<int>(0, (int)bestActions.size() - 1))(generator);
}