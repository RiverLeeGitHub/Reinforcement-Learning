#### Including Sarsa and Q-Learning methods.

* Sarsa is an on-policy algorithm. It uses the TD algorithm to estimate <img src="http://chart.googleapis.com/chart?cht=tx&chl= q^{\pi}" style="border:none;">, and we can then act <img src="http://chart.googleapis.com/chart?cht=tx&chl= {\pi}" style="border:none;">
greedily with respect to <img src="http://chart.googleapis.com/chart?cht=tx&chl= q^{\pi}" style="border:none;">
* Q-Learning changes the Bellman optimality equation into an update rule comparing to Sarsa.
