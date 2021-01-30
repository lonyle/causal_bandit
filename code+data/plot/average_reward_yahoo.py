from numpy import nan
from numpy import isnan

average_reward_vec = [
	nan, 
	nan, 
	nan, 
	nan, 
	nan, 
	nan, 
	nan, 
	nan, 
	nan, 
	0.029754525167369206, 
	nan, 
	0.03205765407554672, 
	nan, 
	0.03523035230352303, 
	0.013836477987421384, 
	0.033996474439687736, 
	0.017369093231162196, 
	0.01920678473434772, 
	0.02103439742637961, 
	0.03081232492997199, 
	0.04040907957096533, 
	0.045535714285714284, 
	0.027640671273445213, 
	0.03390651489464761, 
	0.0287458661918087, 
	0.032242185577159736, 
	0.03135108235879572, 
	0.02815115394369769, 
	0.03344067376764925, 
	0.03043367993913264, 
	0.014471057884231538, 
	nan, 
	nan, 
	0.012893982808022923, 
	nan]

pscore_correlation_action = []
for reward in average_reward_vec:
	if isnan(reward):
		correlation = 0
	elif reward > 0.035:
		correlation = 1
	else:
		correlation = -1
	pscore_correlation_action.append(correlation)