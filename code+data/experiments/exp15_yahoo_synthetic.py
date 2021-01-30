# instead of using the slow yahoo's real data, we want to use a faster version

import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 21,
	"synthetic": True,
	"binary_outcome": True,
	"context_dim": 0,
	"function_form": 'constant', # plus a constant
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"algorithm": 'UCB',
	"UCB_beta": 0.05,
	"match_machine": "IPS_weighting",
	"propensity_score_setting": {
		"prob_vec": [1],
		"propensity_score_vecs": [
			[1/21] * 21
		],
	},
	"bias": [0] * 21,
	"arm_bias_vec": [
		0.029754525167369206, 
		0.03205765407554672, 
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
		0.012893982808022923
	],
	'N_offline': 10000
}

T = 100000
repeat_times = 5

option = sys.argv[1]
if option == 'only_offline':
	T = 100
	repeat_times = 100

exp_utils.output_algorithm(params, T, option, 'data/result_exp15_'+option+'.json', repeat_times)


