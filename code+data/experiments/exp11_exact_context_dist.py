# to see the improvement of performance when we use the exact context distribution

import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 3,
	"context_dim": 6,
	"synthetic": True,
	"function_form": 'linear',
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"N_offline": 10,
	"algorithm": "UCB",
	"match_machine": "IPS_weighting",
	"propensity_score_setting" : {
		"prob_vec": [0.25]*4,
		"propensity_score_vecs": [
			[0.2, 0.8],
			[0.4, 0.6],
			[0.6, 0.4],
			[0.8, 0.2]
		]
	},
	"reward_pscore_correlation": -1, # added on 2020-09-26, for the bias of pscore
	"bias": [0]*3,
	"arm_bias_vec": [0, 0.5, 1],
	"true_context_distribution": True
}

T = 2000
repeat_times = 2000

option = 'offline_online'

for true_context_distribution in [False, True]:
	params["true_context_distribution"] = true_context_distribution
	exp_utils.output_algorithm(params, T, option, \
		'data/result_exp11_'+option+'_'+str(true_context_distribution)+'_'+str(params['N_offline'])+'.json', repeat_times)