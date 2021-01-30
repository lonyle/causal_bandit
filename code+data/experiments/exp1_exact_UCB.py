# input: a synthetic dataset generated by a linear function Y = f(X, a)
# call the exact_mathing + UCB algorithm

import sys
sys.path.append('.')
import numpy as np

import exp_utils

N_arm = 2

params = {
	"N_arm": N_arm,
	"context_dim": N_arm * 2,
	"synthetic": True,
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"function_form": 'linear',
	"N_offline": 100,
	"algorithm": "UCB",
	"match_machine": "exact",
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
	"bias": [0] * N_arm,
	"arm_bias_vec": np.arange(N_arm) * 0.5
}

T = 1000
repeat_times=500

for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_online']:
	if option == 'only_offline':
		params['batch_mode'] = True
	exp_utils.output_algorithm(params, T, option, 'data/result_exp1_'+option+'.json', repeat_times)