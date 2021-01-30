# we want to compare with the version that uses all logged data in a batch

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
	#"N_offline": 1000,
	"N_offline": 100,
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
	"batch_mode": True
}

T = 1000
repeat_times = 500

# for T in range(600, 1001, 100):
# 	print ('T:', T)
for option in ['offline_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_exp13_'+option+'.json', repeat_times)
