# linear regression + linUCB (under sigmoid enviroment)

import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 2,
	"context_dim": 10,
	"synthetic": True,
	"function_form": 'sigmoid',
	"N_offline": 20,
	"algorithm": "LinUCB",
	"match_machine": "linear_matching",
	"dimension": 5,
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"propensity_score_setting" : {
		"prob_vec": [0.25]*4,
		"propensity_score_vecs": [
			[0.2, 0.8],
			[0.4, 0.6],
			[0.6, 0.4],
			[0.8, 0.2]
		]
	},
	"arm_bias_vec": [0, 0.5]
}

T = 500
repeat_times=200

for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_exp5_'+option+'.json', repeat_times)