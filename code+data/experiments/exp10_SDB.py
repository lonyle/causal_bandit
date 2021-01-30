# stochastic delayed bandit

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
	"N_offline": 100,
	"algorithm": "SDB",
	"match_machine": "SDB",
	"propensity_score_setting" : {
		"prob_vec": [0.25]*4,
		"propensity_score_vecs": [
			[0.2, 0.8],
			[0.4, 0.6],
			[0.6, 0.4],
			[0.8, 0.2]
		]
	},
	"reward_pscore_correlation": -1,
	"bias": [0] * 3,
	"arm_bias_vec": [0, 0.5, 1]
}

T = 500
repeat_times = 500

for option in ['only_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_exp10_'+option+'.json', repeat_times)