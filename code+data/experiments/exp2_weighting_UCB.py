# UCB + inverse propensity weighting

import sys
sys.path.append('.')
import numpy as np

import exp_utils

N_arm = 3

params = {
	"N_arm": N_arm,
	"context_dim": N_arm * 2,
	"synthetic": True,
	"function_form": 'linear',
	#'binary_outcome': True,
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
	"bias": [0] * N_arm,
	"arm_bias_vec": np.arange(N_arm) * 0.5
	#"arm_bias_vec": [0, 0.5]
}

T = 1000
repeat_times=500

for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_offline']:
#for option in ['offline_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_exp2_'+option+'.json', repeat_times)






