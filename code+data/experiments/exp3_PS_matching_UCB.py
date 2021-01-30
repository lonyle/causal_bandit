# UCB + propensity score matching

import sys
sys.path.append('.')
import numpy as np

import exp_utils

N_arm = 2

params = {
	"N_arm": N_arm,
	"context_dim": N_arm * 2,
	"synthetic": True,
	"function_form": 'linear',
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"N_offline": 100,
	"algorithm": "UCB",
	"match_machine": "PS_matching",
	"match_on_action": False,
	"N_type": 3,
	"propensity_score_setting" : {
		"prob_vec": [0.25]*4,
		"propensity_score_vecs": [
			[0.2, 0.8],
			[0.4, 0.6],
			[0.6, 0.4],
			[0.8, 0.2]
		],
		#"output_type": "all_dimension" # the propensity score field record the probability of all dimensions
	},
	"reward_pscore_correlation": -1, # added on 2020-09-26, for the bias of pscore
	"bias": [0] * N_arm,
	"arm_bias_vec": np.arange(N_arm) * 0.5
	#"arm_bias_vec": [0, 0.5]
}

if params['match_on_action'] == False:
	params['propensity_score_setting']['output_type'] = "all_dimension"

T = 1000
repeat_times=500

for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_exp3_'+option+'_'+str(N_arm)+'.json', repeat_times)


