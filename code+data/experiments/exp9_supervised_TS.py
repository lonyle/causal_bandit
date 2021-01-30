# this implements thompson sampling with informed prior


import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 2,
	"model_name": 'xgboost:binary', # ['xgboost:binary', 'historic_average:binary', 'linear_regression:binary']
	"context_dim": 4,
	"synthetic": True,
	"function_form": 'sigmoid',
	"binary_outcome": True,
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"N_offline": 50,
	"algorithm": "thompson_sampling",
	"match_machine": "supervised_matching",
	#"match_machine": "IPS_weighting",
	"propensity_score_setting" : {
		"prob_vec": [0.25]*4,
		"propensity_score_vecs": [
			[0.2, 0.8],
			[0.4, 0.6],
			[0.6, 0.4],
			[0.8, 0.2]
		],
	},
	"reward_pscore_correlation": -1,
	"bias": [0, 0],
	"arm_bias_vec": [0, 0.5]
}

T = 100
repeat_times = 200

for option in ['offline_online', 'only_offline', 'only_online']:
	if params['match_machine'] == 'IPS_weighting':
		model_name = "IPSW_TS"
	else:
		model_name = params['model_name']
	exp_utils.output_algorithm(params, T, option, 'data/result_exp9_' +model_name+'_'+ option+'.json', repeat_times)
