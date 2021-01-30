# this implements thompson sampling with informed prior


import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 3,
	"model_name": 'xgboost', # ['xgboost', 'historic_average', 'linear_regression']
	"context_dim": 6,
	"synthetic": True,
	"function_form": 'linear',
	#"binary_outcome": True,
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"N_offline": 100,
	"algorithm": "thompson_sampling_gaussian",
	"TS_scale": 0.25,
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
	"bias": [0, 0, 0],
	"arm_bias_vec": [0, 0.5, 1]
}

################ this is a user-specified param ################
model_name = sys.argv[1] 
if model_name not in ['xgboost', 'linear_regression', 'historic_average', 'IPSW']:
	print ('wrong model name!')
	exit()

params['model_name'] = model_name

if model_name == 'IPSW':
	params['match_machine'] = 'IPS_weighting'
################################################################

T = 500
repeat_times = 1000

for option in ['offline_online']:#, 'only_offline', 'only_online']:
	if params['match_machine'] == 'IPS_weighting':
		model_name = "IPSW"
	else:
		model_name = params['model_name']
	exp_utils.output_algorithm(params, T, option, 'data/result_exp91_' +model_name+'_'+ option+'.json', repeat_times)
