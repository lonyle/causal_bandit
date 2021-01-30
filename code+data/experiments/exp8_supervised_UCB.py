# this uses the supervised learning as the offline evaluator
# and the UCB as the online bandit oracle

import sys
sys.path.append('.')
import numpy as np

import exp_utils

N_arm = 3

params = {
	"N_arm": N_arm,
	"model_name": 'xgboost',
	"context_dim": N_arm * 2,
	"synthetic": True,
	"function_form": 'linear',
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"N_offline": 100,
	"algorithm": "UCB",
	"match_machine": "supervised_matching",
	"propensity_score_setting" : {
		"prob_vec": [0.25]*4,
		"propensity_score_vecs": [
			[0.2, 0.8],
			[0.4, 0.6],
			[0.6, 0.4],
			[0.8, 0.2]
		],
	},
	"reward_pscore_correlation": -1, # added on 2020-09-26, for the bias of pscore
	"bias": [0]*N_arm,
	"arm_bias_vec": np.arange(N_arm)*0.5
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


# default: T=1000, repeat_times=200 
T = 500
repeat_times = 500


for option in ['offline_online']:#, 'only_offline', 'only_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_exp8_' + model_name + '_' + option+'.json', repeat_times)
