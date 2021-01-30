# this uses the supervised learning as the offline evaluator
# and the UCB as the online bandit oracle

# this time, we do experiment on Yahoo's dataset

import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 35,
	"N_offline": 10000,
	"model_name": 'xgboost',
	"synthetic": False,
	"obs_data_filename": 'data/_yahoo_pscore_from_20000_correlation.csv',
	"exp_data_filename": "data/_yahoo_reindex_exp_80000.csv",
	"context_names": ['context1', 'context2', 'context3', 'context4', 'context5', 'context6'], # user features
	"choice_names": [
		"article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"
	],
	"treatment_name": 'drawn_article',
	"outcome_name": "reward",
	"function_form": 'linear',
	"algorithm": "UCB",
	"UCB_beta": 0.05,
	"match_machine": "supervised_matching",
	"bias": [0] * 35
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
T = 200000
repeat_times = 10
is_append = True


#for option in ['offline_online', 'only_offline', 'only_online']:
for option in ['offline_online']:
	exp_utils.output_algorithm(params, T, option, \
		'data/result_real8_tmp_yahoo_' + model_name + '_' + option+'.json', \
		repeat_times, is_append=is_append)
