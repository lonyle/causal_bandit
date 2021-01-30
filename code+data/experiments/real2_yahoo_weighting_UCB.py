'''
	run the IPSW+UCB algorithm on the yahoo dataset
'''

import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 35,
	"N_offline": 10000,
	"synthetic": False,
	"obs_data_filename": "data/_yahoo_pscore_from_80000_uniform.csv",
	#"obs_data_filename": "data/_yahoo_pscore_from_20000_correlation.csv",
	"exp_data_filename": "data/_yahoo_reindex_exp_80000.csv",
	"context_names": ['context1', 'context2', 'context3', 'context4', 'context5', 'context6'], # user features
	"choice_names": [
		"article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"
	],
	"treatment_name": "drawn_article",
	"outcome_name": "reward",
	"algorithm": "UCB",
	"UCB_beta": 0.05,
	"match_machine": "IPS_weighting",
	'bias': [0] * 35
}

T = 200000
repeat_times = 30
is_append = True

option = sys.argv[1]
# if option == 'only_offline':
# 	T = 100
# 	repeat_times = 700

#for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_online']:
exp_utils.output_algorithm(params, T, option, 'data/result_real2_tmp_yahoo_'+option+'.json', \
		repeat_times, is_append=is_append)