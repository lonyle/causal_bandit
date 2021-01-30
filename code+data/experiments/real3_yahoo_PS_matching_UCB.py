'''
	run the IPSW+UCB algorithm on the yahoo dataset
'''

import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 35,
	"N_offline": 1000,
	"synthetic": False,
	"obs_data_filename": "data/_yahoo_pscore_from_20000.csv",
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
	"match_machine": "PS_matching",
	"N_type": 10,
	'bias': [0] * 35
}

T = 1000
repeat_times = 100

for option in ['offline_online', 'only_offline', 'only_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_real3_yahoo_'+option+'.json', repeat_times)