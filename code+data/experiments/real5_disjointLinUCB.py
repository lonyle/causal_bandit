# this is to run yahoo news article recommendation
# which uses a disjoint linear bandit model

import sys
sys.path.append('.')

import exp_utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--repeat_times', type=int, default=50)
args = parser.parse_args()

params = {
	"N_arm": None, # not applicable
	"N_offline": 500,
	"synthetic": False,
	"obs_data_filename": "data/_yahoo_obs_20000.csv",
	"exp_data_filename": "data/_yahoo_exp_80000.csv",
	"articles_filename": 'data/_yahoo-webscope-articles.txt',
	"context_names": ['context1', 'context2', 'context3', 'context4', 'context5', 'context6'], # user features
	"choice_names": [
		"article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"
	],
	"treatment_name": "drawn_article",
	"outcome_name": "reward",
	"algorithm": "DisjointLinUCB",
	"match_machine": "disjoint_linear_matching",
	"dimension": 6
}

T = 4000
repeat_times = args.repeat_times


for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_online']:
	output_filename = 'data/result_real5_'+option+str(repeat_times)+'.json'
	exp_utils.output_algorithm(params, T, option, output_filename, repeat_times)