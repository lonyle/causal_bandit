# UCB + inverse propensity weighting

import sys
sys.path.append('.')
import numpy as np

import exp_utils

context_dim = 8

params = {
	"N_arm": 2,
	"context_dim": context_dim,
	"synthetic": True,
	"function_form": 'linear',
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
	"reward_pscore_correlation": -0.2, # added on 2020-09-26, for the bias of pscore
	"bias": [0.0, 0],
	"arm_bias_vec": [0, 0.5],
	"unobserved_context_dims": np.random.choice(context_dim, 0, replace=False),
	"has_pscore": False
}

T = 100
repeat_times=1000


def run(num_unobserved_context_dim):
	print ('num_unobserved_context_dim:', num_unobserved_context_dim)
	params['unobserved_context_dims'] = np.random.choice(context_dim, num_unobserved_context_dim, replace=False)

	#for option in ['offline_online', 'only_offline', 'only_online']:
	#for option in ['only_online']:
	for option in ['offline_online']:
		exp_utils.output_algorithm(params, T, option, \
			'data/result_exp14_'+option+'_hidden_dim'+str(num_unobserved_context_dim)+'.json', \
			repeat_times)

if __name__ == '__main__':
	for n in range(context_dim+1):
		run(n)