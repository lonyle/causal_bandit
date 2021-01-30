# we have more than two actions

import sys
sys.path.append('.')
import numpy as np

import exp_utils

N_arm = 10

params = {
	"N_arm": N_arm,
	"context_dim": 2*N_arm,
	"synthetic": True,
	"function_form": 'linear',
	"context_names": None,
	"treatment_name": 'action',
	"outcome_name": 'reward',
	"N_offline": 100,
	"algorithm": "UCB",
	"match_machine": "IPS_weighting",
	"reward_pscore_correlation": -1,
	"bias": [0] * N_arm,
	"arm_bias_vec": np.arange(N_arm) * 0.1
}

T = 200
repeat_times = 200

for option in ['offline_online', 'only_offline', 'only_online']:
	exp_utils.output_algorithm(params, T, option, \
		'data/result_exp12_multiple_action_'+option+'_'+str(N_arm)+'.json', repeat_times)