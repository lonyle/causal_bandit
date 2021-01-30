import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 2,
	'N_offline': 20,
	"synthetic": False,
	"obs_data_filename": "data/lalonde_ps.csv",
	"exp_data_filename": "data/lalonde_ps_unconfounded.csv",
	"context_names": ['age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75'],
	"treatment_name": "treat",
	"outcome_name": "re78",
	"algorithm": "LinUCB",
	"match_machine": "linear_matching",
	"dimension": 9
}

T = 50
repeat_times=20

for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_real4_'+option+'.json', repeat_times)