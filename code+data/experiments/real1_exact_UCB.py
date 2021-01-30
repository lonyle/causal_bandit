import sys
sys.path.append('.')

import exp_utils

params = {
	"N_arm": 2,
	'N_offline': 20,
	"synthetic": False,
	"obs_data_filename": "data/lalonde_ps.csv",
	"exp_data_filename": "data/lalonde_ps_unconfounded.csv",
	#"context_names": ['age', 'educ', 'black', 'married'],
	"context_names": ['black', 'married'],
	"treatment_name": "treat",
	"outcome_name": "re78",
	"algorithm": "UCB",
	"match_machine": "exact"
}

T = 50
repeat_times=20

for option in ['offline_online', 'only_offline', 'only_online']:
#for option in ['only_online']:
	exp_utils.output_algorithm(params, T, option, 'data/result_real1_'+option+'.json', repeat_times)