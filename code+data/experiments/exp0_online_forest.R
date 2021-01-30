source('run_online_forest.R')
library(rjson)

repeat_times <- 1
T <- 200
T_retrain <- 10
N_offline <- 100

is_append <- TRUE# for debug, set it to False

for (option in c('offline_online', 'only_online', 'only_offline') ) {
	repeat_online_forest(repeat_times, T, T_retrain, N_offline, option, 
						 context_names = NULL,
						 treatment_name = NULL,
						 outcome_name = NULL,
						 choice_names = NULL,
						 num_actions=2,
						 output_file_prefix = "data/result_online_forest_",
						 is_append=is_append)
}
