source('run_online_forest.R')
#library(rjson)

repeat_times <- 1
T <- 500
T_retrain <- 50
N_offline <- 200

online_filename = "data/lalonde_ps_unconfounded.csv"
offline_filename = "data/lalonde_ps.csv"
online_data <- read.csv(file=online_filename, header=TRUE, sep=',')

context_names <- c('age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75')
treatment_name <- 'treat'
outcome_name <- 're78'

for (option in c('offline_online', 'only_online', 'only_offline') ) {
	repeat_online_forest(repeat_times, T, T_retrain, N_offline, option,
		context_names, treatment_name, outcome_name)
}