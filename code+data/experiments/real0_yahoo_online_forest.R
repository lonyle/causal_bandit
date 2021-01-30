source('run_online_forest.R')

repeat_times <- 5
T <- 3500
T_retrain <- 50
N_offline <- 500

is_append <- TRUE # for debugging, set it to False

online_filename = "data/_yahoo_reindex_exp_80000.csv"
offline_filename = "data/_yahoo_reindex_obs_20000.csv"
online_data <- read.csv(file=online_filename, header=TRUE, sep=',')

context_names <- c('context1', 'context2', 'context3', 'context4', 'context5', 'context6')
treatment_name <- "drawn_article"
outcome_name <- "reward"
choice_names <- c("article1", "article2", "article3", "article4", "article5", 
		"article6", "article7", "article8", "article9", "article10", 
		"article11", "article12", "article13", "article14", "article15", 
		"article16", "article17", "article18", "article19", "article20", "article21"
)

output_file_prefix = 'data/result_real0_yahoo3500_'

args = commandArgs(trailingOnly=TRUE)
option <- args[1]

#for (option in c('offline_online', 'only_online', 'only_offline')) {
#for (option in c('only_offline')) {
#for (option in c('offline_online')) {
#for (option in c('only_online')) {
repeat_online_forest(repeat_times, T, T_retrain, N_offline, option,
		context_names, treatment_name, outcome_name, choice_names, 
		output_file_prefix, is_append=is_append)
#}
