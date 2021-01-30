source('online_forest.R')
source('prepare_data.R')
library(jsonlite)

# online_filename = 'data/linear_data_online.csv'
# offline_filename = 'data/linear_data_offline.csv'
# online_filename = 'data/sigmoid_data_online.csv'
# offline_filename = 'data/sigmoid_data_offline.csv'
online_filename = 'data/compare_data_online.csv'
offline_filename = 'data/compare_data_offline.csv'

online_data <- read.csv(file=online_filename, header=TRUE, sep=',')
context_dim = 4

feed_sample <- function(t) {
	" feed one row of data from simulator
	  both the context and potential outcomes
	"
	return (online_data[t,])
}

get_colnames <- function(N_columns) {
	X_colnames = c()
	for (i in 0 : (N_columns-1)) {
		X_colnames = c(X_colnames, paste0('context',i))
	}
	X_colnames
}

load_offline_data <- function(filename, 
							  N_offline=NULL, 
							  context_names=NULL, 
							  treatment_name=NULL, 
							  outcome_name=NULL) {
	data <- read.csv(file=filename, header=TRUE, sep=',')

	print (paste0('the number of offline samples: ', N_offline))
	if ( is.null(N_offline) == FALSE ) {
		data <- data[1:N_offline, ]
	}
	
	X_colnames <- get_colnames(context_dim)

	if ( is.null(context_names) ) {
		X = data[, X_colnames]
	} else {
		X = data[, context_names]
	}
	
	if ( is.null(outcome_name) ) {
		Y = data$reward
	} else {
		Y = data[[outcome_name]]
	}
	
	if ( is.null(treatment_name) ) {
		W = data$action
	} else {
		W = data[[treatment_name]]
	}	

	ret <- list("W" = W, "Y" = Y, "X" = X)
	return (ret)
}

run_online_forest <- function(T, 
						  T_retrain,
						  N_offline=NULL, 
						  option='offline_online',
						  context_names=NULL, 
						  treatment_name=NULL, 
						  outcome_name=NULL,
						  num_actions=35,
						  choice_names=NULL, # if choice_names is not NULL, one can only choose from a subset of actions
						  alg_option=NULL) { 
	" run the algorithm, for T rounds
	  every T_retrain rounds, we update the structure of the forest
	  logging the regret in each round, logging the 
	"
	result_vec <- c()

	online_data <- online_data[sample(nrow(online_data)),] # shuffle the online data

	data <- load_offline_data(offline_filename, N_offline, 
		context_names, treatment_name, outcome_name)
	if (option != 'only_online') {
		forest <- new_forest(data, num_actions=num_actions)
	} else {
		forest <- NULL
	}
	t <- 1
	count_sample <- 1
	while (t <= T) {
		print (paste0('############# round ', t, ' #############') )
	
		sample <- feed_sample(count_sample)
		count_sample <- count_sample + 1

		if ( is.null(context_names) ) {
			X_colnames <- get_colnames(context_dim)
		} else {
			X_colnames <- context_names
		}
		context <- sample[, X_colnames]
		# print (context)

		if (!is.null(choice_names)) {
			choices <- sample[, choice_names]
			choices <- choices[!is.na(choices)] # remove all the NA
		} else {
			choices <- c(0, 1) # the default choices
		}

		if ( is.null(alg_option) ) { # default
			action <- draw_arm_epsilon_greedy_multiaction(forest, context, t+N_offline, choices, num_actions=num_actions)
		} else if ( alg_option == 'epsilon_greedy' ) { 
			action <- draw_arm_epsilon_greedy(forest, context, t+N_offline)			
		} else {
			print ('unsupported option!!!')
		}


		# action <- draw_arm_UCB(forest, context, 1) # based on each action
		#action <- draw_arm_by_treatment_effect(forest, context)		
		# action <- draw_arm_sampling(forest, context)

		if ( is.null(treatment_name) ) {
			reward <- as.double(sample[paste0('reward', action)])  # a.k.a. reward, concat the string
		} else {
			reward <- as.double(sample[[outcome_name]])
			# print (action)
			# print (sample[[treatment_name]])
			if (action != sample[[treatment_name]]) {
				### print ('actions do not match, skip this sample')
				next
			}
		}
		
		if (option != 'only_offline') {
			#forest <- update(forest, context, action, reward)
			data$X <- rbind(data$X, context)
			row.names(data$X) <- NULL
			data$W <- c(data$W, action)
			data$Y <- c(data$Y, reward)
		}		
		#print (data)

		print (paste0('///////// the chosen action is ', action, ' /////////'))

		if ( is.null(treatment_name) ) { # when we use synthetic data, we can get the regret
			# calculate the regret
			max_reward <- -Inf	
			for (action in 0:1) {
				tmp_reward <- as.double(sample[paste0('reward', action)])
				print (paste0('true reward for action', action, ': ', tmp_reward))
				if (tmp_reward > max_reward) {
					max_reward <- tmp_reward
				}
			}
			regret <- max_reward - reward
			print (paste0('regret: ', regret))
			result_vec <- c(result_vec, regret)
		} else {
			print (paste0('reward: ', reward))
			result_vec <- c(result_vec, reward)
		}
		t <- t+1 # update the time only after we receive a feedback from this round
		if (t %% T_retrain == 0 && option != 'only_offline') {
			forest <- new_forest(data, num_actions=num_actions)
		}
	}
	return (result_vec)
}

repeat_online_forest <- function(repeat_times, T, T_retrain, N_offline, option,
							 context_names = NULL,
							 treatment_name = NULL,
							 outcome_name = NULL,
							 choice_names = NULL,
							 output_file_prefix = "data/result_online_forest_",
							 is_append = TRUE,
							 num_actions=35,
							 alg_option=NULL) {
	" run the experiment for N times, compute the average cumulated regret
	  we have three options: 'only_offline', 'only_online', 'offline_online'
	"
	for (i in 1:repeat_times) {
		regret_vec <- run_online_forest(T, T_retrain, N_offline, option,
			context_names, treatment_name, outcome_name, choice_names, alg_option, num_actions=num_actions)
		cumulative_regret_vec <- c()
		cumulative_regret <- 0
		for (t in 1:length(regret_vec)) {
			cumulative_regret <- cumulative_regret + regret_vec[t]
			cumulative_regret_vec <- c(cumulative_regret_vec, cumulative_regret)
		}
		exportJSON <- toJSON(cumulative_regret_vec)
		write(exportJSON, paste0(output_file_prefix, option, '.json'), append=is_append)
	}	
}
