library(grf)

draw_arm_epsilon_greedy_multiaction <- function(forest, 
											context, 
											t, 
											choices, 
											num_actions=35,
											stored_estimations=NULL) { # we can store the estimated values
	if (is.null(forest)) {
		return (sample( choices, size=1 ))
	}
	max_reward <- -Inf
	max_arm_idx <- NULL
	epsilon <- t**(-0.5) # in fact, epsilon is a tunning parameter

	# print (choices)

	random_number <- runif(1)
	if (random_number < epsilon) {
		return (sample(choices, size=1))
	}

	for (action in choices) {
		# print (paste0('point estimate for action ', action))	
		result = predict_action.causal_forest(forest,
											  context, 
											  action,
											  num_actions,
											  estimate.variance=FALSE)
		estimated_reward = result$predictions
		# print (paste0('point estimate for action ', action, ': ', estimated_reward))
		if (is.nan(estimated_reward)) {
			next
		}
		if (estimated_reward > max_reward) {
			max_reward <- estimated_reward
			max_arm_idx <- action
		}
	}
	return (max_arm_idx)
}

draw_arm_epsilon_greedy <- function(forest, context, t) {
	" 
		the epsilon-greedy strategy
	"
	if (is.null(forest)) {
		return (sample(c(0,1), size=1))
	}
	result = predict(forest, context, estimate.variance=FALSE)
	point_estimate = result$predictions
	epsilon <- t**(-0.5)	
	random_number <- runif(1)
	# with probability 1-epsilon
	if (random_number > epsilon) {
		if (point_estimate > 0) {
			return (1)
		} else {
			return (0)
		}
	} else{ # with probability epsilon
		return (sample(c(0,1), size=1))
	}
}

draw_arm_by_treatment_effect <- function(forest, context) {
	" 
	  the least modification of the grf package
	  estimate the treatment effect (with the W.hat, Y.hat) and the confidence interval
	  draw the arm probablistically with the probability corresponding to the confidence interval
	"
	if (is.null(forest)) {
		return (sample(c(0,1), size=1))
	}
	result = predict(forest, context, estimate.variance=TRUE)
	sigma.hat = sqrt(result$variance.estimate)
	point_estimate = result$predictions
	if ( is.na(sigma.hat) ) {
		return (sample(c(0,1), size=1))
	}
	random_estimate = rnorm(1, point_estimate, sigma.hat)
	if (random_estimate > 0) {
		return (1)
	} else {
		return (0)
	}
}

draw_arm_UCB <- function(forest, context, eta=1.96) {
	" draw the arm according to the forest and the context, according to predictions on each action
	  currently, we only support binary actions (can be extendted to multiple actions)
	"
	if (is.null(forest)) {
		return (sample(c(0,1), size=1))
	}
	max_ucb <- -Inf
	max_arm_idx <- NULL
	for (action in 0:1) {
		ucb <- confidence_bound(forest, context, action, eta)$ucb
		print (paste0('ucb for action', action, ': ', ucb))
		if (ucb > max_ucb) {
			max_ucb <- ucb
			max_arm_idx <- action
		}
	}	
	return (max_arm_idx)
}

draw_arm_sampling <- function(forest, context) {
	if (is.null(forest)) {
		return (sample(c(0,1), size=1))
	}
	max_reward <- -Inf
	max_arm_idx <- NULL
	for (action in 0:1) {
		result = predict_action.causal_forest(forest, context, action, estimate.variance=TRUE)
		if ( is.na(sqrt(result$variance.estimates)) ) {
			random_reward = result$predictions
		}
		random_reward = rnorm(1, result$predictions, sqrt(result$variance.estimates))
		if (random_reward > max_reward) {
			max_reward <- random_reward
			max_arm_idx <- action
		}
	}
	return (max_arm_idx)
}

confidence_bound <- function(forest,
							 context, action,
							 eta) { # eta is the scaling parameter
	result = predict_action.causal_forest(forest, context, action, estimate.variance=TRUE)
	sigma.hat = sqrt(result$variance.estimates)
	point_estimate = result$predictions

	print (paste0('point estimate for action ', action, ': ', point_estimate))
	return (list(ucb = point_estimate + eta * sigma.hat,
			 	lcb = point_estimate - eta * sigma.hat))
}

update <- function(forest, context, action, outcome) {
	" adding the feedback to the data, this does not update the forest structure
	  the alternative method is to directly update the forest, but it is hard
	"
	if (is.null(forest)) {
		return (NULL)
	}
	new_forest <- update_add_one_sample.causal_forest(forest, context, outcome, action)
	return (new_forest)
}

new_forest <- function(data, num_actions=35) {
	" re-train the forest structure according to data 
	"
	W <- data$W
	Y <- data$Y
	X <- data$X
	forest = causal_forest(X, Y, W, num.trees=4000, num.actions=num_actions)
	return (forest)
}


