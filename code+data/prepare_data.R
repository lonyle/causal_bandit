" the data should be pre-processed, so that we know which is the treatment(intervention) and which is the outcome " 

load_data <- function(dataset_name) {
	if (dataset_name == "lalonde") {
		return (load_lalonde_data())
	}
	else if (dataset_name == "mindset") {
		return (load_mindset_data())
	}
	else if (dataset_name == 'gmG') {
		return (load_gmG_data())
	}
	else if (dataset_name == 'gmD') {
		return (load_gmD_data())
	}
	else if (dataset_name == 'gmB') {
		return (load_gmB_data())
	}
	else if (dataset_name == 'gmL') {
		return (load_gmL_data())
	}
	else if (dataset_name == 'gmInt') {
		return (load_gmInt_data())
	}
	else if (dataset_name == 'gmI') {
		return (load_gmI_data())
	}
	else {
		print (paste('wrong name of dataset!!', 'dataset_name'))
	}
}

# the first dataset
load_lalonde_data <- function() {
	" this data is from the package MatchIt
		this is from economics, recording multi-dimension features of individuals
		the treatment is treat, the outcome is re78(real earning in 1978)
		this data has ground truth ATT=1800, by randomized experiments
		ref: https://dango.rocks/blog/2018/11/15/Causality1-Playaround-with-the-Lalonde-Dataset/
		614 samples
	"
	library(data.table)
	data(lalonde, package='MatchIt')
	lalonde <- as.data.table(lalonde)
	W = lalonde$treat # the treatment
	Y = lalonde$re78 # the outcome
	X = lalonde[, c('age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75')] # confounders (all the columns except for treat and outcome)
	ret <- list("treatment" = W, "outcome" = Y, "confounder" = X, 'raw_data' = lalonde, 'treatment_name' = 'treat', 'outcome_name' = 're78')
	return (ret) # data (treat, outcome, covariate)
}

load_mindset_data <- function() {
	" this data comes from a national study of learning mindset
		ref: http://mindsetscholarsnetwork.org/about-the-network/current-initatives/national-mindset-study/#
		ref paper: Estimating Treatment Effects with Causal Forests: An Application
		ref link: https://github.com/grf-labs/grf/tree/master/experiments/acic18
		this dataset is large, the treatment is mindset, the outcome is the school performance?
		this data has randomized experiments, which provides ground truth
		this data can use causal forest, known from grf github
		10000+ samples
	"
	data <- read.csv(file='data/acic_2018_synthetic_data.csv', header=TRUE, sep=',')
	Z = data$Z
	Y = data$Y
	X = data[, c('C1', 'C2', 'C3', 'XC', 'X1', 'X2', 'X3', 'X4', 'X5')]
	ret <- list("treatment" = Z, "outcome" = Y, "confounder" = X, 'raw_data' = data, 'treatment_name'='Z', 'outcome_name' = 'Y')
	return (ret)
}

load_acic_2017_data <- function() {
	" this is from the second dataset challenge in ACIC
		ref: https://causal.unc.edu/files/2017/03/ACIC_Data_Analysis_Challenge.pdf
		this is a simulated data with a data generating process
		ref using bayesian additive regression tree: https://github.com/ck37/atlantic-causal-2017
	"
}

load_gmG_data <- function() {
	" Graphical Model 8-Dimensional Gaussian Example Data
		ref: https://rdrr.io/cran/pcalg/man/gmG.html
		simulated data, 500 samples
	"
	library('pcalg')
	data('gmG')
	raw_data <- as.data.frame(gmG8$x) # not binary treatment
	Z = raw_data$Ctrl
	Y = raw_data$Goal
	X = raw_data[, c('Author', 'Bar', 'V5', 'V6', 'V7', 'V8')]
	ret <- list("treatment" = Z, "outcome" = Y, "confounder" = X, 'raw_data' = raw_data, 'treatment_name'='Ctrl', 'outcome_name' = 'Goal')
	return (ret)
}

load_gmD_data <- function() {
	" discrete data, from pcalg package
	  10000 samples
	"
	library('pcalg')
	data(gmD)
	raw_data <- as.data.frame(gmD$x)
	Z = raw_data$X2
	Y = raw_data$X5
	X = raw_data[, c('X1', 'X3', 'X4')]
	ret <- list("treatment" = Z, "outcome" = Y, "confounder" = X, 'raw_data' = raw_data, 'treatment_name'='X2', 'outcome_name' = 'X5')
	return (ret)
}

load_gmB_data <- function() {
	" binary data, from pcalg package
	  5000 samples
	"
	library(pcalg)
	data(gmB)
	raw_data <- as.data.frame(gmB$x)
	Z = raw_data$V5
	Y = raw_data$V1
	X = raw_data[, c('V2', 'V3', 'V4')]
	ret <- list("treatment" = Z, "outcome" = Y, "confounder" = X, 'raw_data' = raw_data, 'treatment_name'='V5', 'outcome_name' = 'V1')
	return (ret)
}

load_gmL_data <- function() {
	" gmL: Latent Variable 4-Dim Graphical Model Data Example
	  10000 samples
	"
	library(pcalg)
	data(gmL)
	raw_data <- as.data.frame(gmL$x)
	Z = raw_data$'2'
	Y = raw_data$'5'
	X = raw_data[, c('3', '5')]
	ret <- list("treatment" = Z, "outcome" = Y, "confounder" = X, 'raw_data' = raw_data, 'treatment_name'='2', 'outcome_name' = '5')
		return (ret)
}

load_gmInt_data <- function() {
	" gmInt: Graphical Model 8-Dimensional Interventional Gaussian Example
	  5000 samples
	"
	library('pcalg')
	data('gmInt')
	raw_data <- as.data.frame(gmInt$x) # not binary treatment
	Z = raw_data$Ctrl
	Y = raw_data$Goal
	X = raw_data[, c('Author', 'Bar', 'V5', 'V6', 'V7', 'V8')]
	ret <- list("treatment" = Z, "outcome" = Y, "confounder" = X, 'raw_data' = raw_data, 'treatment_name'='Ctrl', 'outcome_name' = 'Goal')
	return (ret)
}

load_gmI_data <- function() {
	" gmI: Graphical Model 7-dim IDA Data Examples
	  10000 samples
	"
	library(pcalg)
	data(gmI)
	raw_data <- as.data.frame(gmI$x)
	Z = raw_data$V1
	Y = raw_data$V7
	X = raw_data[, c('V2', 'V3', 'V4', 'V5', 'V6')]
	ret <- list("treatment" = Z, "outcome" = Y, "confounder" = X, 'raw_data' = raw_data, 'treatment_name'='V1', 'outcome_name' = 'V7')
	return (ret)
}
