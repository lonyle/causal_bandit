import algorithm_framework
import exact_matching
import UCB
import thompson_sampling
import linUCB
import environment
import json
import IPS_weighting
import PS_matching
import linear_matching
import supervised_matching
import stochastic_delayed_bandit

import realdata_evaluator

import pandas as pd
import numpy as np
import time

def run_algorithm(params, T, option='offline_online'):
	N_arm = params['N_arm']

	choice_names = None
	if 'choice_names' in params: # if we specify the filed 'choice_names'
		choice_names = params['choice_names']

	if params['synthetic'] == True:
		context_dim = params['context_dim']
		N_offline = params['N_offline']
		if 'propensity_score_setting' in params:
			propensity_score_setting = params['propensity_score_setting']
		else:
			propensity_score_setting = None
		binary_outcome = False
		if 'binary_outcome' in params:
			binary_outcome = params['binary_outcome']

		env = environment.Environment(context_dim, N_arm, params['function_form'], params['arm_bias_vec'], binary_outcome)
			
		######### added on 2020-09-27 #########
		#######################################
		if 'reward_pscore_correlation' in params: # we generate the pscore related to the reward
			correlation = params['reward_pscore_correlation']
			ave_reward = params['arm_bias_vec']
			#pscore_correlation_action = params['pscore_correlation_action']
			def pscore_func(context):
				weight_vec = [] # sigmoid propensity
				for action in range(N_arm):
					reward = env.generate_feedback(context, action)
					#is_max_reward_action = 1 if ave_reward[action] == max(ave_reward) else -1
					is_max_reward_action = ave_reward[action] - ave_reward[(action+1)%N_arm]
					exponent1 = (reward - ave_reward[action]) * (is_max_reward_action) * correlation
					#exponent2 = reward * correlation
					#exponent3 = (reward - ave_reward[action]) * pscore_correlation_action[action] * correlation
					weight_vec.append( np.exp(exponent1) )
				sum_weight = np.sum(weight_vec)
				a_prob_vec = []
				for action in range(N_arm):
					a_prob_vec.append(weight_vec[action]/sum_weight)
				#print (a_prob_vec)
				return a_prob_vec
		else:
			pscore_func = None

		offline_data = env.generate_offline_data(N_offline, propensity_score_setting, pscore_func)
		######### added on 2020-10-09 #########
		#######################################
		# delete come contexts so that there are some unobserved contexts
		if 'unobserved_context_dims' in params:
			for d in params['unobserved_context_dims']:
				del offline_data['context'+str(d)]

	else: # real-data
		offline_df = pd.read_csv(params['obs_data_filename'])

		offline_data = realdata_evaluator.get_offline_data(offline_df, params['N_offline'])

		experimental_df = pd.read_csv(params['exp_data_filename'])

		evaluator = realdata_evaluator.OnlineEvaluator(experimental_df, \
			params['context_names'], params['treatment_name'], params['outcome_name'], choice_names)

	### choose an online bandit oracle
	if params['algorithm'] == 'UCB':
		if 'UCB_beta' in params:
			algorithm = UCB.UCB(N_arm, beta=params['UCB_beta'])
		else:
			algorithm = UCB.UCB(N_arm)
	elif params['algorithm'] == 'LinUCB':
		algorithm = linUCB.LinUCB(N_arm, params['dimension'], params['synthetic'])
	elif params['algorithm'] == 'DisjointLinUCB':
		algorithm = linUCB.DisjointLinUCB(N_arm, params['dimension'], params['synthetic'])
	elif params['algorithm'] == 'thompson_sampling':
		algorithm = thompson_sampling.ThompsonSampling(N_arm)
	elif params['algorithm'] == 'thompson_sampling_gaussian':
		algorithm = thompson_sampling.ThompsonSamplingGaussian(N_arm, scale=params['TS_scale'])
	elif params['algorithm'] == 'SDB':
		algorithm = stochastic_delayed_bandit.StochasticDelayedBandit(N_arm)
		if option != 'only_online':
			print ('the running of stochastic_delayed_bandit only allows only_online option!!')
			option = 'only_online'
	else:
		print ('the online algorithm is not supported!')

	### choose an offline matching machine
	if params['match_machine'] == 'exact':
		match_machine = exact_matching.ExactMatching(offline_data, \
			params['context_names'], params['treatment_name'], params['outcome_name'], params['bias'])
	elif params['match_machine'] == 'IPS_weighting':
		has_pscore = False if 'has_pscore' in params and params['has_pscore']==False else True
		match_machine = IPS_weighting.WeightedMatching(offline_data, N_arm, \
			params['context_names'], params['treatment_name'], params['outcome_name'], 
			params['bias'], choice_names, has_pscore)
	elif params['match_machine'] == 'PS_matching':
		if 'match_on_action' in params and params['match_on_action'] == True:
			match_machine = PS_matching.PropensityScoreMatching(offline_data, N_arm, params['N_type'], \
				params['context_names'], params['treatment_name'], params['outcome_name'], params['bias'],
				match_on_action=True)
		else:
			match_machine = PS_matching.PropensityScoreMatching(offline_data, N_arm, params['N_type'], \
				params['context_names'], params['treatment_name'], params['outcome_name'], params['bias'])
	elif params['match_machine'] == 'linear_matching':
		match_machine = linear_matching.LinearMatching(offline_data, params['dimension'], \
			params['context_names'], params['treatment_name'], params['outcome_name'], params['synthetic']) # for linear matching, we don't need the bias param

	elif params['match_machine'] == 'disjoint_linear_matching':
		match_machine = linear_matching.DisjointLinearMatching(offline_data, params['dimension'],\
			params['context_names'], params['treatment_name'], params['outcome_name'], params['synthetic'],\
			params['choice_names'], params['articles_filename'])
	elif params['match_machine'] == 'supervised_matching':
		match_machine = supervised_matching.SupervisedMatching(offline_data, N_arm, params['model_name'], \
			params['context_names'], params['treatment_name'], params['outcome_name'], params['bias'])
	elif params['match_machine'] == 'SDB': # in the implementation of this algorithm, the offline evaluator and the online bandit oracle are together
		algorithm.init_offline_data(offline_data, params['context_names'], \
			params['treatment_name'], params['outcome_name'], params['bias'])
		match_machine = algorithm
	else:
		print ('the match machine is not supported!')

	### set up the algorithm runner
	algorithm_runner = algorithm_framework.AlgorithmFramework(algorithm, offline_data, match_machine, option)
	# if we set specify to use the true context distribution
	if 'true_context_distribution' in params:
		if params['true_context_distribution'] == True:
			algorithm_runner.get_environment_for_context(env)

	# use the offline data to do a batch update of the online bandit oracle
	if 'batch_mode' in params:
		if params['batch_mode'] == True:
			algorithm_runner.batch_mode_status = True

	if params['synthetic'] == True:
		return run_algorithm_core(algorithm_runner, env, T)
	else:
		return evaluator.evaluate_algorithm(algorithm_runner, T)

def run_algorithm_core(algorithm_runner, env, T):
	# really run the algorithm
	regret_vec = []
	for t in range(T):
		#print ('time', t)
		context = env.generate_context()
		action = algorithm_runner.real_draw_arm(context)
		reward = env.generate_feedback(context, action)
		
		# for action in [0, 1]:
		# 	print ('real reward for action ', action ,':', env.generate_feedback(context, action) )
		regret = env.get_psudo_regret(context, action, algorithm_runner.algorithm.contextual)

		algorithm_runner.real_feedback(reward)

		regret_vec.append(regret)
	# print ('real theta:', env.theta)
	return regret_vec

def output_algorithm(params, T, option, output_filename, repeat_times=1, is_append=False):
	print (option)
	if is_append == False:
		open(output_filename, 'w').close()
	sum_cumulative_regret = 0

	start_time = time.time()
	for n in range(repeat_times):
		#print ('repeat for', n, 'times')
		regret_vec = run_algorithm(params, T, option)
		cumulative_regret_vec = []
		cumulative_regret = 0
		for t in range(len(regret_vec)):
			cumulative_regret += regret_vec[t]
			cumulative_regret_vec.append(cumulative_regret)

		sum_cumulative_regret += cumulative_regret
		with open(output_filename, 'a') as output_file:
			json.dump(cumulative_regret_vec, output_file)
			output_file.write('\n')

	print ('time elasped:', time.time()-start_time)
	print ('the average cum. regret after', T, 'rounds:', sum_cumulative_regret/repeat_times)

