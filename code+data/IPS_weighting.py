# match the sample according to the weight
# a match machine that return the reward

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

def get_features(data, feature_colnames):
	return data.loc[:, feature_colnames]

class WeightedMatching:
	def __init__(self, offline_data, N_action, context_names, treatment_name='action', \
			outcome_name='reward', bias=0, choice_names=None, has_pscore=True):
		self.offline_data = offline_data

		self.N_action = N_action
		self.treatment_name = treatment_name
		self.outcome_name = outcome_name
		# modified on 9-12, for the default case where we do not want to specify all the contexts
		if context_names:
			self.context_names = context_names
		else:
			names = list(offline_data.keys())
			names.remove(self.treatment_name)
			names.remove(self.outcome_name)
			self.context_names = names

		self.choice_names = choice_names

		self.has_pscore = has_pscore # by default suppose the offline_data includes the propensity scores
		self.preprocessing(N_action)
		self.bias = bias

	def preprocessing(self, N_action):
		# pre-process the samples, and store the number of samples to be canceled in the class
		# print ('constructing a IPS_weighting matching machine...')
		self.count_action = [0] * N_action
		count_action_numerator = [0] * N_action
		count_action_denominator = [0] * N_action
		sum_reward_action = [0] * N_action
		self.average_reward_action = [0] * N_action

		## added on 2020-10-09
		if 'propensity_score' not in self.offline_data or self.has_pscore == False:
			self.offline_data['propensity_score'] = self.estimate_propensity_score()

		for idx in range(len(self.offline_data[self.treatment_name])):
			action = self.offline_data[self.treatment_name][idx]
			
			propensity_score = self.offline_data['propensity_score'][idx]

			count_action_numerator[action] += 1/propensity_score
			count_action_denominator[action] += (1/propensity_score)**2

			sum_reward_action[action] += self.offline_data[self.outcome_name][idx]/propensity_score

		for action in range(N_action):
			# print (action)
			# print (count_action_denominator)

			#action_numerator is the sum of weights(ps), action_denominator is the sum of weight^2 
			if count_action_denominator[action] > 0:
				self.count_action[action] = (count_action_numerator[action])**2 / count_action_denominator[action] 
				self.average_reward_action[action] = sum_reward_action[action] / count_action_numerator[action]
			else: 
				self.count_action[action] = 0
				self.average_reward_action[action] = None # not defined

		# print ('after preprocessing, the average:')
		# for action in range(self.N_action):
		# 	print ('action:', action, 'count:', self.count_action[action])
		# 	print ('reward:', self.average_reward_action[action])
		
	def estimate_propensity_score(self, method='xgboost'):
		# added on 2020-10-09
		# use logistic regression to predict the propensity score
		input_df = pd.DataFrame.from_dict(self.offline_data, orient='index').transpose()
		X_train = get_features(input_df, self.context_names)
		y_train = input_df[self.treatment_name]

		if method == 'logistic_regression':
			reg = LogisticRegression(random_state=0).fit(X_train, y_train)
			propensity_score_vec = reg.predict_proba(X_train).ravel()
		elif method == 'xgboost':
			train_data = xgb.DMatrix(X_train, label=y_train)
			param = {'objective': 'binary:logistic'}
			num_round = 2
			bst = xgb.train(param, train_data, num_round)

			predictions = bst.predict(train_data)
			propensity_score_vec = predictions

		return propensity_score_vec

	def find_sample_reward(self, context, action):
		# since we pre-processed the data, we only need to match the action
		if self.count_action[action] > 1:
			self.count_action[action] -=1
			return self.average_reward_action[action] + self.bias[action]
		else:
			return False

	def get_pending_action(self, context, update_pending):
		# currently, we do not support pending action
		return False

