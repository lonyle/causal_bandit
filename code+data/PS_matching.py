# propensity score stratification, find the nearest propensity score
import numpy as np

class PropensityScoreMatching:
	def __init__(self, offline_data, N_action, N_type, \
			context_names, treatment_name='action', outcome_name='reward', bias=0,\
			match_on_action=False): # the offline_data has the field propensity_score, N_type is the number of stratification
		self.offline_data = offline_data
		self.N_type = N_type
		self.N_action = N_action

		self.treatment_name = treatment_name
		self.outcome_name = outcome_name
		self.match_on_action = match_on_action # whether to match the propensity score only on the selected action

		# modified on 9-12, for the default case where we do not want to specify all the contexts
		if context_names:
			self.context_names = context_names
		else:
			names = list(offline_data.keys())
			names.remove(self.treatment_name)
			names.remove(self.outcome_name)
			self.context_names = names
		
		self.bias = bias

		self.preprocessing()

	def preprocessing(self):
		# pre-compute the average values
		#print ('preprocessing for PS match machine')
		N_type_all_dim = self.N_type ** (self.N_action-1)
		self.count_action = np.zeros( shape=(self.N_action, N_type_all_dim) )
		self.average_reward_action = np.zeros( shape=(self.N_action, N_type_all_dim) )
		sum_reward_action = np.zeros( shape=(self.N_action, N_type_all_dim) )

		for idx in range(len(self.offline_data[self.treatment_name])):
			action = self.offline_data[self.treatment_name][idx]
			propensity_score = self.offline_data['propensity_score'][idx]
			Type = self.get_closest_type(propensity_score)
			self.count_action[action][Type] += 1
			sum_reward_action[action][Type] += self.offline_data[self.outcome_name][idx]

		for action in range(self.N_action):
			for Type in range(self.N_type):
				if self.count_action[action][Type] > 1:
					self.average_reward_action[action][Type] = sum_reward_action[action][Type] / self.count_action[action][Type]

	def get_closest_type(self, propensity_score):
		# now propensity_score is a prob_vec like "0.211,0.789"
		# we only need to match the first N_arm-1 dimensions
		#print (propensity_score)
		if self.match_on_action == False:
			prob_vec = list(map(float, propensity_score.split(',')))
			Type = 0
			for action in range(self.N_action-1):
				dim_type = round( prob_vec[action] / (1/(self.N_type-1)) )
				Type = Type * self.N_type + dim_type
		else:
			Type = round( propensity_score / (1/(self.N_type-1)) )
		return Type


	def find_sample_reward(self, context, action):
		# to be compatible with other APIs, the context is the propensity_score
		propensity_score = context
		Type = self.get_closest_type(propensity_score)
		if self.count_action[action][Type] > 1:
			self.count_action[action][Type] -= 1
			return self.average_reward_action[action][Type] + self.bias[action]
		else:
			return False

	def get_pending_action(self, context, update_pending):
		# currently, we do not support pending action
		return False