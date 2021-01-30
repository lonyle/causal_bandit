# serve the data according to a function Y=f(X,a) where f is a linear/non-linear function
# the contexts and arm_features are uniformly distributed
# the parameters theta_X and theta_a are normally distributed
import numpy as np
import pandas as pd

def linear_function(theta, feature_vec): # theta is the parameter
	return np.dot(theta, feature_vec) #+ np.random.normal(0,1,1)[0] # we may add some random noise later

def sigmoid_function(theta, feature_vec):
	# this is a sigmoid function
	x = np.dot(theta, feature_vec)
	return 1 / (1 + np.exp(-x)) #+ np.random.normal(0,1,1)[0]

def compare_function(theta, feature_vec):
	n = len(theta)
	compare_vec = []
	for i in range(n):
		compare_vec.append(1 if theta[i] > feature_vec[i] else 0)
	return sum(compare_vec) / n

def constant_function(theta, feature_vec):
	return 0

def phi(X, a, N_arm=2):
	''' the phi function in https://banditalgs.com/2016/10/19/stochastic-linear-bandits/
		return the feature vector based on the context X and arm a
	'''
	# choose some terms in X according to a, truncate to N_arm parts, context_dim is N_arm times the action dim
	length = len(X) // N_arm
	return X[a*length: (a+1)*length]	

class Environment:
	def __init__(self, context_dim, N_arm, function_form_name, arm_bias_vec=[0, 0.5], binary_outcome=False):
		self.context_dim = context_dim
		self.N_arm = N_arm
		
		if context_dim % N_arm != 0:
			print ('environment.py: we suggest the number of context dimensions to multiple N_arm')

		#self.arm_bias = [0, 0] # the arm bias is the arm's expected reward, for linear, 0.0
		self.arm_bias = arm_bias_vec
		# theta dim is context_dim / N_arm
		self.theta = np.random.normal(0, 1, context_dim//self.N_arm)

		# generate the random features for the arms
		# self.arm_features = []
		# for arm_idx in range(N_arm):
		# 	self.arm_features.append(np.random.uniform(-1, 1, arm_feature_dim))

		self.binary_outcome = binary_outcome

		self.function_form_name = function_form_name
		if function_form_name == 'linear':
			self.function = linear_function
		elif function_form_name == 'sigmoid':
			self.function = sigmoid_function
		elif function_form_name == 'compare':
			self.function = compare_function
		elif function_form_name == 'constant':
			self.function = constant_function

	def generate_offline_data(self, N_sample, propensity_score_setting, pscore_func=None):
		# for the offline data, randomly choose an arm with a certain probability
		# updated on 2019-07-31: there are a set of propensity_score_vec, each with a certain probability
		# propensity_score_setting contains two fields: "propensity_score_vecs", "prob_vec"

		# updated on 2020-09-26: generate the propensity function according to the pscore_function
	
		context_vec = []
		action_vec = []
		reward_vec = []
		propensity_score_vec = []
		for n in range(N_sample):
			context = self.generate_context()

			if pscore_func == None: # the pscore is randomly generated by the pscore_setting
				prob_vec = propensity_score_setting['prob_vec']
				propensity_score_idx = np.random.choice(len(prob_vec), 1, p=prob_vec)[0]
				a_prob_vec = propensity_score_setting['propensity_score_vecs'][propensity_score_idx]
			else: # a_prob_vec depends on the context
				a_prob_vec = pscore_func(context)

			arm = np.random.choice(self.N_arm, 1, p=a_prob_vec)[0]
			if "output_type" in propensity_score_setting and propensity_score_setting['output_type'] == 'all_dimension':
				propensity_score = ','.join(map('{:.3f}'.format, a_prob_vec))
			else:
				propensity_score = a_prob_vec[arm]
			reward = self.generate_feedback(context, arm)
			
			propensity_score_vec.append(propensity_score)
			context_vec.append(context)
			action_vec.append(arm)
			reward_vec.append(reward)

		offline_data = dict()
		for d in range(self.context_dim):
			offline_data['context'+str(d)] = []
			for idx in range(N_sample):
				offline_data['context'+str(d)].append(context_vec[idx][d])
		offline_data["action"] = action_vec
		offline_data["reward"] = reward_vec
		offline_data['propensity_score'] = propensity_score_vec
		return offline_data


	def generate_context(self):
		context = list( np.random.uniform(-1, 1, self.context_dim//self.N_arm*self.N_arm) )
		return context

	def generate_feedback(self, X, a):
		feature_vec = phi(X, a, self.N_arm)
		if self.binary_outcome == False:
			return self.function(self.theta, feature_vec) + self.arm_bias[a]
		else:
			prob = self.function(self.theta, feature_vec)
			prob += (1-prob) * self.arm_bias[a]
			prob = min(prob, 1)
			prob = max(prob, 0)
			if np.random.random() < prob:
				return 1
			else:
				return 0

	def get_average_reward(self, a):
		# # compare the sampling reward and the bias
		# reward_vec = []
		# for sample in range(1000):
		# 	X = self.generate_context()
		# 	reward = self.generate_feedback(X, a)
		# 	reward_vec.append(reward)
		# print ('average reward:', np.average(reward_vec), '\t bias:', self.arm_bias[a])
		if self.binary_outcome == True and self.function_form_name == 'sigmoid':
			return 0.5 + 0.5*self.arm_bias[a] # this is for sigmoid function. The avereage is 0.5, and the remaining part to change is 0.5
		else:
			return self.arm_bias[a]

	def get_psudo_regret(self, X, a, contextual):
		average_reward = self.generate_feedback(X, a) if contextual else self.get_average_reward(a)

		return self.generate_max_reward(X, contextual) - average_reward

	def generate_max_reward(self, X, contextual):
		max_reward = -np.inf
		for a in range(self.N_arm):
			if contextual == True:
				reward = self.generate_feedback(X, a)
			else:
				reward = self.get_average_reward(a)
			if reward > max_reward:
				max_reward = reward
		return max_reward

	def dump_data(self, N_sample_offline, N_sample_online, offline_filename, online_filename):
		# dump both offline data and online interactions to two csv files
		propensity_score_setting = {
			"prob_vec": [1], 
			"propensity_score_vecs":[ [1/self.N_arm] * self.N_arm ]
		}
		offline_data = self.generate_offline_data(N_sample_offline, propensity_score_setting)
		pd.DataFrame.from_dict(offline_data, orient='index').transpose().to_csv(offline_filename)

		online_data = dict()
		for d in range(self.context_dim):
			online_data['context'+str(d)] = []
		for action in range(self.N_arm):
			online_data['reward'+str(action)] = []
			
		for t in range(N_sample_online):
			context = self.generate_context()
			for d in range(self.context_dim):
				online_data['context'+str(d)].append(context[d])
			for action in range(self.N_arm):
				reward = self.generate_feedback(context, action)
				online_data['reward'+str(action)].append(reward)

		pd.DataFrame.from_dict(online_data, orient='index').transpose().to_csv(online_filename)		

	
if __name__ == '__main__':
	# sample usage, generate csv files for the linear and non-linear case

	# linear_env = Environment(4, 2, linear_function) # dimension=4, N_arm=2
	# linear_env.dump_data(10000, 10000, 'data/linear_data_offline.csv', 'data/linear_data_online.csv')

	# nonlinear_env = Environment(10, 2, sigmoid_function)
	# nonlinear_env.dump_data(10000, 10000, 'data/sigmoid_data_offline.csv', 'data/sigmoid_data_online.csv')

	compare_env = Environment(4, 2, compare_function)
	compare_env.dump_data(10000, 10000, 'data/compare_data_offline.csv', 'data/compare_data_online.csv')

















