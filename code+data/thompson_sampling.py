# use thompson sampling as the online learning oracle
# in this experiment, we assume a binary outcome

import numpy as np

class ThompsonSampling: # beta distribution
	def __init__(self, N_arm):
		self.N_arm = N_arm
		self.count_success = [1] * N_arm # each arm has an entry
		self.count_failure = [1] * N_arm

		self.contextual = False

	def draw_arm(self, raw_choices=[], t=None, option='offline_online'):
		# the thompson sampling oracle do ont need the time t
		max_reward = -np.inf
		opt_action = None
		if len(raw_choices) == 0:
			choices = list(range(self.N_arm))
		else:
			choices = raw_choices
		for action in choices:
			posterior_reward = np.random.beta(self.count_success[action], self.count_failure[action])
			if posterior_reward > max_reward:
				max_reward = posterior_reward
				opt_action = action
		return opt_action

	def update(self, action, reward, is_online=None):
		# is_online indicates that whether the feedback is from the online phase
		# this thompson sampling oracle do not need to know whether the feedback is online
		# now, we accept the non-binary outcome
		if reward not in [0, 1]:
			if reward < 0 or reward > 1:
				print ('we assume a binary outcome!!')
				return -1
			else:
				if np.random.random() < reward:
					reward = 1
				else:
					reward = 0
		if reward == 1:
			self.count_success[action] += 1
		else:
			self.count_failure[action] += 1

class ThompsonSamplingGaussian:
	def __init__(self, N_arm, scale=0.5):
		self.N_arm = N_arm
		self.scale = scale

		self.ave_vec = [0] * N_arm
		self.count_vec = [0] * N_arm

		self.contextual = False

	def draw_arm(self, raw_choices=[], t=None, option='offline_online'):
		# in this implementation, we do not need to know t
		max_reward = -np.inf
		opt_action = None
		if len(raw_choices) == 0:
			choices = list(range(self.N_arm))
		else:
			choices = raw_choices

		for action in choices:
			posterior_reward = np.random.normal(self.ave_vec[action], self.scale*np.sqrt(1/(self.count_vec[action]+1)) )
			if posterior_reward > max_reward:
				max_reward = posterior_reward
				opt_action = action
		return opt_action

	def arm_distribution_sampling(self):
		# estimate the probability to choose each arm by 100 samples
		N_sample = 100
		count_samples = [0] * self.N_arm
		for n in range(N_sample):
			sample_arm = self.draw_arm()
			count_samples[sample_arm] += 1

		prob_vec = []
		for arm in range(self.N_arm):
			prob_vec.append(count_samples[arm] / N_sample)

		return np.asarray(prob_vec)

	def update(self, action, reward, is_online=None):
		# in this implementation, we do not need to know whether the feedback is online
		self.ave_vec[action] = (self.ave_vec[action]*self.count_vec[action] + reward)/(self.count_vec[action]+2)
		self.count_vec[action] += 1







