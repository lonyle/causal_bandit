# a class UCB
import numpy as np

import sys
sys.path.insert(0, 'plot')
import average_reward_yahoo

class UCB:
	# N_arm: the number of arms
	def __init__(self, N_arm, beta=2):
		self.cum_vec = [0] * N_arm
		self.count_vec = [0] * N_arm
		self.ave_vec = [-np.inf] * N_arm

		self.online_cum_vec = [0] * N_arm
		self.online_count_vec = [0] * N_arm
		self.online_ave_vec = [None] * N_arm

		self.N_arm = N_arm
		self.contextual = False
		self.beta = beta

	def draw_arm(self, raw_choices, t, option='offline_online'):
		# here, we add the parameter raw_choices to process numbered choices such as in Yahoo
		# the parameter t is the total time that UCB is played, maintained by algorithm framework
		choices = []
		for raw_choice in raw_choices:
			if not np.isnan(raw_choice):
				choices.append(int(raw_choice))

		if len(choices) == 0:
			choices = list(range(self.N_arm))
		#print (choices)
		for i in choices:
			if self.count_vec[i] == 0 and option != 'only_offline':
				return i

		# choose the arm with the upper confidence bound
		
		confidence_bound_vec = np.asarray(self.ave_vec)
		online_t = sum(self.count_vec)

		max_UCB = -np.inf
		if option != 'only_offline': # for the only offline strategy, use the mean instead of the ucb
			for i in choices:
				#print ('choice:', i)
				confidence_bound_vec[i] += np.sqrt(self.beta*np.log(t)/self.count_vec[i])

				# added on 9-13: when the empirical average is out of the online confidence interval, use the online confidence bound
				# if self.online_count_vec[i] > 0:
				# 	online_width = np.sqrt(2*np.log(online_t)/self.online_count_vec[i])
				# 	#if self.ave_vec[i] < self.online_ave_vec[i] - online_width or self.ave_vec[i] > self.online_ave_vec[i] + online_width:
				# 	if confidence_bound_vec[i] < self.online_ave_vec[i] - online_width or confidence_bound_vec[i] > self.online_ave_vec[i] + online_width:
				# 		confidence_bound_vec[i] = self.online_ave_vec[i] + online_width

				if confidence_bound_vec[i] > max_UCB:
					max_UCB = confidence_bound_vec[i]
					max_i = i

		# modified on 2-20: only choose one from the the choice set
		else:
			max_i = choices[ np.argmax( confidence_bound_vec[choices] ) ]
		
		# max_reward_true = -np.inf
		# max_action_true = None
		# for action in choices:
		# 	tmp_reward = average_reward_yahoo.average_reward_vec[action]
		# 	if tmp_reward > max_reward_true:
		# 		max_reward_true = tmp_reward
		# 		max_action_true = action

		# print ('true max action:', max_action_true, max_reward_true)
		# print ('chosen action  :', max_i, np.max(confidence_bound_vec[choices]))
		# print (self.count_vec)

		return max_i

	def update(self, action, reward, is_online):
		# update the information according to the feedback
		self.cum_vec[action] += reward
		self.count_vec[action] += 1
		self.ave_vec[action] = self.cum_vec[action]/self.count_vec[action]

		# added on 9-13: the update function to distinguish the online feedback from the offline data
		if is_online:
			self.online_cum_vec[action] += reward
			self.online_count_vec[action] += 1
			self.online_ave_vec[action] = self.online_cum_vec[action]/self.online_count_vec[action]


		
