# this implements the stochastic delayed bandit algorithm 
# the difference between this algorithm and thompson sampling is that 
# this algorithm combine one thompson-sampling instance with another instance

import numpy as np
import thompson_sampling
import queue

class StochasticDelayedBandit:
	def __init__(self, N_action, alpha=0.01):
		
		self.N_action = N_action
		self.alpha = alpha

		self.last_action = 0 # set the default to 0	

		self.contextual = False

		self.base_alg = thompson_sampling.ThompsonSamplingGaussian(N_action, 1)
		self.heuristic_alg = thompson_sampling.ThompsonSamplingGaussian(N_action, 0.01)


	def init_offline_data(self, offline_data, context_names, \
			treatment_name='action', outcome_name='reward', bias=0):

		self.offline_data = offline_data
		self.treatment_name = treatment_name
		self.outcome_name = outcome_name

		if context_names:
			self.context_names = context_names
		else:
			names = list(offline_data.keys())
			names.remove(self.treatment_name)
			names.remove(self.outcome_name)
			self.context_names = names

		self.bias = bias

		self.choice_names = None # for compatibility

		self.init_queue_from_data()

	def init_queue_from_data(self):
		# initialize the queue using the offline data
		# here, the offline data is ordered
		self.queue_vec = []
		for action in range(self.N_action):
			self.queue_vec.append( queue.Queue() )

		for idx in range( len(self.offline_data[self.treatment_name]) ):
			action = self.offline_data[self.treatment_name][idx]
			self.queue_vec[action].put( self.offline_data[self.outcome_name][idx] )


	def update_by_queue(self):
		action = self.last_action
		while not self.queue_vec[action].empty():
			reward = self.queue_vec[action].get()
			self.base_alg.update(action, reward)
			action = self.base_alg.draw_arm()
		self.last_action = action

	def get_sampling_dist_sdb(self, alpha):
		# get sample distribution according to base and heuristic
		# run the simulation for 100 times to take the average
		heuristic_prob = self.heuristic_alg.arm_distribution_sampling()
		base_prob = self.base_alg.arm_distribution_sampling()
		q_vec = (1-alpha) * heuristic_prob + alpha * base_prob

		return q_vec

		## the original version to adjust does not work for the case N=2, because we cannot re-distribute the probability when a single arm has all the elements in the queue

		# print (q_vec)

		queue_size_vec = [0] * self.N_action
		for action in range(self.N_action):
			queue_size_vec[action] = self.queue_vec[action].qsize()
		max_queue_size = np.max(queue_size_vec)

		print (queue_size_vec, max_queue_size)

		u_vec = [0] * self.N_action
		d = 0 # re-distributed prob. mass
		redistributed_actions = []
		redistributed_actions.append(self.last_action)

		for action in range(self.N_action):
			u_vec[action] = max(0, (max_queue_size-queue_size_vec[action])/max_queue_size )
			if u_vec[action] < q_vec[action] and (action not in redistributed_actions):
				d += q_vec[action] - u_vec[action]
				q_vec[action] = u_vec[action]
				redistributed_actions.append(action)
		
		print (u_vec)

		other_sum = 0
		for action in range(self.N_action):
			if action not in redistributed_actions:
				other_sum += q_vec[action]
		multiplier = (other_sum+d)/other_sum

		for action in range(self.N_action):
			if action not in redistributed_actions:
				q_vec[action] *= multiplier

		return q_vec

	# we have two interfaces: draw_arm() and update()
	# implement this method as an only_online algorithm
	def draw_arm(self, raw_choices, t, option='offline_online'):
		self.update_by_queue()
		q_vec = self.get_sampling_dist_sdb(self.alpha)
		#print (sum(q_vec))
		arm = np.random.choice(self.N_action, p=q_vec)
		return arm

	def update(self, action, reward, is_online=None):
		# 1. update the queue
		self.queue_vec[action].put(reward)

		# 2. update the heuristic algorithm
		self.heuristic_alg.update(action, reward)



