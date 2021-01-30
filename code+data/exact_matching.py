# the simplest case of matching mathine
# a match machine should return the reward (feedback)
# modified on 2019-07-28: use dictionary. the value is also a dictionary with a counter and a list

#from numba import jitclass

#@jitclass
class ExactMatching:
	def __init__(self, offline_data, \
			context_names=None, treatment_name='action', outcome_name='reward', \
			bias=0, choice_names=None):
		self.offline_data = offline_data
		self.bias = bias
		
		self.treatment_name = treatment_name
		self.outcome_name = outcome_name

		if context_names:
			self.context_names = context_names
		else:
			names = list(offline_data.keys())
			names.remove(self.treatment_name)
			names.remove(self.outcome_name)
			self.context_names = names
			
		#self.context_dim = len(offline_data.keys()) - 2
		self.choice_names = choice_names
		
		self.construct_dict()
		self.pending_action_dict = dict()

	def find_sample_reward(self, context_vec, action):
		# find the key 
		key = self.context_action_to_key(context_vec, action)
		if key not in self.data_dict:
			return False
		
		if self.data_dict[key]["count"] > 0:
			reward = self.data_dict[key]["reward_list"][  self.data_dict[key]["count"]-1  ]
			self.data_dict[key]["count"] -= 1
			return reward + self.bias[action]
		else:
			context_key = self.context_to_key(context_vec)
			if context_key not in self.pending_action_dict:
				self.pending_action_dict[context_key] = []
			self.pending_action_dict[context_key].append(action)
			return False

	def get_pending_action(self, context_vec, update_pending):
		#return False
		# return a pending action if we can find one
		context_key = self.context_to_key(context_vec)
		if context_key in self.pending_action_dict:
			if self.pending_action_dict[context_key]:
				if update_pending:
					return self.pending_action_dict[context_key].pop()
				else:
					return self.pending_action_dict[context_key][-1]
		return False

	def context_to_key(self, context_vec):
		key = ""
		for context in context_vec:
			key += "{:.3f}".format(context)
		return key

	def context_action_to_key(self, context_vec, action):
		key = ""
		for context in context_vec:
			key += "{:.3f}".format(context)
		key += str(action)
		return key

	def construct_dict(self):
		self.data_dict = dict()
		for idx in range(len(self.offline_data[self.treatment_name])):
			# because offline data in file are organized in colums
			context_vec = []
			for name in self.context_names:
				context_vec.append(self.offline_data[name][idx])
			action = self.offline_data[self.treatment_name][idx]

			key = self.context_action_to_key(context_vec, action)
			if key not in self.data_dict:
				self.data_dict[key] = {"count":0, "reward_list":[]}
			self.data_dict[key]["count"] += 1
			self.data_dict[key]["reward_list"].append(self.offline_data[self.outcome_name][idx])



