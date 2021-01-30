import numpy as np

class AlgorithmFramework:
	def __init__(self, algorithm, offline_data, match_machine, option='offline_online'):
		self.algorithm = algorithm # the algorithm oracle has 'draw_arm' and 'update' two APIs
		self.match_machine = match_machine
		self.offline_data = offline_data
		self.option = option

		self.N_offline = len(self.offline_data[self.match_machine.treatment_name])
		self.context_dim = len(offline_data.keys()) - 3 # exclude "action", "reward" and "propensity score"

		# if the match_machine is ps_matching, the context is the propensity score
		self.context_pool = []
		self.choice_pool = [] # has the same index as the context
		for idx in range(self.N_offline):
			if self.match_machine.__class__.__name__ == 'PropensityScoreMatching':
				self.context_pool.append(self.offline_data['propensity_score'][idx])
			else:
				context_vec = []
				for context_name in self.match_machine.context_names:
					context_vec.append(self.offline_data[context_name][idx])
				self.context_pool.append(context_vec)

				# added on Jan 29
				if self.match_machine.choice_names != None:
					choice_vec = []
					for choice_name in self.match_machine.choice_names:
						choice_vec.append(self.offline_data[choice_name][idx])
					self.choice_pool.append(choice_vec)

		self.context_generator = False
		self.t = 0 # self.maintain a time count

		self.batch_mode_status = None # by default, batch_mode_status=None which means we do not use the batch mode
		# When batch_mode_status = True, we need to do the batch update
		# When batch_mode_status = False, we are in the online phase and we skip the batch update

	######## public ########
	def real_draw_arm(self, context, choices=[], do_match=True, update_pending=True):
		# sometimes, we choose from a selected subset
		self.real_context = context
		self.real_choices = choices

		# do the matching first
		if self.option != 'only_online':
			if self.batch_mode_status == None:
				if do_match:
					self.match_all_possible()
				action = self.match_machine.get_pending_action(context, update_pending)
				if action: # short-cut
					print ('context:', context, 'pending action:', action)
					return action
			elif self.batch_mode_status == True:
				self.match_all_possible_batch()
			elif self.batch_mode_status == False:
				pass # do nothing
			else:
				print ('invalid value for batch_mode_status')
		
		action = self.choose_arm(context, choices)
		self.real_action = action
		return action

	def real_feedback(self, reward):
		if self.option != 'only_offline':
			# if only_offline, do not need to update the online feedback
			self.update(self.real_context, self.real_action, reward, is_online=True)

		if self.match_machine.__class__.__name__ != 'PropensityScoreMatching':
			self.context_pool.append(self.real_context)
			if self.match_machine.choice_names != None:
				self.choice_pool.append(self.real_choices)
	########################

	def choose_arm(self, context, choices):
		if self.algorithm.contextual == True:
			if len(choices) == 0: # default choices from a fixed set of arms
				action = self.algorithm.draw_arm(context, self.t, self.option)
			else:
				action = self.algorithm.draw_arm(context, choices, self.t, self.option)
		else: # now, context-independent decisions support dynamic choices
			action = self.algorithm.draw_arm(choices, self.t, self.option)
		return action
		
	def update(self, context, action, reward, is_online=False):
		# depending on whether the algorithm has context or not
		if self.algorithm.contextual == True:
			self.algorithm.update(context, action, reward, is_online)
		else:
			self.algorithm.update(action, reward, is_online)
		self.t += 1

	def match_all_possible(self):
		# match until there is no matched
		while True:	
			sample = self.match(self.t)
			if sample:
				self.update(sample['context'], sample['action'], sample['reward'])
			else:
				return

	def match(self, t):
		# different algorithms 
		random_context, random_choice_set = self.generate_random_context()
		if self.algorithm.contextual == True:
			action = self.algorithm.draw_arm(random_context, random_choice_set, t)
		else:
			action = self.algorithm.draw_arm(random_choice_set, t)
		reward = self.match_machine.find_sample_reward(random_context, action)
		if not reward:
			return False
		else:
			return {"context": random_context, "reward": reward, "action": action}

	## updated on 2020-10-08: match all the data points in a batch mode
	def match_all_possible_batch(self):
		for action in range(self.algorithm.N_arm):
			while True:
				random_context, _ = self.generate_random_context()
				reward = self.match_machine.find_sample_reward(random_context, action)
				if not reward: # no more offline data
					break
				else:
					self.update(random_context, action, reward)


	def get_environment_for_context(self, env):
		self.context_generator = True
		self.env = env

	def generate_random_context(self):
		# generate a random context from the context distribution
		# TODO: if we have both context and choice_set, we also need to generate the choice_set

		if self.context_generator == True:
			random_choice_set = [] # set the random choice set to empty (do not consider Yahoo data)
			return self.env.generate_context(), random_choice_set

		random_idx = np.random.randint(len(self.context_pool))
		random_context = self.context_pool[random_idx]

		if len(self.choice_pool) > 0:
			random_choice_set = self.choice_pool[random_idx]
		else:
			random_choice_set = []
		return random_context, random_choice_set

