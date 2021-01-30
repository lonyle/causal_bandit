# prepare the data so that it can be used by the linUCB algorithm
import linUCB
import numpy as np
import pandas as pd
import json

import realdata_evaluator

class DisjointLinearMatching:
	def __init__(self, offline_data, dimension, \
			context_names, treatment_name='action', outcome_name='reward', \
			synthetic=True, choice_names=None, articles_filename=None):

		self.offline_data = offline_data
		self.dimension = dimension
		self.article2idx, _, self.n_articles = linUCB.init_articles(articles_filename)

		self.V_online = np.zeros((self.n_articles, dimension, dimension))# record the matched status
		for i in range(self.n_articles):
			self.V_online[i] = np.identity(dimension)

		self.treatment_name = treatment_name
		self.outcome_name = outcome_name
		self.context_names = context_names
		self.choice_names = choice_names

		self.synthetic = synthetic

		# allocate memory for array-like objects
		self.V_offline = np.zeros((self.n_articles, dimension, dimension))
		self.b_offline = np.zeros((self.n_articles, dimension))
		self.V_inv_offline = np.zeros((self.n_articles, dimension, dimension))
		self.theta_hat_offline = np.zeros((self.n_articles, dimension))

		self.preprocessing()
	
	def preprocessing(self):		
		for i in range(self.n_articles):
			self.V_offline[i] = np.identity(self.dimension)		

		for idx in range(len(self.offline_data[self.treatment_name])):
			context_vec = []
			for context_name in self.context_names:
				context_vec.append( self.offline_data[context_name][idx] )
			action = self.offline_data[self.treatment_name][idx]
			reward = self.offline_data[self.outcome_name][idx]
			a_vec = np.asarray( context_vec )

			article_index = self.article2idx[action]

			self.V_offline[article_index] += np.outer(a_vec, a_vec.T)
			# print (reward * a_vec)
			# print (self.b_offline[article_index])
			self.b_offline[article_index] += reward * a_vec

		for i in range(self.n_articles):
			self.V_inv_offline[i] = np.linalg.inv(self.V_offline[i])
			self.theta_hat_offline[i] = self.V_inv_offline[i].dot(self.b_offline[i])

		# theta_hat_offline = []
		# for i in range(self.n_articles):
		# 	print (self.theta_hat_offline[i])
		# 	theta_hat_offline.append(self.theta_hat_offline[i].ravel().tolist())

		# data = {'theta_hat_offline': theta_hat_offline}
		# # print (data)
		# with open('data/_yahoo_linear_model_articles.json', 'w') as f:
		# 	json.dump(data, f, indent=4)

	def find_sample_reward(self, context_vec, action):
		# print (action)
		# the action is the article index, no need to do the mapping
		article_index = action
		a_vec = np.asarray( context_vec )
		V_tmp = self.V_online[article_index] + np.outer(a_vec, a_vec.T)
		V_inv_tmp = np.linalg.inv(V_tmp)
		tmp_UCB = a_vec.T.dot(V_inv_tmp).dot(a_vec)
		offline_UCB = a_vec.T.dot(self.V_inv_offline[article_index]).dot(a_vec)
		if (tmp_UCB > offline_UCB): # the offline bound is tighter
			self.V_online[article_index] = V_tmp
			return a_vec.dot(self.theta_hat_offline[article_index])
		else:
			return False

	def get_pending_action(self, context, update_pending):
		# currently, we do not support pending action
		return False

class LinearMatching:
	def __init__(self, offline_data, dimension, \
			context_names, treatment_name='action', outcome_name='reward', \
			synthetic=True, bias=[0,0], choice_names=None):

		self.offline_data = offline_data
		self.dimension = dimension
		self.V_online = np.identity(self.dimension)

		self.context_dim = len(offline_data.keys()) - 3

		self.treatment_name = treatment_name
		self.outcome_name = outcome_name
		self.choice_names = choice_names

		# modified on 9-12, for the default case where we do not want to specify all the contexts
		if context_names:
			self.context_names = context_names
		else:
			names = list(offline_data.keys())
			names.remove(self.treatment_name)
			names.remove(self.outcome_name)
			self.context_names = names

		self.synthetic = synthetic
		self.bias = bias

		self.preprocessing()

	def preprocessing(self):
		# construct the bound
		#print ('processing the offline data for the use of linUCB')
		self.V_offline = np.identity(self.dimension)
		self.b_offline = np.zeros(self.dimension)

		for idx in range(len(self.offline_data[self.treatment_name])):
			context_vec = []
			for context_name in self.context_names:
				context_vec.append( self.offline_data[context_name][idx] )
			action = self.offline_data[self.treatment_name][idx]
			reward = self.offline_data[self.outcome_name][idx]
			a_vec = linUCB.arm_context_feature(context_vec, action, self.synthetic)

			self.V_offline += np.outer(a_vec, a_vec.T)
			self.b_offline += reward * a_vec

		self.V_inv_offline = np.linalg.inv(self.V_offline)
		self.theta_hat_offline = self.V_inv_offline.dot(self.b_offline)


	def find_sample_reward(self, context, action):
		# first get the estimated confidence interval bound
		a_vec = linUCB.arm_context_feature(context, action, self.synthetic)
		V_tmp = self.V_online + np.outer(a_vec, a_vec.T)
		V_inv_tmp = np.linalg.inv(V_tmp)
		tmp_UCB = a_vec.T.dot(V_inv_tmp).dot(a_vec)
		offline_UCB = a_vec.T.dot(self.V_inv_offline).dot(a_vec)
		if (tmp_UCB > offline_UCB): # the offline bound is tighter than the online bound
			self.V_online = V_tmp
			return a_vec.dot(self.theta_hat_offline) + self.bias[action]

		else:
			return False

	def get_pending_action(self, context, update_pending):
		# currently, we do not support pending action
		return False

if __name__ == '__main__':
	orig_data_filename = 'data/_yahoo-webscope-logs.txt'
	column_names = ["timestamp", "context1", "context2", "context3", "context4", "context5", "context6", \
		"drawn_article", "reward", "article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"]

	orig_df = pd.read_csv(orig_data_filename, sep=" ", usecols=range(30), header=None, names=column_names)
	N_offline = 100000
	offline_data = realdata_evaluator.get_offline_data(orig_df, N_offline)
	dimension = 6
	context_names = ['context1', 'context2', 'context3', 'context4', 'context5', 'context6']
	treatment_name = 'drawn_article'
	outcome_name = 'reward'
	choice_names = [
		"article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"
	]
	articles_filename = 'data/_yahoo-webscope-articles.txt'
	match_machine = DisjointLinearMatching(offline_data, dimension, \
			context_names, treatment_name=treatment_name, outcome_name=outcome_name, \
			synthetic=False, choice_names=choice_names, articles_filename=articles_filename)


