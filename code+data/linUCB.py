# in this version, we consider shared parameter for all arms
# reference : https://banditalgs.com/2016/10/19/stochastic-linear-bandits/
import numpy as np
import environment
import pandas as pd

def arm_context_feature(context_vec, arm_idx, synthetic=True):
	# we use the phi function in environment.py for consistency
	#print (context_vec, arm_idx)
	if synthetic == True:
		return np.asarray( environment.phi(context_vec, arm_idx) )
	else: # real data, linear regression including the arm
		return np.asarray( context_vec + [arm_idx] )

def init_articles(articles_filename):
	article2idx = dict()
	articles_np = np.loadtxt(articles_filename)
	articles_feature = dict()
	index = 0
	for article in articles_np:
		key = article[0]
		article2idx[key] = index
		articles_feature[index] = [float(x) for x in article[1:]]
		index += 1

	return article2idx, articles_feature, index # return the number of articles

def convert_index(exp_data_filename='data/_yahoo_exp_80000.csv', 
		obs_data_filename='data/_yahoo_obs_20000.csv', 
		articles_filename='data/_yahoo-webscope-articles.txt', 
		output_exp_data_filename='data/_yahoo_reindex_exp_80000.csv',
		output_obs_data_filename='data/_yahoo_reindex_obs_20000.csv'):
	'''
		convert the indices in the log file to the smaller ones
	'''
	article2idx, _, n_articles = init_articles(articles_filename)
	offline_df = pd.read_csv(obs_data_filename)
	experimental_df = pd.read_csv(exp_data_filename)

	def convert_function(article_number):
		if isinstance(article_number, int):
			return int(article2idx[article_number])
		elif (article_number == 109453.0): # a special case
			# print (article_number)
			return int(article2idx[int(article_number)])
		else: # not a number
			return article_number

	column_names = [
		"drawn_article", "article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"
	]

	for column_name in column_names:
		print ('column_name:', column_name)
		offline_df[column_name] = offline_df[column_name].map(convert_function)
		experimental_df[column_name] = experimental_df[column_name].map(convert_function)

	offline_df.to_csv(output_obs_data_filename, index=False)
	experimental_df.to_csv(output_exp_data_filename, index=False)


class DisjointLinUCB:
	'''
		v2: linUCB with disjoint parameters for different arm_idx
		this is useful for the yahoo news recommendation case
	'''
	def __init__(self, N_arm, dimension, synthetic=True, beta=1,\
			articles_filename='data/_yahoo-webscope-articles.txt'):
		self.article2idx, self.articles_feature, n_articles = init_articles(articles_filename)

		self.V = np.zeros((n_articles, dimension, dimension))
		self.V_inv = np.zeros((n_articles, dimension, dimension))
		self.b = np.zeros((n_articles, dimension))
		self.theta_hat = np.zeros((n_articles, dimension))
		self.beta = beta

		self.has_been_drawn = [False] * n_articles # indicating whether some arm has already been drawn

		for i in range(n_articles):
			self.V[i] = np.identity(dimension)

		#self.drawn_arm = None # the recent drawn arm

		self.synthetic = synthetic
		self.contextual = True

	
	def draw_arm(self, user_features, choices, t, option='offline_online'):
		max_UCB = -np.inf
		max_arm = None # the arm number starts from 0
		choices_index = []
		for choice in choices:
			if not np.isnan(choice):
				choices_index.append( self.article2idx[choice] )

		# added on 2-2, randomly pick one of the not drawn arms
		for arm_idx in choices_index:
			if self.has_been_drawn[arm_idx] == False:
				self.has_been_drawn[arm_idx] = True
				return arm_idx

		# print (choices_index)
		for arm in range(len(choices_index)):
			a_vec = np.asarray(user_features) # TODO: extract the features
			arm_idx = choices_index[arm]
			a_mean = a_vec.dot(self.theta_hat[arm_idx])
			if option == 'only_offline':
				a_UCB = a_mean
			else:
				a_UCB = a_mean + np.sqrt(self.beta) * np.sqrt( a_vec.T.dot(self.V_inv[arm_idx]).dot(a_vec) )

			if a_UCB > max_UCB:
				max_arm = arm
				max_UCB = a_UCB
		return choices_index[max_arm] # we only return the index of the max_arm

	def update(self, user_features, arm_idx, reward, is_online):
		# different from the updating for the sharing parameter case
		a_vec = np.asarray(user_features)
		self.V[arm_idx] += np.outer(a_vec, a_vec.T)
		self.V_inv[arm_idx] = np.linalg.inv(self.V[arm_idx])
		self.b[arm_idx] += reward * a_vec
		self.theta_hat[arm_idx] = self.V_inv[arm_idx].dot(self.b[arm_idx])


class LinUCB:
	'''
		v1: linUCB with shared parameters
	'''
	def __init__(self, N_arm, dimension, synthetic=True, beta=1):
		# d is the number of dimensions
		# beta is the parameter controloing the confidence bound
		self.dimension = dimension# the number of dimensions of features as input to a linear function
		self.N_arm = N_arm
		self.V = np.identity(dimension)
		self.b = np.zeros(dimension)
		self.theta_hat = np.zeros(dimension) # this is a shared parameter
		self.beta = beta
		self.synthetic = synthetic

		self.contextual = True

	def draw_arm(self, context_vec, t, option='offline_online'):
		V_inv = np.linalg.inv(self.V)
		# print ('V', self.V)
		# print ('V_inv:', V_inv)
		self.theta_hat = V_inv.dot(self.b)
		max_UCB = -np.inf
		max_arm_idx = None
		for arm_idx in range(self.N_arm):
			a_vec = arm_context_feature(context_vec, arm_idx, self.synthetic)
			#print ('theta_hat:', self.theta_hat)
			a_mean = a_vec.dot(self.theta_hat)
			if option == 'only_offline':
				a_UCB = a_mean
			else:
				a_UCB = a_mean + np.sqrt(self.beta) * np.sqrt( a_vec.T.dot(V_inv).dot(a_vec) )
			#print ('arm:', arm_idx, '\t', 'a_mean:', a_mean, '\t', 'a_UCB:', a_UCB)
			if a_UCB > max_UCB:
				max_arm_idx = arm_idx 
				max_UCB = a_UCB

		return max_arm_idx		

	def update(self, context_vec, arm_idx, reward, is_online):
		# added on 9-13: is_online - the update function to distinguish the online feedback from the offline data
		a_vec = arm_context_feature(context_vec, arm_idx, self.synthetic)
		self.V += np.outer(a_vec, a_vec.T)
		self.b += reward * a_vec


if __name__ == '__main__':
	convert_index()

	
