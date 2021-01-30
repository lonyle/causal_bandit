# generate the yahoo data with certain propensity score

import sys
sys.path.insert(0, 'plot')
sys.path.append('.')

import numpy as np
import pandas as pd
import json

import linUCB
import average_reward_yahoo
ave_reward = average_reward_yahoo.average_reward_vec
pscore_correlation_action = average_reward_yahoo.pscore_correlation_action

print (pscore_correlation_action)

choice_names = [
		"article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"]

pscore_clusters = [0.3, 0.4, 0.5, 0.6, 0.7] # this is the setting of pscore for different clusters of items

def context2pscore(context_vec):
	# map the context vector to pscore
	# each dimension of the context corresponds to one cluster
	pscore = np.dot(pscore_clusters, context_vec)
	if pscore > 1:
		print ('pscore > 1!!', pscore)
		pscore = 1
	if pscore < 0:
		print ('pscore < 0!!', pscore)
		pscore = 0
	return pscore

def generate_offline_data_pscore_correlation(df, output_filename, \
		articles_filename, max_pscore=0.1):
	context_names=['context2', 'context3', 'context4', 'context5', 'context6']
	theta_hat_vec = json.load(open('data/_yahoo_linear_model_articles.json'))
	article2idx, _, n_article = linUCB.init_articles(articles_filename)

	## for simplicity, if high prob, pscore=0.1 (keep_prob=1), if low prob 0.02 prob (keep_prob=0.2)
	pscore_vec = []
	index_to_delete = []
	for index, row in df.iterrows():
		context_vec = []
		for context_name in context_names:
			context_vec.append(row[context_name])

		article = row['drawn_article']
		action = int(article)#article2idx[article]
		#print (action)

		## only consider the chosen action
		reward = row['reward']
		if (reward - ave_reward[action]) * pscore_correlation_action[action] < 0:
			pscore = 0.2 # high prob
		else:
			pscore = 0.02 # low prob

		#print (action, reward, (reward - ave_reward[action]) * pscore_correlation_action[action])

		pscore_vec.append(pscore)

		prob_to_keep = pscore / max_pscore
		if np.random.random() < 1-prob_to_keep:
			index_to_delete.append(index)	

		# get the pscore for each action
		# choices = [ article2idx[row[choice_name]] for choice_name in choice_names ]
		# action_idx_in_choices = choices.index(action)
		# weight_vec = []
		# for action in choices:
		# 	predicted_reward = np.asarray(context_vec).dot(theta_hat_vec[action])
		# 	exponent = (predicted_reward-ave_reward[action]) * pscore_correlation_action[action] * correlation
		# 	weight_vec.append(np.exp(exponent))
		# sum_weight = np.sum(weight_vec)
		# a_prob_vec = [weight / sum_weight for weight in weight_vec]
		# pscore = a_prob_vec[action_idx_in_choices]		
	
	df['propensity_score'] = pscore_vec
	df.drop(df.index[index_to_delete], inplace=True)
	print (len(index_to_delete))
	r, c = df.shape
	print ('the number of remaining rows:', r)
	df.to_csv(output_filename)


def generate_offline_data_pscore_uniform(df, output_filename,
			context_names=['context2', 'context3', 'context4', 'context5', 'context6']):
	
	r, c = df.shape
	pscore_vec = [0.05] * r
	df['propensity_score'] = pscore_vec
	df.to_csv(output_filename)


def generate_offline_data_pscore(df, output_filename,
			context_names=['context2', 'context3', 'context4', 'context5', 'context6']):
	index_to_delete = []
	pscore_vec = []
	for index, row in df.iterrows():
		context_vec = []
		for context_name in context_names:
			context_vec.append(row[context_name])
		pscore = context2pscore(context_vec)

		pscore_vec.append(pscore)

	max_pscore = max(pscore_vec)
	print ('max_pscore:', max_pscore)

	df['propensity_score'] = pscore_vec

	for index, row in df.iterrows():
		prob_to_keep = pscore / max_pscore # in order to better ultilize the samples
		if np.random.random() < 1 - prob_to_keep:
			index_to_delete.append(index)

	df.drop(df.index[index_to_delete], inplace=True)

	df.to_csv(output_filename)


if __name__ == "__main__":
	df = pd.read_csv('data/_yahoo_reindex_exp_80000.csv')
	output_filename = 'data/_yahoo_pscore_from_20000_correlation.csv'
	articles_filename = 'data/_yahoo-webscope-articles.txt'

	#generate_offline_data_pscore(df, output_filename)
	#generate_offline_data_pscore_uniform(df, output_filename)

	generate_offline_data_pscore_correlation(df, output_filename, articles_filename)







