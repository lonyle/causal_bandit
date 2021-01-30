'''
	pre-process the logged data of yahoo news recommendation
	we apply a square function on all the contexts, to generate both observational and experimental data file
'''

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
import linUCB

def output_obs_exp_data(input_filename, output_obs_filename, output_exp_filename, 
						map_function):
	'''
		non-linear mapping
	'''
	column_names = ["timestamp", "context1", "context2", "context3", "context4", "context5", "context6", \
		"drawn_article", "reward", "article1", "article2", "article3", "article4", "article5", \
		"article6", "article7", "article8", "article9", "article10", \
		"article11", "article12", "article13", "article14", "article15", \
		"article16", "article17", "article18", "article19", "article20", "article21"]

	df = pd.read_csv(input_filename, sep=" ", usecols=range(30), header=None, names=column_names)
	
	for column_name in ["context1", "context2", "context3", "context4", "context5", "context6"]:
		df[column_name] = df[column_name].apply(map_function)

	shuffled_df = df.sample(frac=1.0,random_state=200) #random state is a seed value
	obs_df = shuffled_df.iloc[:20000, :]
	exp_df = shuffled_df.iloc[20000:, :]

	obs_df.to_csv(output_obs_filename, index=False)
	exp_df.to_csv(output_exp_filename, index=False)

	return obs_df, exp_df

if __name__ == '__main__':
	input_filename = 'data/_yahoo-webscope-logs.txt'
	output_obs_filename = 'data/_yahoo_nonlinear_obs_20000.csv'
	output_exp_filename = 'data/_yahoo_nonlinear_exp_80000.csv'
	map_function = lambda x: np.exp(10*x)
	output_obs_exp_data(input_filename, output_obs_filename, output_exp_filename, map_function)

	linUCB.convert_index(exp_data_filename = output_exp_filename,
		obs_data_filename = output_obs_filename,
		articles_filename='data/_yahoo-webscope-articles.txt',
		output_exp_data_filename='data/_yahoo_nonlinear_reindex_exp_80000.csv',
		output_obs_data_filename='data/_yahoo_nonlinear_reindex_obs_20000.csv')