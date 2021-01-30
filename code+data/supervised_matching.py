# instead of other matching machines, we use supervised learning to do warm-start
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression

def get_features(data, feature_colnames):
	return data.loc[:, feature_colnames]

class SupervisedMatching:
	# as the first version, we use supervised learning (e.g. xgboost) to train 
	# two different models for action=0 or action=1

	def __init__(self, offline_data, N_arm, model_name='xgboost', context_names=None,\
			treatment_name='action', outcome_name='reward', bias=0, choice_names=None):

		# the offline_data should have the same format as environment.generate_offline_data
		self.N_arm = N_arm
		self.offline_data = offline_data
		self.bias = bias
		self.model_name = model_name

		self.treatment_name = treatment_name
		self.outcome_name = outcome_name
		self.choice_names = choice_names

		if context_names:
			self.context_names = context_names
		else:
			names = list(offline_data.keys())
			names.remove(self.treatment_name)
			names.remove(self.outcome_name)
			names.remove('propensity_score')
			self.context_names = names

		if self.model_name.split(':')[0] == 'xgboost':
			self.train_model_xgboost()
		elif self.model_name.split(':')[0] == 'linear_regression':
			self.train_model_linear_regression()
		elif self.model_name.split(':')[0] == 'historic_average':
			self.train_model_average() # output the average outcome, only for context-independent
		else:
			print (self.model_name.split(':'))
			print ('unknown name!')

	def train_model_xgboost(self):
		# prepare the data as the features and labels
		# split the data according to the actions, 0 or 1. Train two different models for different actions
		
		#print (self.offline_data)
		#data_filename = 'wechat/data/obs1_100000_redpoint1_ps.csv'
		#input_df = pd.read_csv(data_filename)
		input_df = pd.DataFrame.from_dict(self.offline_data, orient='index').transpose()

		self.model_vec = []
		self.data_count_vec = []
		for action in range(self.N_arm): # TODO: change it to multiple actions
			action_df = input_df[input_df[self.treatment_name] == action ]
			if action_df.shape[0] == 0:
				self.model_vec.append(None)
				self.data_count_vec.append(0)
				#print ('action with no data:', action)
				continue

			train_df, validation_df = np.split(action_df.sample(frac=1), [int(0.8*len(action_df))])
			X_train = get_features(train_df, self.context_names)
			y_train = train_df[self.outcome_name]

			train_data = xgb.DMatrix(X_train, label=y_train)

			param = {'objective': 'reg:squarederror', 'max_depth': 6}
			num_round = 100
			bgt = xgb.train(param, train_data, num_round)
			#print (len(y_train), y_train)
			self.data_count_vec.append(len(y_train))
			self.model_vec.append(bgt)
		return self.model_vec

	def train_model_linear_regression(self):
		input_df = pd.DataFrame.from_dict(self.offline_data, orient='index').transpose()

		self.model_vec = []
		self.data_count_vec = []
		for action in range(self.N_arm):
			action_df = input_df[input_df[self.treatment_name] == action ]
			if action_df.shape[0] == 0:
				self.model_vec.append(None)
				self.data_count_vec.append(0)
				#print ('action with no data:', action)
				continue

			train_df, validation_df = np.split(action_df.sample(frac=1), [int(0.8*len(action_df))])
			X_train = get_features(train_df, self.context_names)
			y_train = train_df[self.outcome_name]

			reg = LinearRegression().fit(X_train, y_train)
			self.data_count_vec.append(len(y_train))
			self.model_vec.append(reg)
		return self.model_vec

	def train_model_average(self):
		input_df = pd.DataFrame.from_dict(self.offline_data, orient='index').transpose()

		self.average_vec = []
		self.data_count_vec = []
		for action in range(self.N_arm):
			action_df = input_df[input_df[self.treatment_name] == action ]			
			y_train = action_df[self.outcome_name]
			average = y_train.mean() # modified because of NaN values
			self.data_count_vec.append(len(y_train))
			self.average_vec.append(average)
		return self.average_vec		

	def find_sample_reward(self, context_vec, action):
		if self.data_count_vec[action] <= 0:
			return False
		# predict the reward 
		if self.model_name.split(':')[0] == 'xgboost':
			X = xgb.DMatrix(np.array(context_vec).reshape((1,-1)), feature_names=self.context_names)
			model = self.model_vec[action]
			reward = model.predict(X)
		elif self.model_name.split(':')[0] == 'linear_regression':
			X = np.array(context_vec).reshape((1,-1))
			model = self.model_vec[action]
			reward = model.predict(X)
		elif self.model_name.split(':')[0] == 'historic_average':
			reward = self.average_vec[action]
		else:
			print ('unknown name')

		if len(self.model_name.split(':')) == 1:
			self.data_count_vec[action] -= 1
			return reward
		elif self.model_name.split(':')[1] == 'binary':
			prob = reward
			self.data_count_vec[action] -= 1
			if np.random.random() < prob:
				return 1
			else:
				return 0
		else:
			print ('unknown parameter')


	def get_pending_action(self, context, update_pending):
		# currently, we do not support pending action
		return False

def test_supervised_matching():
	import environment

	linear_env = environment.Environment(4, 2, environment.linear_function) # dimension=4, N_arm=2
	N_sample = 5000
	propensity_score_setting = {
		"prob_vec": [0.25]*4,
		"propensity_score_vecs": [
			[0.2, 0.8],
			[0.4, 0.6],
			[0.6, 0.4],
			[0.8, 0.2]
		]
	}
	offline_data = linear_env.generate_offline_data(N_sample, propensity_score_setting)

	supervised_matcher = SupervisedMatching(offline_data, model_name='xgboost')


if __name__ == '__main__':
	test_supervised_matching()




