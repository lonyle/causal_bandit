# fill the missed fields: "propensity score"
# use a certain propensity score to get offline data that strictly satisfy the propensity
# hence the unconfoundedness holds, because propensity score is from all known contexts

import pandas as pd
import numpy as np
import xgboost as xgb

def get_features(data, feature_colnames):
    return data.loc[:, feature_colnames]

def predict_propensity_score(input_df, context_names, treatment_name, outcome_name):
	''' predict the propensity score for each context
	'''
	# we split the data into training and validation set
	train_df, validation_df = np.split(input_df.sample(frac=1), [int(1*len(input_df))])
	if not context_names: # use default names
		context_names = list(input_df).remove(treatment_name).remove(outcome_name)
	X_train = get_features(train_df, context_names)
	X_validation = get_features(validation_df, context_names)

	y_train = train_df[treatment_name]
	y_validation = validation_df[treatment_name]

	train_data = xgb.DMatrix(X_train, label=y_train)

	param = {'objective': 'binary:logistic'}
	num_round = 100
	bst = xgb.train(param, train_data, num_round)

	predictions = bst.predict(train_data)

	return predictions

def output_propensity_score_column(input_filename, context_names, treatment_name, outcome_name):
	input_df = pd.read_csv(input_filename)
	predicted_ps = predict_propensity_score(input_df, context_names, treatment_name, outcome_name)
	input_df['propensity_score'] = predicted_ps

	output_df = input_df.sample(frac=1).reset_index(drop=True)

	output_df.to_csv(input_filename[:-4] + '_ps.csv', index=False)

def generate_uncounfouded_offline_data(input_df, treatment_name):
	''' use the predicted propensity score to get offline data
		with probability specified by propensity score, select an action, retain the data row only if the action is matched
	'''
	retained_indices = []
	for index, row in input_df.iterrows():
		propensity_score = row['propensity_score']
		if np.random.random() < propensity_score: # binary treatment, the prob for treatment to be 1
			treatment = 1
		else:
			treatment = 0
		if treatment == row[treatment_name]:
			retained_indices.append(index)

	retained_df = input_df.iloc[retained_indices, :]
	return retained_df

def output_uncounfouded_offline_data(input_filename, treatment_name):
	input_df = pd.read_csv(input_filename)
	retained_df = generate_uncounfouded_offline_data(input_df, treatment_name)
	retained_df.to_csv(input_filename[:-4] + '_unconfounded.csv', index=False)


if __name__ == '__main__':
	context_names = ['age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75']
	treatment_name = 'treat'
	outcome_name = 're78'
	input_filename = 'data/lalonde.csv'
	
	output_propensity_score_column(input_filename, context_names, treatment_name, outcome_name)
	
	output_uncounfouded_offline_data(input_filename[:-4] + '_ps.csv', treatment_name)




