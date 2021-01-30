''' compare the four methods
'''

import numpy as np

N_SAMPLE = 50 # offline
N_ROUND = 10000 # online

class ThompsonSampling:
	def __init__(self, K):
		self.K = K
		self.count_vec = [[1, 1]] * K
	def draw_arm(self, t):
		random_reward_vec = [None] * self.K
		for i in range(self.K):
			random_reward_vec[i] = np.random.beta(self.count_vec[i][1], self.count_vec[i][0])

		return np.argmax(random_reward_vec)

	def update(self, i, reward):
		self.count_vec[i][reward] += 1

class UCB:
	# K: the number of arms
	def __init__(self, K):
		self.cum_vec = [0] * K
		self.count_vec = [0] * K
		self.ave_vec = [None] * K
		self.K = K

	def draw_arm(self, t):
		for i in range(self.K):
			if self.count_vec[i] == 0:
				return i
		# choose the arm with the upper confidence bound
		confidence_bound_vec = np.asarray(self.ave_vec)
		for i in range(self.K):
			confidence_bound_vec[i] += np.sqrt(0.01*np.log(t)/self.count_vec[i])
		i = np.argmax( confidence_bound_vec )
		return i


	def update(self, i, reward):
		# update the information according to the feedbacks
		self.cum_vec[i] += reward
		self.count_vec[i] += 1
		self.ave_vec[i] = self.cum_vec[i]/self.count_vec[i]


def run_online(agent, t0):
	# run the online algorithm
	psedo_regret = 0
	for t in range(t0, t0 + N_ROUND):
		i = agent.draw_arm(t)
		if i == 0: # video below
			reward = np.random.binomial(1, 0.06)
		if i == 1: # no video below
			reward = np.random.binomial(1, 0.09)
		agent.update(i, reward)
		psedo_regret += (0 if i == 1 else 0.03)

	#print ('psedo_regret in first 100 rounds:', psedo_regret)
	return psedo_regret


def pure_offline():
	# two Gaussian distribution, one N(0,1), another N(1,1). 
	# Calculate the probability of making a error

	MonteCarlo_round = 10000
	error_count = 0
	for i in range(MonteCarlo_round):
		N_sample = N_SAMPLE

		samples_0 = np.concatenate( (np.random.binomial(1, 0.11, N_sample), np.random.binomial(1, 0.01, N_sample) ))
		samples_1 = np.concatenate( (np.random.binomial(1, 0.14, N_sample), np.random.binomial(1, 0.04, N_sample) ))

		if np.average(samples_0) > np.average(samples_1):
			error_count += 1

	error_rate = error_count / MonteCarlo_round
	print ('pure offline error_rate (causal inference):', error_rate)
	print ('error_rate to regret:', error_rate * N_ROUND * 0.03, \
		'std:', np.sqrt(error_rate*(1-error_rate)) * N_ROUND * 0.03 )


def pure_online():
	MonteCarlo_round = 100
	psedo_regret_vec = []
	for i in range(MonteCarlo_round):
		# only use the UCB algorithm
		ucb_agent = UCB(K=2)
		psedo_regret = run_online(ucb_agent, 0)
		psedo_regret_vec.append(psedo_regret)
	print ('the average psedo_regret for pure online:', np.average(psedo_regret_vec), 'std:', np.std(psedo_regret_vec))

def online_ABtest():
	MonteCarlo_round = 10000
	N_sample = 2000#int(N_SAMPLE * 5)

	psedo_regret_vec = []
	error_count = 0
	for i in range(MonteCarlo_round):
		psedo_regret = 0.03 * N_sample
		samples_0 = np.random.binomial(1, 0.06, N_sample)
		samples_1 = np.random.binomial(1, 0.09, N_sample)

		if np.average(samples_0) > np.average(samples_1):
			psedo_regret += (N_ROUND - 2*N_sample) * 0.03
			error_count += 1

		psedo_regret_vec.append(psedo_regret)
	print ('error rate of ABtest:', error_count / MonteCarlo_round)
	print ('the average psedo_regret for online ABtest:', np.average(psedo_regret_vec), 'std:', np.std(psedo_regret_vec))

def wrong_initialization_offline():
	# empirical average. We do not generate the data rows and then calculate the averages from the data
	# instead, we generate the samples from each of the sub-populations
	MonteCarlo_round = 10000
	N_sample = int(N_SAMPLE)

	psedo_regret_vec = []
	for i in range(MonteCarlo_round):
		psedo_regret = 0
		samples_0 = np.concatenate( (np.random.binomial(1, 0.11, N_sample*3), np.random.binomial(1, 0.01, N_sample) ))
		samples_1 = np.concatenate( (np.random.binomial(1, 0.14, N_sample), np.random.binomial(1, 0.04, N_sample*3) ))

		if np.average(samples_0) > np.average(samples_1):
			psedo_regret += (N_ROUND) * 0.03

		psedo_regret_vec.append(psedo_regret)
	print ('the average psedo_regret for empirical mean:', np.average(psedo_regret_vec), 'std:', np.std(psedo_regret_vec))


def combined():
	MonteCarlo_round = 100
	psedo_regret_vec = []
	for i in range(MonteCarlo_round):
		# init
		N_sample = N_SAMPLE

		samples_0 = np.concatenate( (np.random.binomial(1, 0.11, N_sample), np.random.binomial(1, 0.01, N_sample) ))
		samples_1 = np.concatenate( (np.random.binomial(1, 0.14, N_sample), np.random.binomial(1, 0.04, N_sample) ))

		ucb_agent = UCB(K=2)
		ucb_agent.count_vec = [N_sample, N_sample]
		ucb_agent.cum_vec = [sum(samples_0), sum(samples_1)]
		ucb_agent.ave_vec = [np.average(samples_0), np.average(samples_1)]

	
		psedo_regret = run_online(ucb_agent, len(samples_0)+len(samples_1))
		psedo_regret_vec.append(psedo_regret)
	print ('the average psedo_regret for combined:', np.average(psedo_regret_vec), 'std:', np.std(psedo_regret_vec))
	
def combined_thompson():
	MonteCarlo_round = 100
	psedo_regret_vec = []
	for i in range(MonteCarlo_round):
		N_sample = N_SAMPLE
		ts_agent = ThompsonSampling(K=2)

		samples_0 = np.concatenate( (np.random.binomial(1, 0.11, N_sample), np.random.binomial(1, 0.01, N_sample) ))
		samples_1 = np.concatenate( (np.random.binomial(1, 0.14, N_sample), np.random.binomial(1, 0.04, N_sample) ))
		for reward in samples_0:
			ts_agent.update(0, reward)
		for reward in samples_1:
			ts_agent.update(1, reward)

		psedo_regret = run_online(ts_agent, len(samples_0)+len(samples_1))
		psedo_regret_vec.append(psedo_regret)
	print ('the average psedo_regret for combined (TS):', np.average(psedo_regret_vec), 'std:', np.std(psedo_regret_vec))


def wrong_initilization_online():
	# it takes the average of all the offline data points
	MonteCarlo_round = 1000
	psedo_regret_vec = []
	for i in range(MonteCarlo_round):
		# init
		N_sample = N_SAMPLE

		samples_0 = np.concatenate( (np.random.binomial(1, 0.11, N_sample*3), np.random.binomial(1, 0.01, N_sample) ))
		samples_1 = np.concatenate( (np.random.binomial(1, 0.14, N_sample), np.random.binomial(1, 0.04, N_sample*3) ))

		ucb_agent = UCB(K=2)
		ucb_agent.count_vec = [N_sample, N_sample]
		ucb_agent.cum_vec = [sum(samples_0), sum(samples_1)]
		ucb_agent.ave_vec = [np.average(samples_0), np.average(samples_1)]

	
		psedo_regret = run_online(ucb_agent, len(samples_0)+len(samples_1))
		psedo_regret_vec.append(psedo_regret)
	print ('the average psedo_regret for wrong_initilization:', np.average(psedo_regret_vec))

def supervised_learning():
	# first generate the data, and then fit the generated data with xgboost or linear regression
	# we have three variables: A, B and C. A (videos above ad.) is very predictive on B (click rate)
	# or, preferences on videos and video-views have co-linearity. video-above-ad and image-above-ad have co-linearity

if __name__ == '__main__':
	pure_offline()
	online_ABtest()
	wrong_initialization_offline()
	#pure_online()
	combined()
	#combined_thompson()
	#wrong_initilization()








