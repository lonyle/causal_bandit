import sys
sys.path.append('.')
import matplotlib.pyplot as plt
import numpy as np
import json

import plot_utils

if __name__ == '__main__':
	fig = plt.figure()
	ax = fig.add_axes([0.17, 0.16, 0.8, 0.8])

	context_dim = 8

	cumulative_reward_vec = []
	for num_unobserved_confounder in range(context_dim):
		filename = 'data/result_exp14_offline_online_hidden_dim'+str(num_unobserved_confounder)+'.json'
		count = 0
		for line in open(filename):
			cumulative_regret_vec = json.loads(line)
			#print ("length:", len(cumulative_regret_vec))
			if count == 0:
				sum_cumulative_regret_vec = np.asarray(cumulative_regret_vec)
			else:
				sum_cumulative_regret_vec += np.asarray(cumulative_regret_vec)
			count += 1

		average_cumulative_regret_vec = sum_cumulative_regret_vec / count

		cumulative_reward_vec.append(average_cumulative_regret_vec[-1])

	plt.plot(np.arange(context_dim), cumulative_reward_vec, color='black', linewidth=3)

	plt.xlabel('num. unobserved confounders', weight='bold', fontsize=22)
	plt.ylabel('cum. regret at $T=100$', weight='bold', fontsize=22)
	plt.savefig('images/exp14_unobserved_confounder.eps', dpi=500)
	plt.show()