# compare our method and other supervised learning algorithms

import sys
sys.path.append('.')
import matplotlib.pyplot as plt 

import plot_utils

if __name__ == '__main__':
	marker_vec = [None, 'x', 'o', '^', '.']
	color_vec = ['lightgreen', 'green', 'blue', 'red', 'black']
	label_vec = ['UCB+IPSW (ours)', 'TS+IPSW (ours)', 'TS+average', 'TS+linear regr.', 'TS+xgboost']

	fig = plt.figure()
	ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])

	idx = 0
	for alg_name in ['IPSW+UCB', 'IPSW', 'historic_average', 'linear_regression', 'xgboost']:
		if alg_name == 'IPSW+UCB':
			filename = 'data/result_exp8_' + 'IPSW' + '_' + 'offline_online'+'.json'
		else:
			filename = 'data/result_exp91_' + alg_name + '_' + 'offline_online'+'.json'
		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], (2,0.01),\
		 color=color_vec[idx], marker=marker_vec[idx], marker_every=50)
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.ylim(0, 80)

	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp91_TS_continuous.eps', dpi=500)
	plt.show()