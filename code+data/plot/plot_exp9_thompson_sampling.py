# plot the performance of our method, and thompson sampling (beta-distribution)

import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	marker_vec = ['x', 'o', '^', '.']
	color_vec = ['green', 'blue', 'red', 'black']
	label_vec = ['IPSW+TS (ours)', 'average+TS', 'linear regr.+TS', 'xgboost+TS']

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.16, 0.8, 0.8])

	idx = 0
	for alg_name in ['IPSW_TS', 'historic_average:binary', 'linear_regression:binary', 'xgboost:binary']:
		filename = 'data/result_exp9_' + alg_name + '_' + 'offline_online' + '.json'

		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], (2, 0.01), \
			color=color_vec[idx], marker=marker_vec[idx])
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.ylim(0, 8)

	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp9_supervised_TS.eps', dpi=500)
	plt.show()