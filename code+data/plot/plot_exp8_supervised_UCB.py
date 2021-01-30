# compare our method and other supervised learning algorithms

import sys
sys.path.append('.')
import matplotlib.pyplot as plt 

import plot_utils

if __name__ == '__main__':
	marker_vec = ['^', 'x', 'o', '.', 'D']
	color_vec = ['red', 'green', 'blue', 'black', 'purple']
	label_vec = ['UCB+linear regr.', 'UCB+IPSW (ours)', 'UCB+average', 'UCB+xgboost', 'SDB']

	fig = plt.figure()
	ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])

	idx = 0
	for alg_name in ['linear_regression', 'IPSW', 'historic_average', 'xgboost', 'SDB']:
		if alg_name == 'SDB':
			filename = 'data/result_exp10_'+'only_online'+'.json'
		else:
			filename = 'data/result_exp8_' + alg_name + '_' + 'offline_online'+'.json'
		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], (2,0.01),\
		 color=color_vec[idx], marker=marker_vec[idx], marker_every=50)
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.ylim(0, 80)

	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp8_supervised_UCB.eps', dpi=500)
	plt.show()