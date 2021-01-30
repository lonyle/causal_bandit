import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	marker_vec = ['x', 'o', '^', '.', 'D']
	linestyle_vec = [(2, 0.01), (2,4), (4,2)]
	color_vec = ['green', 'blue', 'red', 'black', 'purple']
	label_vec = ['IPSW+UCB (ours)', 'average+UCB', 'linear regr.+UCB', 'xgboost+UCB', 'SDB']

	fig = plt.figure()
	ax = fig.add_axes([0.19, 0.16, 0.77, 0.8])

	idx = 0
	for alg_name in ['IPSW', 'historic_average', 'linear_regression', 'xgboost']:
	#for option in ['only_offline']:
		filename = 'data/result_exp16_' + alg_name + '_' + 'offline_online' + '.json'
		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], (2,0.01),\
			color=color_vec[idx], marker=marker_vec[idx], marker_every=10000)
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	#plt.ylim(0, 750)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.savefig('images/exp16_yahoo_supervised_synthetic.eps', dpi=500)
	plt.savefig('images/exp16_yahoo_supervised_synthetic.png', dpi=500)
	plt.show()