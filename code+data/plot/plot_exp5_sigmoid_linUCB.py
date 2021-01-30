import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	linestyle_vec = [(1,2), (2,1), (2, 0.01)]

	fig = plt.figure()
	ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])

	idx = 0
	for option in ['only_offline', 'only_online', 'offline_online']:
	#for option in ['only_offline']:
		filename = 'data/result_exp5_' + option + '.json'
		plot_utils.plot_cumulative_regret(ax, filename, option, linestyle_vec[idx])
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cummulative regret', weight='bold')
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp5_sigmoid_linUCB.eps', dpi=500)
	plt.show()
