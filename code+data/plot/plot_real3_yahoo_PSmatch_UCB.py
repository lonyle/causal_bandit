import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	linestyle_vec = [(2, 0.01), (2,4), (4,2)]
	color_vec = ['green', 'blue', 'red']

	fig = plt.figure()
	ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])

	idx = 0

	for option in ['offline_online', 'only_offline', 'only_online']:
	#for option in ['only_offline']:
		filename = 'data/result_real3_yahoo_' + option + '.json'
		plot_utils.plot_cumulative_regret(ax, filename, option, linestyle_vec[idx], color_vec[idx])
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative reward', weight='bold')
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.ylim(0, 40)
	plt.savefig('images/real3_yahoo_PSmatch_UCB.eps', dpi=500)
	plt.show()
