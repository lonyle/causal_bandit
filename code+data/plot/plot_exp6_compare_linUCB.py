import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

linestyle_vec = [(2, 0.01), (2,4), (4,2)]
color_vec = ['green', 'black', 'red']

def plot(ax, marker=None):
	idx = 0
	for option in ['offline_online', 'only_offline', 'only_online']:
	#for option in ['only_offline']:
		filename = 'data/result_exp6_' + option + '.json'
		plot_utils.plot_cumulative_regret(ax, filename, option, linestyle_vec[idx], color_vec[idx], marker)
		idx += 1

if __name__ == '__main__':
	fig = plt.figure()
	ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])

	plot(ax)

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp6_compare_linUCB.eps', dpi=500)
	plt.savefig('images/exp6_compare_linUCB.png', dpi=500)
	plt.show()
