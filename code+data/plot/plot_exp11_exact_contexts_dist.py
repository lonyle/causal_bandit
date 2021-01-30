import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

def plot_N_offline(ax, N_offline):
	marker_vec = ['x', 'o']
	color_vec = ['green', 'red']
	label_vec = ['empirical context dist.', 'true context dist.']

	idx = 0
	for true_context_distribution in [False, True]:
		filename = 'data/result_exp11_' + option + '_' + str(true_context_distribution) + '_'+str(N_offline) + '.json'

		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], (2, 0.01), \
			color=color_vec[idx], marker=marker_vec[idx], marker_every=200)

		idx += 1


if __name__ == '__main__':	
	fig = plt.figure()
	ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])

	option = 'offline_online'
	
	plot_N_offline(ax, 10)
	ax.text(1500, 29.7, "$N=10$", size=20)

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')

	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.ylim(0, 50)

	plot_N_offline(ax, 50)
	ax.text(1500, 16.4, "$N=50$", size=20)
	plot_N_offline(ax, 100)
	ax.text(1500, 7.5, "$N=100$", size=20)

	plt.savefig('images/exp11_exact_contexts_dist.eps', dpi=500)
	plt.show()