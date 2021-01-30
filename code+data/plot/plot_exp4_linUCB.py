import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	linestyle_vec = [(2, 0.01), (2,4), (4,2)]
	color_vec = ['green', 'black', 'red']
	label_vec = ['offline+online', 'only_offline', 'only_online']

	fig = plt.figure()
	ax = fig.add_axes([0.13, 0.16, 0.82, 0.8])

	idx = 0
	for option in ['offline_online', 'only_offline', 'only_online']:
	#for option in ['only_offline']:
		filename = 'data/result_exp4_' + option + '.json'
		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], \
			linestyle_vec[idx], color_vec[idx], confidence_interval=True)
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.ylim(0, 9.7)
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp4_linUCB.eps', dpi=500)
	plt.savefig('images/exp4_linUCB.png', dpi=500)
	plt.show()