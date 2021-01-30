import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	linestyle_vec = [(2, 0.01), (2,4), (4,2)]
	color_vec = ['green', 'black', 'red']
	label_vec = ['offline+online', 'only_offline', 'only_online']

	fig = plt.figure()
	ax = fig.add_axes([0.19, 0.16, 0.77, 0.8])

	idx = 0

	for option in ['offline_online', 'only_offline', 'only_online']:
	#for option in ['only_offline']:
		filename = 'data/result_exp15_' + option + '.json'
		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], \
				linestyle_vec[idx], color_vec[idx], xlimit=100000)
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.ylim(0, 750)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.savefig('images/exp15_yahoo_synthetic.eps', dpi=500)
	plt.savefig('images/exp15_yahoo_synthetic.png', dpi=500)
	plt.show()