import sys
sys.path.append('.')
import matplotlib.pyplot as plt
import numpy as np

import plot_utils

marker_vec = ['x', 'o']
color_vec = ['green', 'red']
label_vec = ['ours', 'batch']
line_vec = [(2, 0.01), (1, 1)]

def plot_regret():
	fig = plt.figure()
	ax = fig.add_axes([0.22, 0.16, 0.75, 0.8])

	option = 'offline_online'
	N_offline = 100

	idx = 0
	for batch_mode in [False, True]:
		if batch_mode:
			filename = 'data/result_real13_yahoo_batch_mode' + option + '.json'
		else:			
			filename = 'data/result_real2_tmp_yahoo_' + option + '.json'

		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], \
			(2, 0.01), color=color_vec[idx], marker=marker_vec[idx], \
			max_reward=0.041, marker_every=10000)

		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')

	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	plt.savefig('images/real13_yahoo_batch_mode.eps', dpi=500)
	plt.show()


if __name__ == '__main__':
	plot_regret()