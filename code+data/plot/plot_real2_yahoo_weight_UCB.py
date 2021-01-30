import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	linestyle_vec = [(2, 0.01), (2,4), (5,5)]
	color_vec = ['green', 'blue', 'red']
	marker_vec = [None, None, '^']

	fig = plt.figure()
	ax = fig.add_axes([0.22, 0.16, 0.76, 0.8])
	#ax = fig.add_axes([0.19, 0.16, 0.77, 0.8])

	idx = 0

	for option in ['offline_online', 'only_offline', 'only_online']:
	#for option in ['offline_online', 'only_online']:
		filename = 'data/result_real2_tmp_yahoo_' + option + '.json'
		if option == 'only_offline':
			filename = 'data/result_real2_yahoo_' + option + '.json'
		if option == 'offline_online':
			option = 'offline+online'
		plot_utils.plot_cumulative_regret(ax, filename, option, linestyle_vec[idx], color_vec[idx],\
				xlimit=200000, max_reward=0.041, marker=marker_vec[idx], marker_every=10000)
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.ylim(0, 1300)
	plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
	#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

	plt.savefig('images/real2_yahoo_weight_UCB.eps', dpi=500)
	plt.show()
