import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

repeat_times = 50

if __name__ == '__main__':
	linestyle_vec = [(2, 0.01), (2,4), (4,2)]
	color_vec = ['green', 'blue', 'red']
	label_vec = ['offline+online', 'only_offline', 'only_online']

	fig = plt.figure()
	ax = fig.add_axes([0.19, 0.16, 0.79, 0.8])

	idx = 0
	for option in ['offline_online', 'only_offline', 'only_online']:
	#for option in ['only_online']:
		filename = 'data/result_real5_' + option + str(repeat_times) + '.json'
		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], linestyle_vec[idx], color_vec[idx], xlimit=3500)
		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative reward', weight='bold')
	plt.ylim(0, 120)
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/real5_yahoo_linUCB'+str(repeat_times)+'.eps', dpi=500)
	plt.show()