import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

linestyle_vec = [(2, 0.01), (2,4), (4,2)]
color_vec = ['green', 'blue', 'red']

def plot(ax):
	idx = 0
	line_vec = []
	label_vec = []
	for option in ['offline_online', 'only_offline', 'only_online']:
		filename = 'data/result_real0_yahoo3500_' + option + '.json'
		line = plot_utils.plot_cumulative_regret(ax, filename, None, linestyle_vec[idx], color_vec[idx])
		idx += 1
		line_vec.append(line)
		label_vec.append(option)
	return line_vec, label_vec

if __name__ == '__main__':
	fig = plt.figure()
	ax = fig.add_axes([0.19, 0.16, 0.79, 0.8])

	line_vec, label_vec = plot(ax)

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative reward', weight='bold')

	first_legend = plt.legend(line_vec, label_vec, loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/real0_yahoo3500'+'.eps', dpi=500)

	plt.show()