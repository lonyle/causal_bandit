import sys
sys.path.append('.')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_utils
import plot_exp6_compare_linUCB

linestyle_vec = [(2, 0.01), (2,4), (4,2)]
color_vec = ['green', 'black', 'red']

def plot(ax):
	idx = 0
	line_vec = []
	label_vec = ['offline+online', 'only_offline', 'only_online']
	for option in ['offline_online', 'only_offline', 'only_online']:
		filename = 'data/result_online_forest_' + option + '.json'
		line = plot_utils.plot_cumulative_regret(ax, filename, None, linestyle_vec[idx], color_vec[idx])
		idx += 1
		line_vec.append(line)
	return line_vec, label_vec

if __name__ == '__main__':
	fig = plt.figure()
	ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])

	line_vec, label_vec = plot(ax)
	plot_exp6_compare_linUCB.plot(ax, marker='o')

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')
	#plt.ylim(0, 17)
	first_legend = plt.legend(line_vec, label_vec, loc='upper left', fontsize=22, frameon=False)

	# another legend
	plt.gca().add_artist(first_legend)
	line1 = Line2D([0], [0], ls='--', dashes=(4,2), color='red', linewidth=3, marker='o', markersize=6, fillstyle='none', markeredgewidth=2)
	line2 = Line2D([0], [0], ls='--', dashes=(4,2), color='red', linewidth=3)
	lines = [line1, line2]
	labels = ['LinUCB', 'forest']
	plt.legend(lines, labels, loc='center right', bbox_to_anchor=(1.04,0.42), fontsize=22, frameon=False)

	t1 = ax.text(135, 22, "$\\rightarrow$", rotation=135, size=30)
	t2 = ax.text(135, 11, "$\\rightarrow$", rotation=-135, size=30)

	plt.savefig('images/exp0_online_forest_epsilon.eps', dpi=500)
	plt.savefig('images/exp0_online_forest_epsilon.png', dpi=500)
	plt.show()
