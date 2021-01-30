import sys
sys.path.append('.')
import matplotlib.pyplot as plt
import numpy as np

import plot_utils

marker_vec = ['x', 'o', '^', 'D']
color_vec = ['green', 'red']
label_vec = ['ours', 'batch', 'ours(time)', 'batch(time)']
line_vec = [(2, 0.01), (1, 1)]

def plot_regret():
	fig = plt.figure()
	ax = fig.add_axes([0.17, 0.16, 0.8, 0.8])

	option = 'offline_online'
	N_offline = 100

	idx = 0
	for batch_mode in [False, True]:
		if batch_mode:
			filename = 'data/result_exp13_' + option + '.json'
		else:
			bias = [0.0, 0]
			filename = 'data/result_exp2_' + option +'_bias'+str(bias) + '.json'

		plot_utils.plot_cumulative_regret(ax, filename, label_vec[idx], (2, 0.01), color=color_vec[idx], marker=marker_vec[idx])

		idx += 1

	plt.xlabel('time $t$', weight='bold')
	plt.ylabel('cumulative regret', weight='bold')

	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp13_batch_mode.eps', dpi=500)
	plt.show()

def plot_time():
	fig = plt.figure()
	ax = fig.add_axes([0.21, 0.16, 0.75, 0.8])

	T_vec = range(100, 1001, 100)

	time_mat = [
		[16.62, 19.28, 24.81, 31.14, 35.84, 41.35, 44.47, 51.60, 57.21, 60.75],
		[15.96, 18.49, 21.36, 26.10, 30.30, 33.55, 37.60, 39.36, 44.68, 48.79]
	]

	for idx in range(2):
		plt.plot(T_vec, np.asarray(time_mat[idx])/500, marker=marker_vec[idx], \
			color=color_vec[idx], label=label_vec[idx], linewidth=3)

	plt.xlabel('num. of online rounds T', weight='bold')
	plt.ylabel('running time (s)', weight='bold')
	plt.legend(loc='upper left', fontsize=22, frameon=False)
	plt.savefig('images/exp13_batch_mode_running_time.eps', dpi=500)
	plt.show()

def plot_time_and_regret():
	fig = plt.figure()

	ax1 = fig.add_axes([0.12, 0.16, 0.67, 0.8])
	ax1.set_xlabel('time $t$', weight='bold')
	ax1.set_ylabel('cumulative regret', weight='bold', color='blue')
	ax1.tick_params(axis='y', labelcolor='blue')

	option = 'offline_online'
	idx = 0
	for batch_mode in [False, True]:
		if batch_mode:
			filename = 'data/result_exp13_' + option + '.json'
		else:
			filename = 'data/result_exp2_' + option + '.json'

		plot_utils.plot_cumulative_regret(ax1, filename, label_vec[idx], \
			line_vec[idx], color='blue', marker=marker_vec[idx], marker_every=100)

		idx += 1

	plt.ylim(0, 9.5)
	l = plt.legend(loc='upper left', fontsize=22, frameon=False)
	for text in l.get_texts():
		text.set_color("blue")

	ax2 = ax1.twinx()
	ax2.set_ylabel('running time (s)', weight='bold', color='red')
	ax2.tick_params(axis='y', labelcolor='red')
	T_vec = range(100, 1001, 100)
	time_mat = [
		[16.62, 19.28, 24.81, 31.14, 35.84, 41.35, 44.47, 51.60, 57.21, 60.75],
		[15.96, 18.49, 21.36, 26.10, 30.30, 33.55, 37.60, 39.36, 44.68, 48.79]
	]
	for idx in range(2):
		plt.plot(T_vec, np.asarray(time_mat[idx])/500, marker=marker_vec[idx+2], \
			color='red', label=label_vec[idx+2], linewidth=3, dashes=line_vec[idx])
	plt.ylim(0, 0.15)
	l = plt.legend(loc='lower right', fontsize=22, frameon=False)
	for text in l.get_texts():
		text.set_color("red")
	#fig.tight_layout()
	plt.savefig('images/exp13_batch_mode_regret_time.eps', dpi=500)
	plt.show()


if __name__ == '__main__':
	#plot_regret()
	#plot_time()
	plot_time_and_regret()

