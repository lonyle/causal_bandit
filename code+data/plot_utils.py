import matplotlib.pyplot as plt
import numpy as np
import json

import matplotlib
font = {'size': 24, 'weight':'bold'}
matplotlib.rc('font', **font)

def plot_cumulative_regret(ax, filename, label, linestyle, color='black', \
		marker=None, xlimit=None, confidence_interval=False, max_reward=None, marker_every=10):
	count = 0
	for line in open(filename):
		cumulative_regret_vec = json.loads(line)#[:xlimit]
		if xlimit != None:
			if len(cumulative_regret_vec) >= xlimit:
				cumulative_regret_vec = cumulative_regret_vec[:xlimit]
			else:
				cumulative_regret_vec = np.arange(1, xlimit+1, 1) * (cumulative_regret_vec[-1]/len(cumulative_regret_vec)) # fill the empty
				#cumulative_regret_vec = np.arange(1, xlimit+1, 1) * cumulative_regret_vec[0] # fill the empty

		if max_reward != None:
			cumulative_regret_vec = np.arange(1, len(cumulative_regret_vec)+1, 1) * max_reward - cumulative_regret_vec
		#print ("length:", len(cumulative_regret_vec))
		if count == 0:
			sum_cumulative_regret_vec = np.asarray(cumulative_regret_vec)
		else:
			sum_cumulative_regret_vec += np.asarray(cumulative_regret_vec)
		count += 1

	average_cumulative_regret_vec = sum_cumulative_regret_vec / count

	print (average_cumulative_regret_vec[-1])


	if confidence_interval == True:
		cumulative_regret_mat = []
		for line in open(filename):
			cumulative_regret_vec = json.loads(line)
			cumulative_regret_mat.append(cumulative_regret_vec)
		cumulative_regret_mat = np.asarray(cumulative_regret_mat)
		p05 = np.percentile(cumulative_regret_mat, 20, axis=0)
		p95 = np.percentile(cumulative_regret_mat, 80, axis=0)
		ax.fill_between(np.arange(len(average_cumulative_regret_vec)), p05, p95,\
			color=color, alpha=0.3, linewidth=0)

	if color == 'black':
		color = 'dimgray'
	plot_line, = ax.plot(np.arange(len(average_cumulative_regret_vec)), average_cumulative_regret_vec, '--', color=color, label=label,\
		dashes=linestyle, linewidth=3, marker=marker, markevery=marker_every, markersize=6, fillstyle='none', markeredgewidth=2)
	#ax.plot(np.arange(len(average_cumulative_regret_vec)), average_cumulative_regret_vec, label=label)
	return plot_line
