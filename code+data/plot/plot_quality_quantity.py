# show how the regret of three kinds of algorithms varies, under different "bias" and "number of logged data"

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import pylab

matplotlib.rcParams['hatch.linewidth'] = 2

font = {'size': 26, 'weight': 'bold'}
matplotlib.rc('font', **font)

data_dict = {
	"100,0.0": [1.4925, 2.5, 13.855],
	"20,0.0": [10.2675, 25.0, 16.15],
	"50,0.0": [6.4925, 12.5, 15.1325],
	"100,0.3": [10.895, 32.5, 15.1775],
	"20,0.3": [14.5425, 66.25, 16.3825],
	"50,0.3": [12.145, 38.75, 15.22],
	"100,0.6": [15.32, 183.75, 14.3925],
	"20,0.6": [14.5875, 155, 15.2375],
	"50,0.6": [14.7225, 185.0, 17.14],
	"100,0.9": [15.285, 242.5, 14.93],
	"20,0.9": [17, 220, 13.585],
	"50,0.9": [16.4675, 235.0, 14.625]
}

def plot_legend():
	# plot the legend only
	ax = plot_bias(0.0, savefig=False)

	figLegend = pylab.figure(figsize=(8.5, 0.5))
	pylab.figlegend(*ax.get_legend_handles_labels(), loc='upper left',\
	frameon=False, fontsize=18, ncol=3, mode=None)

	#figLegend.text(0.84, 0.22, 'influence\n(PSRL)', fontsize=18)

	output_filename = 'images/plot_legend_quality_quantity.eps'
	figLegend.savefig(output_filename, dpi=200)
	figLegend.show()

def plot_bias(bias, savefig=True):
	# plot the figure for a certain bias
	ind = np.arange(3)
	height = 1/4

	if bias == 0.0:
		fig = plt.figure(figsize=(5.3 ,4.8))
		ax = fig.add_axes([0.235, 0.18, 0.75, 0.78])
	else:
		fig = plt.figure(figsize=(4.5 ,4.8))
		ax = fig.add_axes([0.03, 0.18, 0.93, 0.78])


	option_name = ['offline_online', 'only_offline', 'only_online']
	hatch_vec = ['x', '\\', '/']
	linestyle_vec = ['-', ':', '--']
	color_vec = ['green', 'blue', 'red']

	for option_num in [1, 2, 0]: # 0 is offline_online, 1 is only_offline, 2 is only_online
		regret_vec = []
		for sample_size in [20, 50, 100]:
			key = str(sample_size) + ',' + str(bias)
			if option_num == 2: # online
				regret_vec.append(15.1325)
			else:
				regret_vec.append(data_dict[key][option_num])

		print (ind+option_num*height)

		rects = ax.barh(ind+option_num*height, regret_vec, height, label=option_name[option_num], linewidth=3,\
			fill=False, ls=linestyle_vec[option_num], hatch=hatch_vec[option_num], \
			edgecolor=color_vec[option_num])

	ax.set_yticks(ind + 1*height)
	ax.set_yticklabels([20, 50, 100], fontsize=24)
	

	if bias == 0.0:
		plt.ylabel('num. of log samples', weight='bold')
		plt.xlabel('total regret(bias='+str(0)+')', weight='bold')
	else:
		plt.xlabel('total regret(bias='+str(bias)+')', weight='bold')
		plt.yticks([])

	if savefig == True:
		plt.savefig('images/plot_quality_quantity_bias'+str(bias)+'.eps', dpi=500)

		#plt.legend()

		plt.show()

	return ax

if __name__ == '__main__':
	plot_bias(0.0)
	plot_bias(0.3)
	plot_bias(0.6)
	plot_bias(0.9)
	plot_legend()



