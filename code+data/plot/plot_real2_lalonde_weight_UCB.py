import sys
sys.path.append('.')
import matplotlib.pyplot as plt

import plot_utils

if __name__ == '__main__':
	for option in ['only_offline', 'only_online', 'offline_online']:
	#for option in ['only_offline']:
		filename = 'data/result_real2_' + option + '.json'
		plot_utils.plot_cumulative_regret(filename, option)

	plt.xlabel('t')
	plt.ylabel('cummulative reward')
	plt.legend()
	plt.savefig('images/real2_lalonde_IPSW_UCB.png', dpi=500)
	plt.show()
