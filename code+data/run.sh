# run the experiments, and plot the figures
# we comment the experiments in our supplementary materials, you can uncomment them to run

###########################################################
#################### on synthetic data ####################
# # algorithm A_{UCB+EM}
# python experiments/exp1_exact_UCB.py 
# python plot/plot_exp1_exact_UCB.py

# # algorithm A_{UCB+IPSW}
python experiments/exp2_weighting_UCB.py
python plot/plot_exp2_IPSW_UCB.py

# algorithm A_{UCB+PSM}
python experiments/exp3_PS_matching_UCB.py 
python plot/plot_exp3_PSmatch_UCB.py

# # algorithm A_{LinUCB+LR} (with linear function)
# python3 experiments/exp4_linUCB.py
# python3 plot/plot_exp4_linUCB.py

# algorithm A_{LinUCB+LR} (with non-linear function, in comparison form)
python experiments/exp6_compare_linUCB.py
#python plot/plot_exp6_compare_linUCB.py

# algorithm A_{Fst+MoF}
Rscript experiments/exp0_online_forest.R
python plot/plot_exp0_online_forest.py
###########################################################


###########################################################
#################### on real Yahoo data ###################
# It may take a while when we set the repeat_time to 5. In fact, to plot our figure, we should set repeat_time=50
python experiments/real5_disjointLinUCB.py --repeat_times=5
python plot/plot_real5_yahoo_linUCB.py
###########################################################