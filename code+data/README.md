# Prerequisite 
one needs to install python (>=3.7), R, and the packages used in the code (e.g. pandas, numpy, matplotlib)

since we modifies the source code of GRF (https://github.com/grf-labs/grf), one needs to recompile the grf library by 
``` $ cd grf/r-package```

``` $ Rscript build_package.R```


# How to run the code
```$ bash run.sh ```

# The example in Section 1: Introduction
the code is in example.py

# Notes
1. call_graph.png illustrates the Call Graph of an experiment
2. all the experiments are in the folder "/experiments"
3. note that the experiment indices in the files may be different from the indices in the paper. For example, in the paper the "propensity score matching + UCB" is A_1 (with index 1), but we run the experiment of this algorithm via "expriments/exp3_PS_matching_UCB.py" (with index 3).

# Data
All the data are in the /data folder
The Yahoo data has the suffix "_yahoo"
