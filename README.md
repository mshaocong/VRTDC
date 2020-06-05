# Variance-Reduced Off-Policy TDC Learning: Non-Asymptotic Convergence Analysis

In the folder `optimizer`, we implemented the policy evaluation algorithms used in this paper (including TD, TDC, VRTD, and VRTDC). In `garnet.py`, we implemented the Garnet problem enviroment. 

## Compare the convergence curve among four algorithms

The comparision among four algorithms is implemented in `main.py` for the Garnet problem and `frozen_lake.py` for the Frozen Lake enviroment. Run

* `python main.py`, and

* `python frozen_lake.py`

to repeat the experiments.

## Compare the asymptotic error among VRTD and VRTDC

The comparision between the asymptotic errors of VRTD and VRTDC is implemented in `test.py` for the Garnet problem and `frozen_lake_test.py` for the Frozen Lake enviroment. Run

* `python test.py`, and 

* `python frozen_lake_test.py`

to repeat the experiments. Or `python testmul.py` to use multiple cores of CPU for different trajectories.
