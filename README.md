# Variance-Reduced Off-Policy TDC Learning: Non-Asymptotic Convergence Analysis

TDC algorithm is a classical policy evaluation algorithm which can guarantee convergence in the off-policy setting. This repo provides the experiments codes for the newly proposed varaince-reduced TDC algorithm in the paper, *Variance-Reduced Off-Policy TDC Learning: Non-Asymptotic Convergence Analysis*. The algorithm is presented as follow,
![VRTDC Algorithm](/figs/alg.png)

In the folder `optimizer`, we implemented the policy evaluation algorithms used in this paper (including TD, TDC, VRTD, and VRTDC). In `garnet.py`, we implemented the Garnet problem enviroment, which is similar to the environment provided in OpenAI but supports more custumized setting and includes more environment information including the transition kernel and stationary distribution under a specified behavior policy. 

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

## Experiments Results
Our experiment results show that VRTDC can outperform TD, TDC, and variance-reduced TD algorithm in the Garnet problem. The left figure shows VRTDC requires less psudo-gradient computations for achieving the same precision. And the right figure shows VRTDC always has smaller asymptotic error compared to VRTD.
![Garnet Problem](/figs/fig1.png)
For the Frozen Lake environment, VRTDC can also outperform TD, TDC, and variance-reduced TD algorithm. The left figure shows VRTDC requires less psudo-gradient computations for achieving the same precision. And the right figure shows VRTDC always has smaller asymptotic error compared to VRTD.
![Frozen Lake](/figs/fig2.png)
