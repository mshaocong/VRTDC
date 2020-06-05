import numpy as np
import matplotlib.pyplot as plt
from garnet import *
from optimizer.tdvanilla import TD
from optimizer.tdc import TDC
from optimizer.vrtdc import VRTDC
from optimizer.vrtd import VRTD
import time


def easy_plot(hist, color, label, cut_off=None, percentile=95, fill=True):
    upper_loss = np.percentile(hist, percentile, axis=0)
    lower_loss = np.percentile(hist, 100 - percentile, axis=0)
    avg_loss = np.mean(hist, axis=0)
    x = np.arange(avg_loss.shape[0])

    if cut_off is None:
        plt.plot(avg_loss, c=color, label=label)
    else:
        plt.plot(list(avg_loss[:cut_off]), c=color, label=label)

    if fill:
        if cut_off is None:
            plt.fill_between(x[:cut_off], lower_loss[:cut_off], upper_loss[:cut_off], color=color, alpha=0.3)
        else:
            plt.fill_between(x[:cut_off], lower_loss[:cut_off], upper_loss[:cut_off], color=color, alpha=0.3)


def mspbe(env, theta):
    v_theta = np.matmul(env.features, theta)  # (S,1)
    left = np.transpose(v_theta - env.bellman_operator(v_theta))  # (1,S)
    right = v_theta - env.bellman_operator(v_theta)  # (S,1)
    stationary = compute_stationary_dist(env.trans_kernel)
    return np.matmul(np.matmul(left, np.diag(stationary)), right)[0, 0]


def compute_stationary_dist(trans_kernel):
    # Compute the stationary distribution
    evals, evecs = np.linalg.eig(trans_kernel.T)
    evec1 = evecs[:, np.isclose(evals, 1)]
    evec1 = evec1[:, 0]
    stationary = evec1 / evec1.sum()
    stationary = stationary.real
    return stationary


def get_grad_info(phi_current_state, reward, phi_next_state, gamma=0.95):
    A_x = np.matmul(phi_current_state, np.transpose(gamma * phi_next_state - phi_current_state))  # Compute A_x
    b_x = reward * phi_current_state  # Compute b_x
    B_x = -gamma * np.matmul(phi_next_state, np.transpose(phi_current_state))  # Compute B_x
    C_x = - np.matmul(phi_current_state, np.transpose(phi_current_state))  # Compute C_x
    return A_x, b_x, B_x, C_x


def get_random_policy(num_state, num_action):
    policy = np.random.uniform(size=(num_state, num_action))
    return policy / policy.sum(axis=1)[:, np.newaxis]


def get_uniform_policy(num_state, num_action):
    policy = np.ones((num_state, num_action))
    return policy / policy.sum(axis=1)[:, np.newaxis]


def get_random_state_action_trans_kernel(num_state, num_action):
    trans_kernel = np.random.uniform(size=(num_state, num_action, num_state))
    return trans_kernel / trans_kernel.sum(axis=2)[:, :, np.newaxis]


def get_features(num_state, num_features):
    features = np.random.uniform(size=(num_state, num_features))
    return features / np.sqrt(np.sum(features ** 2, axis=1))[:, np.newaxis]


def evaluate_AbC(garnet_env, gamma=0.95, target_policy=None):
    stationary = compute_stationary_dist(garnet_env.trans_kernel)
    A = np.zeros((garnet_env.num_features, garnet_env.num_features))
    C = np.zeros((garnet_env.num_features, garnet_env.num_features))
    b = np.zeros((garnet_env.num_features, 1))
    if target_policy is None:
        for s in garnet_env.state_space:
            As = np.zeros((garnet_env.num_features, garnet_env.num_features))
            Cs = np.zeros((garnet_env.num_features, garnet_env.num_features))
            bs = np.zeros((garnet_env.num_features, 1))
            for s_pine in garnet_env.state_space:
                trans_prob = garnet_env.trans_kernel[s, s_pine]
                phi_current_state = garnet_env.phi_table(s)
                phi_next_state = garnet_env.phi_table(s_pine)
                As += trans_prob * np.matmul(phi_current_state,
                                             np.transpose(gamma * phi_next_state - phi_current_state))
                Cs += trans_prob * np.matmul(phi_current_state, np.transpose(phi_current_state))
                bs += trans_prob * garnet_env.reward[s_pine] * phi_current_state
            A += stationary[s] * As
            b += stationary[s] * bs
            C += stationary[s] * Cs
        return A, b, C
    else:
        behavior_policy = garnet_env.behavior_policy
        for s in garnet_env.state_space:
            for a in garnet_env.action_space:
                Asa = np.zeros((garnet_env.num_features, garnet_env.num_features))
                Csa = np.zeros((garnet_env.num_features, garnet_env.num_features))
                bsa = np.zeros((garnet_env.num_features, 1))
                rho_sa = target_policy[s, a] / behavior_policy[s, a]
                for s_pine in garnet_env.state_space:
                    trans_prob = garnet_env.state_action_trans_kernel[s, a, s_pine]
                    phi_current_state = garnet_env.phi_table(s)
                    phi_next_state = garnet_env.phi_table(s_pine)
                    Asa += rho_sa * trans_prob * np.matmul(phi_current_state,
                                                           np.transpose(gamma * phi_next_state - phi_current_state))
                    Csa += trans_prob * np.matmul(phi_current_state, np.transpose(phi_current_state))
                    bsa += rho_sa * trans_prob * garnet_env.reward[s_pine] * phi_current_state
                A += stationary[s] * Asa
                b += stationary[s] * bsa
                C += stationary[s] * Csa
        return A, b, C
