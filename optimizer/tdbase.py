import numpy as np


class TD_Base:
    def __init__(self, env, alpha=0.01, target_policy=None, gamma=0.95):
        self.features = env.features
        self.num_features = env.num_features
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        if target_policy is None:
            self.target_policy = env.behavior_policy
        else:
            self.target_policy = target_policy

        self.theta = np.zeros((self.num_features, 1))

    def set_theta(self, theta):
        self.theta = theta

    def _phi_table(self, state):
        return self.features[state, :].reshape((self.num_features, 1))

    def _extract_grad_info(self, current_state, reward, next_state, action=None):
        phi_current_state = self._phi_table(current_state)
        phi_next_state = self._phi_table(next_state)
        gamma = self.gamma
        if action is None:
            A_x = np.matmul(phi_current_state, np.transpose(gamma * phi_next_state - phi_current_state))  # Compute A_x
            b_x = reward * phi_current_state  # Compute b_x
            B_x = -gamma * np.matmul(phi_next_state, np.transpose(phi_current_state))  # Compute B_x
            C_x = - np.matmul(phi_current_state, np.transpose(phi_current_state))  # Compute C_x
            return A_x, b_x, B_x, C_x
        else:
            rho_sa = self.target_policy[current_state, action]/self.env.behavior_policy[current_state, action]
            A_x = rho_sa * np.matmul(phi_current_state, np.transpose(gamma * phi_next_state - phi_current_state))  # Compute A_x
            b_x = rho_sa * reward * phi_current_state  # Compute b_x
            B_x = -gamma * rho_sa * np.matmul(phi_next_state, np.transpose(phi_current_state))  # Compute B_x
            C_x = - np.matmul(phi_current_state, np.transpose(phi_current_state))  # Compute C_x
            return A_x, b_x, B_x, C_x

    def update(self, current_state, reward, next_state, action=None):
        raise NotImplementedError
