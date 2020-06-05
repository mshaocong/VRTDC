import optimizer.tdbase
import numpy as np


class TDC(optimizer.tdbase.TD_Base):
    def __init__(self, env, alpha=0.01, beta=0.006, target_policy=None, gamma=0.95):
        super(TDC, self).__init__(env, alpha, target_policy, gamma)
        self.beta = beta
        self.w = np.zeros((self.num_features, 1))

    def set_w(self, w):
        self.w = w

    def update(self, current_state, reward, next_state, action=None):
        A_x, b_x, B_x, C_x = self._extract_grad_info(current_state, reward, next_state, action)
        alpha = self.alpha
        beta = self.beta
        theta = np.copy(self.theta)
        w = np.copy(self.w)
        self.theta = theta + alpha * (np.matmul(A_x, theta) + b_x + np.matmul(B_x, w))
        self.w = w + beta * (np.matmul(A_x, theta) + b_x + np.matmul(C_x, w))
        return self.theta, self.w
