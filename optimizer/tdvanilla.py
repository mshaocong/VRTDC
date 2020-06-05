import optimizer.tdbase
import numpy as np


class TD(optimizer.tdbase.TD_Base):
    def __init__(self, env, alpha=0.01, target_policy=None, gamma=0.95):
        super(TD, self).__init__(env, alpha, target_policy, gamma)

    def update(self, current_state, reward, next_state, action=None):
        A_x, b_x, B_x, C_x = self._extract_grad_info(current_state, reward, next_state, action)
        alpha = self.alpha
        theta = np.copy(self.theta)
        self.theta = theta + alpha * (np.matmul(A_x, theta) + b_x)
        return self.theta
