import optimizer.tdbase
import numpy as np
import copy


class VRTD(optimizer.tdbase.TD_Base):
    def __init__(self, env, batch_size=1000, alpha=0.01, target_policy=None, gamma=0.95):
        super(VRTD, self).__init__(env, alpha, target_policy, gamma)
        self.batch_size = batch_size
        self.w = np.zeros((self.num_features, 1))

        self._grad_cache = {"A": np.zeros((self.env.num_features, self.env.num_features)),
                            "b": np.zeros((self.env.num_features, 1)),
                            "B": np.zeros((self.env.num_features, self.env.num_features)),
                            "C": np.zeros((self.env.num_features, self.env.num_features))}
        self.batch_grad_info = None

        self._pars_cache = {"theta": np.copy(self.theta), "w": np.copy(self.w)}
        self.theta_tilde = np.copy(self.theta)
        self.w_tilde = np.copy(self.w)

        self._trajectory_cache = []
        self._trajectory = None

    def set_w(self, w):
        self.w = w

    def _get_batch_grad_info(self):
        A = self.batch_grad_info["A"]
        b = self.batch_grad_info["b"]
        B = self.batch_grad_info["B"]
        C = self.batch_grad_info["C"]
        return A, b, B, C

    def update(self, current_state, reward, next_state, action=None):
        A_x, b_x, B_x, C_x = self._extract_grad_info(current_state, reward, next_state, action)

        if len(self._trajectory_cache) < self.batch_size:
            # 开始预存这个trajectory里的full gradient info
            self._trajectory_cache.append((current_state, reward, next_state, action))
            self._grad_cache["A"] = self._grad_cache["A"] + A_x / self.batch_size
            self._grad_cache["b"] = self._grad_cache["b"] + b_x / self.batch_size
            self._grad_cache["B"] = self._grad_cache["B"] + B_x / self.batch_size
            self._grad_cache["C"] = self._grad_cache["C"] + C_x / self.batch_size
        else:
            # 记录了刚好M个样本的时候. 进入下一个batch
            # cache全部重置, theta-tilde更新成前一个batch的均值, theta重设为theta-tilde
            self._trajectory = copy.deepcopy(self._trajectory_cache)
            self._trajectory_cache = [(current_state, reward, next_state, action)]

            self.theta_tilde = np.copy(self._pars_cache["theta"])
            self.w_tilde = np.copy(self._pars_cache["w"])
            self.theta = np.copy(self.theta_tilde)
            self.w = np.copy(self.w_tilde)

            self.batch_grad_info = copy.deepcopy(self._grad_cache)
            self._grad_cache = {"A": A_x / self.batch_size,
                                "b": b_x / self.batch_size,
                                "B": B_x / self.batch_size,
                                "C": C_x / self.batch_size}
            # 这段代码区分每次theta-tilde是取均值还是取上个batch最后一次theta
            self._pars_cache = {"theta": np.copy(self.theta), "w": np.copy(self.w)}

        if self._trajectory is None:
            # 前M个样本, 不更新参数
            self._pars_cache["theta"] = np.copy(self.theta)
            self._pars_cache["w"] = np.copy(self.w)
            return self.theta, self.w
        else:
            s, r, s_pine, a = self._trajectory[np.random.choice(self.batch_size)]
            A_x, b_x, B_x, C_x = self._extract_grad_info(s, r, s_pine, a)  # Compute the current gradient info

            alpha = self.alpha
            theta = np.copy(self.theta)

            A, b, B, C = self._get_batch_grad_info()  # Averaged gradient over this batch
            grad_1_theta = np.matmul(A_x, theta) + b_x
            grad_2_theta = np.matmul(A_x, self.theta_tilde) + b_x
            grad_3_theta = np.matmul(A, self.theta_tilde) + b

            self.theta = theta + alpha * (grad_1_theta - grad_2_theta + grad_3_theta)

            # 这段代码区分每次theta-tilde是取均值还是取上个batch最后一次theta
            # self._pars_cache["theta"] = self._pars_cache["theta"] + self.theta / self.batch_size
            # self._pars_cache["w"] = self._pars_cache["w"] + self.w / self.batch_size
            self._pars_cache["theta"] = np.copy(self.theta)
            self._pars_cache["w"] = np.copy(self.w)

        return self.theta, self.w
"""    
class VRTD(optimizer.tdbase.TD_Base):
    def __init__(self, env, batch_size=1000, alpha=0.01, target_policy=None, gamma=0.95):
        super(VRTD, self).__init__(env, alpha, target_policy, gamma)
        self.batch_size = batch_size

        self._grad_cache = {"A": np.zeros((self.env.num_features, self.env.num_features)),
                            "b": np.zeros((self.env.num_features, 1)),
                            "B": np.zeros((self.env.num_features, self.env.num_features)),
                            "C": np.zeros((self.env.num_features, self.env.num_features))}
        self.batch_grad_info = None

        self._pars_cache = {"theta": np.zeros((self.num_features, 1))}
        self.theta_tilde = np.copy(self.theta)

        self._trajectory_cache = []
        self._trajectory = None

    def _get_batch_grad_info(self):
        A = self.batch_grad_info["A"]
        b = self.batch_grad_info["b"]
        B = self.batch_grad_info["B"]
        C = self.batch_grad_info["C"]
        return A, b, B, C

    def update(self, current_state, reward, next_state, action=None):
        A_x, b_x, B_x, C_x = self._extract_grad_info(current_state, reward, next_state, action)

        if len(self._trajectory_cache) < self.batch_size:
            # 开始预存这个trajectory里的full gradient info
            self._trajectory_cache.append((current_state, reward, next_state, action))
            self._grad_cache["A"] = self._grad_cache["A"] + A_x / self.batch_size
            self._grad_cache["b"] = self._grad_cache["b"] + b_x / self.batch_size
            self._grad_cache["B"] = self._grad_cache["B"] + B_x / self.batch_size
            self._grad_cache["C"] = self._grad_cache["C"] + C_x / self.batch_size
        else:
            # 记录了刚好M个样本的时候. 进入下一个batch
            # cache全部重置, theta-tilde更新成前一个batch的均值, theta重设为theta-tilde
            self._trajectory = copy.deepcopy(self._trajectory_cache)
            self._trajectory_cache = [(current_state, reward, next_state, action)]

            self.theta_tilde = np.copy(self._pars_cache["theta"])
            self.theta = np.copy(self.theta_tilde)

            self.batch_grad_info = copy.deepcopy(self._grad_cache)
            self._grad_cache = {"A": A_x / self.batch_size,
                                "b": b_x / self.batch_size,
                                "B": B_x / self.batch_size,
                                "C": C_x / self.batch_size}
            self._pars_cache = {"theta": np.zeros((self.num_features, 1))}

        if self._trajectory is None:
            # 前M个样本, 不更新参数
            return self.theta
        else:
            s, r, s_pine, a = self._trajectory[np.random.choice(self.batch_size)]
            A_x, b_x, B_x, C_x = self._extract_grad_info(s, r, s_pine, a)  # Compute the current gradient info

            alpha = self.alpha
            theta = np.copy(self.theta)

            A, b, B, C = self._get_batch_grad_info()  # Averaged gradient over this batch
            grad_1 = np.matmul(A_x, theta) + b_x
            grad_2 = np.matmul(A_x, self.theta_tilde) + b_x
            grad_3 = np.matmul(A, self.theta_tilde) + b

            self.theta = theta + alpha * (grad_1 - grad_2 + grad_3)

            self._pars_cache["theta"] = self._pars_cache["theta"] + self.theta / self.batch_size
        return self.theta
"""