import gym
import matplotlib.pyplot as plt
from garnet import *
from utils import *
from optimizer.tdvanilla import TD
from optimizer.tdc import TDC
from optimizer.vrtdc import VRTDC
from optimizer.vrtd import VRTD
import time


def easy_simulation(env, alpha, beta, batch_size, trajectory_length=50000, num_simulation=100, gamma=0.95,
                    target=None):
    ini_start = time.time()

    print("Initialization...")
    A, b, C = evaluate_AbC(env, gamma=gamma, target_policy=target)
    theta_ast = -np.matmul(np.linalg.inv(A), b)
    # R = np.sum(theta_ast**2)*2

    # ini_theta = np.zeros_like(theta_ast)
    ini_theta = theta_ast + 0.6 * np.random.normal(scale=1.0, size=theta_ast.shape)
    # ini_theta = theta_ast + np.random.normal(scale=1.0, size=theta_ast.shape)
    # ini_w = np.random.normal(size=theta_ast.shape)
    print("Optimal Theta:", theta_ast)
    print("Initial Theta:", ini_theta)
    print("Initialization Completed. Time Spent:", time.time() - ini_start)

    all_td_hist = []
    all_tdc_hist = []
    all_vrtdc_hist = []
    all_vrtd_hist = []
    for _ in range(num_simulation):
        env.reset()
        current_state = env.current_state

        td = TD(env, alpha=alpha, target_policy=target, gamma=gamma)
        td.set_theta(ini_theta)

        tdc = TDC(env, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        tdc.set_theta(ini_theta)

        vrtdc = VRTDC(env, batch_size=batch_size, alpha=alpha, beta=beta, target_policy=target, gamma=gamma)
        vrtdc.set_theta(ini_theta)

        vrtd = VRTD(env, batch_size=batch_size, alpha=alpha, target_policy=target, gamma=gamma)
        vrtd.set_theta(ini_theta)

        print("Start Training. Simulation:", _ + 1)
        train_start = time.time()

        td_hist = [np.sum((td.theta - theta_ast) ** 2)]
        tdc_hist = [np.sum((td.theta - theta_ast) ** 2)]
        vrtdc_hist = [np.sum((td.theta - theta_ast) ** 2)]
        vrtd_hist = [np.sum((td.theta - theta_ast) ** 2)]
        count = 1
        for i in range(trajectory_length):
            # if i % 100 == 0:
            #    env.reset()
            next_state, reward, action = env.step()
            # action = None

            if count <= batch_size:
                # 第1到M次更新的时候, 此时只计算一次梯度
                vrtdc_hist.append(np.sum((vrtdc.theta - theta_ast) ** 2))
            else:
                if count % batch_size == 0:
                    # 第mM次更新梯度的时候(m>1), 此时计算一个batch_size的梯度
                    vrtdc_hist = vrtdc_hist + [np.sum((vrtdc.theta - theta_ast) ** 2) for _ in range(batch_size)]
                elif count % batch_size != 0:
                    # 第mM+1到mM+M-1, 此时每次迭代 计算两次梯度
                    vrtdc_hist.append(np.sum((vrtdc.theta - theta_ast) ** 2))
                    vrtdc_hist.append(np.sum((vrtdc.theta - theta_ast) ** 2))
            vrtdc.update(current_state, reward, next_state, action)
            # if np.sum(theta**2) > R:
            #    vrtdc.theta = vrtdc.theta / np.sqrt(np.sum(theta**2))*R

            if count <= batch_size:
                # 第1到M次更新的时候, 此时只计算一次梯度
                vrtd_hist.append(np.sum((vrtd.theta - theta_ast) ** 2))
            else:
                if count % batch_size == 0:
                    # 第mM次更新梯度的时候(m>1), 此时计算一个batch_size的梯度
                    vrtd_hist = vrtd_hist + [np.sum((vrtd.theta - theta_ast) ** 2) for _ in range(batch_size)]
                elif count % batch_size != 0:
                    # 第mM+1到mM+M-1, 此时每次迭代 计算两次梯度
                    vrtd_hist.append(np.sum((vrtd.theta - theta_ast) ** 2))
                    vrtd_hist.append(np.sum((vrtd.theta - theta_ast) ** 2))
            vrtd.update(current_state, reward, next_state, action)

            tdc.update(current_state, reward, next_state, action)
            tdc_hist.append(np.sum((tdc.theta - theta_ast) ** 2))

            td.update(current_state, reward, next_state, action)
            td_hist.append(np.sum((td.theta - theta_ast) ** 2))

            current_state = np.copy(next_state)
            if (i + 1) % 10000 == 0:
                print("Current iteration:", i + 1, ". Time Spent:", time.time() - train_start)
                train_start = time.time()
            count += 1
        all_td_hist.append(td_hist)
        all_tdc_hist.append(tdc_hist)
        all_vrtdc_hist.append(vrtdc_hist)
        all_vrtd_hist.append(vrtd_hist)
    return all_td_hist, all_tdc_hist, all_vrtdc_hist, all_vrtd_hist


np.random.seed(91)

# Compare TD, TDC, and VRTDC
num_states = 500
num_actions = 20
branching_factor = 50
num_features = 15

print("Set Up the Simulation Environment...")
env = Garnet(num_states, num_actions, branching_factor, num_features)
print("Done.")

gamma = 0.95
max_num_iteration = 50000
batch_size = 3000
alpha = 0.1
beta = 0.02
target = get_random_policy(num_states, num_actions)
num_simulation = 100


hist_td, hist_tdc, hist_vrtdc, hist_vrtd = easy_simulation(env, alpha, beta, batch_size,
                                                           trajectory_length=max_num_iteration,
                                                           num_simulation=num_simulation, gamma=gamma, target=target)


"""import pickle
with open('hist.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([hist_td, hist_tdc, hist_vrtdc, hist_vrtd], f)
"""
plt.figure()
easy_plot(hist_tdc, "orange", "TDC")
easy_plot(hist_td, "g", "TD")
easy_plot(hist_vrtd, "b", "VRTD: M=3000", cut_off=len(hist_td[0]))
easy_plot(hist_vrtdc, "r", "VRTDC: M=3000", cut_off=len(hist_td[0]))

plt.legend(loc=1)
plt.ylabel(r"$||\theta - \theta^\ast ||^2$")
plt.xlabel("# of gradient computations")
# plt.savefig('fig1.png', dpi=300)
plt.show()
