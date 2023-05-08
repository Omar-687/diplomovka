
'''CVXPY is a Python package for convex optimization.
It allows users to formulate and solve convex optimization
problems in a simple and intuitive way,
using a high-level modeling language. '''
import time
import numpy as np
import torch as to
import torch.nn as nn
import torchvision
from actcrit import *

# PPC is not a classification algorithm
'''
In the SAC algorithm, the 
standard deviation of the Gaussian distribution
used to sample actions is learned by the neural network.



'''
# 3*10**-4
class PPCController(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, alpha, min_timestamp, time_unit=12):
        super(PPCController, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.fc2 = nn.Linear(dim_hid, dim_hid)
        self.fc3 = nn.Linear(dim_hid, dim_out)
        self.relu = nn.ReLU()
        # self.optimizer = to.optim.Adam(self.parameters(),lr=alpha)
        self.o1 = 0.1
        self.o2 = 0.2
        self.o3 = 2
        self.log_std_min = -20
        self.log_std_max = 2
        self.alpha = alpha
        self.min_timestamp = min_timestamp
        self.time_unit = 12 / 60
        self.control_space = [i for i in range(0,151,15)]

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        log_std = to.clamp(x, min=self.log_std_min, max=self.log_std_max)
        return x, log_std


    def sample_action(self, mean, log_std):
        with to.no_grad():
            std = log_std.exp()
            normal = to.randn_like(mean)
            action = to.tanh(mean + std * normal)
            action = action.detach().cpu().numpy()
        return action
    def calc_MEF(self,N,a_js,d_js,e_js):
        mef = -float('inf')
        N = len(a_js)
        for j in range(N):
            temp = a_js[j] - (1 - e_js[j]) * d_js[j]
            if temp > mef:
                mef = temp
        return mef


    def minibatches(self, inputs):
        yield inputs
    def psi(self, t):
        return (1 - t) / 24

    # Define the original function
    # Define the original function
    def original_normalized_mse(self, sts, uts, N, L, T, xi):
        res = 0
        for k in range(L):
            for t in range(T):
                temp = 0
                for j in range(N):
                    temp += sts[k][t][j] - uts[k][t]
                temp = abs(temp) ** 2
                res += temp
        return res / (L * T * xi)

    def numpy_normalized_mse(self, sts, uts, N, L, T, xi):
        res = 0
        for k in range(L):
            for t in range(T):
                res += abs(np.sum(sts[k][t][:N] - uts[k][t]))**2
        return res / (L * T * xi)
    def original_mpe(self, sts, ej, L, T, N):
        res = 0
        for k in range(L):
            for t in range(T):
                for j in range(N):
                    res += sts[k][t][j] / (L * T * np.sum(ej))
        res = 1 - res
        return res
    # def optimized_mpe(self, sts, ej, L, T, N):
    #     res = 0
    #     for j in range(N):
    #         r = sts[:, :, j] + ej[j] * np.arange(T).reshape(1, T)
    #         r = r.flatten()
    #         res += np.sum(np.diff(np.sort(r)) > 0)
    #
    #     res /= (L * T * np.sum(ej))
    #     res = 1 - res
    #     return res

    def calculate_s_t(self,e, d, u ,x_0):
        L, T, N = e.shape

        # initialize state variables
        x_t = x_0
        s_t = np.zeros((L, T, N))

        for k in range(L):
            for t in range(T):
                for j in range(N):
                    # calculate x_t+1
                    # x_t1 = A[k] @ x_t[k][t] + B[k] @ u[k][t][j]
                    x_t1 = 0
                    # calculate d_t+1
                    d_t1 = d[k][t][j]

                    # calculate s_t
                    s_t[k][t][j] = e[k][t][j] * x_t1 + d_t1

                    # # update state variables
                    # x_t[k][t] = x_t1

        return s_t
    def greedy_approx_mef(self,pt):
        return 1
    def indicator(self,condition):
        if condition:
            return 1
        return 0
    def phi(self,pt):
        return 1
    def reward_fun(self,p_t,x_t,t0,uts,T,ajs,N,ej):
        res = 0
        res += self.greedy_approx_mef(p_t) + self.o1 * np.sum(uts[t0][:N])
        temp = 0
        for i in range(N):
            for t in range(1,T):
                temp += self.indicator((ajs[i] <= t <= ajs[i]+self.time_unit)) *(ej[i] - np.sum(uts[1:T][i]))
        res -= temp
        res -= self.o3 * abs(self.phi(p_t) - np.sum(uts[t0][:N]))

        return res


    def train_minibatch(self, inputs, state, T=10):
        batch_size = 256
        (_, count) = inputs.shape
        loss_fn = nn.MSELoss()
        C_t = 0
        us = np.array([])
        cts = np.array([])

        for i in range(T):
            for j in range(0, count, 1):
                # agg knows f1...ft initial state x1
                x = inputs[:, j]
                a_t_j = x[0]
                d_t_j = x[1]
                e_t_j = x[2]
                # These
                # functions
                # represent
                # the
                # prices
                # paid
                # for electricity at different locations on the grid at different times during the year

                a_t_j_timestamp = (time.mktime(time.strptime(a_t_j, "%a, %d %b %Y %H:%M:%S %Z")) - time.mktime(time.strptime(self.min_timestamp, "%a, %d %b %Y %H:%M:%S %Z"))) / 3600
                d_t_j_timestamp = (time.mktime(time.strptime(d_t_j, "%a, %d %b %Y %H:%M:%S %Z")) - time.mktime(time.strptime(self.min_timestamp, "%a, %d %b %Y %H:%M:%S %Z"))) / 3600

                xt = [d_t_j_timestamp, e_t_j]
                u = 0


                s_t = self.calculate_s_t(e_t_j,d_t_j_timestamp, u, xt)
                x_t = [d_t_j_timestamp - self.time_unit,e_t_j - s_t ]
                K = np.array([])


                # update of state
                e_t_j = float(x[2]) - s_t
                d_t_j_timestamp -= self.time_unit
                x_t = [d_t_j_timestamp, e_t_j]

                u_t = -K @ s_t

                # to learn phi function we use actor critic architecture

                # Initialize actor and critic networks
                state_dim = 4
                action_dim = 2
                actor = Actor(state_dim, action_dim,3)
                critic = Critic(state_dim, action_dim,3)

                # Define optimizer for actor and critic networks
                actor_optim = optim.Adam(actor.parameters(), lr=0.001)
                critic_optim = optim.Adam(critic.parameters(), lr=0.001)

                # Initialize replay buffer
                replay_buffer = []

                # Define hyperparameters
                # A
                # typical
                # value
                # for gamma is between 0.9 and 0.99, but it can be adjusted depending on the time horizon of the problem
                # A typical value for tau is between 0.01 and 0.001, but it can also be adjusted depending on the problem and the desired stability of the training process.In practice, researchers and practitioners often perform a grid search or use other hyperparameter tuning techniques to find suitable values for these parameters.

                gamma = 0.99
                tau = 0.001
                batch_size = 32

                # Reset environment to initial state
                state = np.zeros(state_dim)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_tensor = actor(state_tensor)
                action = action_tensor.detach().numpy()[0]



                # operator knows costs c1...ct and feedback  p1..pt
                c_t = self.psi(i % 24)
                u_t = 0

                us = np.append(us, u_t)
                cts = np.append(cts, c_t)


                predicted_agg_flexibility = 0
                actual_agg_flexibility = 0
                # mse_error =

                # mean, log_std = self.forward(curr_state)
                # action = self.sample_action(mean, log_std)
                # # Compute loss and backward pass
                # loss = loss_fn(mean, x)
                # self.optimizer.zero_grad()
                # loss.backward()
                # self.optimizer.step()
        return np.sum(cts)

