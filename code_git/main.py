import json
import numpy as np
import matplotlib.pyplot as plt
from util import *
from controller2 import *
# from contoller import *


# Opening JSON file
DATA_FILE_CALTECH = 'acndata_sessions_acn.json'
DATA_FILE_JPL = 'acndata_sessions_jpl.json'

data,data2 = None,None
with open(DATA_FILE_CALTECH) as f:
    data = json.load(f)
with open(DATA_FILE_JPL) as f2:
    data2 = json.load(f2)






U = 10 #
W = 54 # number of chargers
X = 108 #state space R^108
time_interval = 12#time interval 12 minutes
power_rating = 150

laxitiy = 0
learning_rate = 3*10**-4
discount_factor = 0.5
relay_buffer_size = 10**6
Beta_param = 1*10**3 - 1*10**6




# a(j) denotes connecting time
# d(j) denotes disconnection time (departure time)
# e(j) - total energy delivered
# r(j) - peak charging rate
start_measurement_time = data['_meta']['start']
end_measurement_time = data['_meta']['end']
print('start ',start_measurement_time)
print('end ',end_measurement_time)
voltage = 220  # volts

# Default maximum charging rate for each EV battery.
default_battery_power = 32 * voltage / 1000  # kW

r_j = default_battery_power
arr_e = np.array([])

agg_xs_t = np.array([])
a_j,d_j,e_j = np.array([]), np.array([]), np.array([])
for i in data['_items'][:10]:
    a_j = np.append(a_j,i['connectionTime'])
    d_j = np.append(d_j,i['disconnectTime'])
    e_j = np.append(e_j,float(i['kWhDelivered']))
    c = np.array([i['connectionTime'],float(i['kWhDelivered'])])
    agg_xs_t = np.append(agg_xs_t,c)

min_timestamp = min(a_j, key=lambda x: time.mktime(time.strptime(x, "%a, %d %b %Y %H:%M:%S %Z")))


arr_e2 = np.array([])
agg2_xs_t = np.array([])
a_j2, d_j2, e_j2 = np.array([]), np.array([]), np.array([])
for i in data2['_items'][:10]:
    a_j2 = np.append(a_j2, i['connectionTime'])
    d_j2 = np.append(d_j2, i['disconnectTime'])
    e_j2 = np.append(e_j2, float(i['kWhDelivered']))
    c = [i['connectionTime'], float(i['kWhDelivered'])]
    agg2_xs_t = np.append(agg2_xs_t, c)

min_timestamp2 = min(a_j, key=lambda x: time.mktime(time.strptime(x, "%a, %d %b %Y %H:%M:%S %Z")))


input1 = np.array([a_j, d_j, e_j])
input2 = np.array([a_j2, d_j2, e_j2])








def laxity_value(d_t,e_t,r):
    return (d_t-e_t)/r

    #
    # print(i)
T = 10
Ct = 0

#
#


'''
cost/time graph comparing offline,MPC,PPC - MPC is higher in costs and PPC is straight line
we measure by MSE error of undelivered energy in (4d)
hard constrains - (4a),(4b),(4e) - depend only on scheduling policy 
there are two violation error metrics that we use 
mean percentage error 
online-learning
combination of MPC(online optimization) and using MEF as penalty term
feasibility information 


trade of f between ensuring future flexibility and minimizing current systems costs



calculation of control sequence minimising an objective function
use of model to predict output in future

various MPC algorithms also called LRPC differ only in model used to represent 
process and the noises and the cost function to be minimized



MPC strategy 
the future outputs at horizon N  called the prediction horizon are predicted
at each instant t


y(t+k/t) k = 1...N depend on past inputs/outputs up to y(t) and on future 
control signals u(t+k/t) k = 0...N-1

the set of future control signals is optimized to keep process as close as 
possible to reference trajectory w(t + k)


these things can differ in various MPC implementations:
predictiom model
objective function
obtaining control law


process model 

different objective functions to obtain control law


in order to obtain u(t+k/t) it is necessary to minimize function J

analytical solution can be used if there is quadratic criterion if 
the model is linear and there are no constraints, otherwise iterative 
solution is needed

N2 + N1 - 1 independent variables


Power electronics systems are inherently non-linear,
 meaning that their behavior cannot be accurately modeled using linear models
 . In particular, the switching dynamics of power converters, such as voltage 
 spikes and current ripple,
 make the system highly non-linear and difficult to control.

agg state x_t = 

T - total number of time slots 







s_t(j) - decision - energy delivered to user j depending on 
scheduling policy pi_t (earliest deadline first/ least laxity )
s_t = pi_t(u_t) = u_t is aggregate substation power 

the decision s_t updates aggregator state  in particular e_t(j)


e_t 

'''





# plt.plot([1, 2, 3, 4])
# plt.plot(arr_e)
# plt.ylabel('Number')
# plt.plot(arr_e2,label = 'jpl')
# plt.xlabel('kWh')
# plt.show()


names = ['caltech', 'jpl']
values = [len(arr_e), len(arr_e2)]



dim_hid = 256
dim_in = 3
dim_out = 2
# model1 = PPCController(dim_in, dim_hid, dim_out,  3*10**-4,min_timestamp)
# model1.train_minibatch(input1,agg_xs_t)

# Test the functions with some random input values
N = 4
L = 3
T = 2
xi = 1.5

sts = np.random.rand(L, T, N)
uts = np.random.rand(L, T)
# orig_res = model1.original_normalized_mse(sts, uts, N, L, T, xi)
# numpy_res = model1.numpy_normalized_mse(sts, uts, N, L, T, xi)
# print(orig_res)
# print(numpy_res)

L = 2
T = 3
N = 3
sts = np.array([[[1, 1, 1], [0, 0, 0], [1, 1, 1]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]]])
ej = np.array([0.5, 0.5])

# Calculate MPE using original_mpe function
# mpe = model1.original_mpe(sts, ej, L, T, N)

# Check that MPE is within bounds
# assert mpe >= 0 and mpe <= 1, f"MPE value of {mpe} is outside expected bounds"
# print(mpe)
# print("Test passed")# print('mpe2 = ',mpe2)

# 0 <= MPE <= 1

# model2 = PPCController(dim_in, dim_hid, dim_out, 3*10**-4)
# model2.train_minibatch(input2, agg2_xs_t)

#
#
# make_histogram(names, values)
# print(len(data))
# plt.show()

T1 = [0, 2, 6, 6]
T2 = [0, 2, 8, 8]
T3 = [0, 3, 10, 10]
t_arr = [T1, T2, T3]
def least_laxity_scheduling(t_arr):
    lv = np.zeros(len(t_arr))
    for t in range(len(t_arr)):
        ri = t_arr[t][0]
        Ci = t_arr[t][1]
        Di = t_arr[t][2]
        Ti = t_arr[t][3]
        lv[t] = Di - (ri + Ci)
    print('lv = ',lv)
    return lv



    print(lax_arr)



# least_laxity_scheduling(t_arr)
model = PPCController_test(dim_in, dim_hid, dim_out, 3*10**-4, min_timestamp);
model.train_minibatch(input1,r_j)

# LLF(d_j,e_j,r_j)


# LLF is an optimal algorithm because if a task set will pass utilization test then it is surely schedulable by LLF. Another advantage of LLF is that it some advance knowledge about which task going to miss its deadline. On other hand it also has some disadvantages as well one is its enormous computation demand as each time instant is a scheduling event. It gives poor performance when more than one task has least laxity.



