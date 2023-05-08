import math
import time
import numpy as np
import torch as to
import torch.nn as nn
import torchvision
from util import *
from actcrit import *
import matplotlib.pyplot as plt

class PPCController_test(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out, alpha, min_timestamp, time_unit=12):
        super(PPCController_test, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hid)
        self.fc2 = nn.Linear(dim_hid, dim_hid)
        self.fc3 = nn.Linear(dim_hid, dim_out)
        self.relu = nn.ReLU()
        self.optimizer = to.optim.Adam(self.parameters(),lr=alpha)
        self.o1 = 0.1
        self.o2 = 0.2
        self.o3 = 2
        self.W = 54
        self.num_of_chargers = self.W
        self.u = 50
        self.log_std_min = -20
        self.log_std_max = 2
        self.alpha = alpha
        self.min_timestamp = min_timestamp
        self.time_unit = 1
        self.control_space = [i for i in range(0,151,15)]

    # normalization of time by earliest time and hours
    # for example first session is 1.1.2016 13:00 - value = 0
    # second session 1.1.2016 14:00 value = 1
    def process_inputs(self, inputs,r_t_j):
        (_, count) = inputs.shape
        new_arr = []
        for j in range(0, count, 1):
            x = inputs[:, j]
            a_t_j = x[0]
            d_t_j = x[1]
            e_t_j = float(x[2])
            a_t_j_timestamp = (time.mktime(time.strptime(a_t_j, "%a, %d %b %Y %H:%M:%S %Z")) - time.mktime(
                time.strptime(self.min_timestamp, "%a, %d %b %Y %H:%M:%S %Z"))) / 3600
            d_t_j_timestamp = (time.mktime(time.strptime(d_t_j, "%a, %d %b %Y %H:%M:%S %Z")) - time.mktime(
                time.strptime(self.min_timestamp, "%a, %d %b %Y %H:%M:%S %Z"))) / 3600
            new_arr.append([a_t_j_timestamp, d_t_j_timestamp, e_t_j, r_t_j])
        return new_arr

    def min_function(self, t):
        t = t % 24
        return 1 - (t / 24)

    # MPE is testing the ratio of s(j) and e(j), - violation of condition 4d)
    # rate of undelivered energy
    def mpe(self, T, N, decisions, energy_requests, L = 1):
        res = 0
        for k in range(1,L+1):
            for t in range(T+1):
                for j in range(N):
                    res += decisions[t][j] / (np.sum(energy_requests))
        res = 1 - res
        return res

    def mpe_updated(self, T, N, decisions, energy_requests, L=1):
        return 1 - (np.sum(decisions) / np.sum(energy_requests))
    # MSE measures - violation of condition 4c)
    def mse(self):
        ...
    def get_other(self,inputs):

        a_t_js = []
        d_t_js = []
        e_t_js = []
        r_t_js = []
        for input in inputs:
            a_t_js.append(input[0])
            d_t_js.append(input[1])
            e_t_js.append(input[2])
            r_t_js.append(input[3])
        return a_t_js, d_t_js, e_t_js, r_t_js
    def histogram_costs(self, time_intervals, costs_array, time):
        plt.bar(time_intervals, costs_array)
        plt.ylabel("Delivered Energy costs")
        plt.xlabel("Time")
        plt.title("Energy Costs")
        plt.xticks(range(len(time)), time)
        plt.show()
    def histogram_charging_evs(self, time_intervals, charging_evs, time, interval_num_evs):
        plt.bar(time_intervals, charging_evs)
        plt.xlabel("Time")
        plt.ylabel("Number of charging electric vehicles")
        plt.title("Charging of electric vehicles")
        plt.xticks(range(len(time)), time)
        plt.yticks(range(len(interval_num_evs)),interval_num_evs)
        plt.show()
    def employ_schedule(self, ideal_schedule, energy_requests, time_horizon, num_of_vehicles):
        costs = {}
        charging_evs = {}
        updated_energy_requests = energy_requests.copy()
        decisions = np.zeros((time_horizon + 1, num_of_vehicles))
        for t in range(time_horizon + 1):
            costs[t] = 0
            charging_evs[t] = 0
        overall_costs = 0
        for key in ideal_schedule.keys():
            value = ideal_schedule[key]
            if value != 0:
                starting_time = value[0]
                ending_time = value[1]
                remaining_time = ending_time - starting_time
                provided_energy = energy_requests[key]
                updated_energy_requests[key] -= provided_energy
                decisions[starting_time][key] = provided_energy
                costs[starting_time] += self.min_function(starting_time) * provided_energy
                overall_costs += self.min_function(starting_time) * provided_energy
                charging_evs[starting_time] += 1
        return costs, overall_costs, charging_evs, decisions, updated_energy_requests

    def get_state(self, time_intervals, arrival_times, departure_times, energy_requests):
        x_t_states = {}
        for time in time_intervals:
            x_t_states = x_t_states.get(time, [])
            for car_id in range(len(arrival_times)):
                if arrival_times[car_id] <= time <= departure_times[car_id]:
                    x_t_states[time].append(departure_times[car_id],energy_requests[car_id])
    def train_minibatch(self, inputs,rj, T = 10):
        (_, count) = inputs.shape
        loss_fn = nn.MSELoss()
        new_inputs = self.process_inputs(inputs,rj)
        optimizer = torch.optim.Adam(self.parameters(),lr=0.001)
        a_t_js, d_t_js, e_t_js, r_t_js = self.get_other(new_inputs)
        arrival_times,\
            departure_times,\
            energy_requests,\
            peak_charging_rates = a_t_js, d_t_js, e_t_js, r_t_js
        time_horizon = T
        time_intervals = np.arange(time_horizon + 1)
        num_evs = len(new_inputs)
        interval_num_evs = np.arange(num_evs + 1)

        N = num_evs

        all_schedules = []

        ideal_schedule = {}
        for interval in time_intervals:
            chargers_per_time = np.ones(time_horizon + 1) * self.num_of_chargers
            schedule, energy_delivered, charging_ev = self.LLF(arrival_times, departure_times, energy_requests,
                                                               peak_charging_rates, interval, time_horizon,
                                                               chargers_per_time)
            all_schedules.append(schedule)
            for key in schedule.keys():
                value = schedule[key]
                if value != 0:
                    # giving more energy in less than hour is ineffective because of min function
                    if value[1] - value[0] >= 1:
                        ideal_schedule[key] = value
            # costs += self.min_function(interval) * energy_delivered
            # costs_array.append(costs)
            # charging_evs.append(charging_ev)

            # edfs = self.EDF(d_t_js, a_t_js, e_t_js, rj, interval,time_horizon)
            # for time_unit in range(interval)
        costs, all_costs, charging_evs, decisions, updated_energy_requests = self.employ_schedule(ideal_schedule, energy_requests, T, num_evs)
        mpe1 = self.mpe(T, N, decisions, energy_requests)
        mpe2 = self.mpe_updated(T, N, decisions, energy_requests)
        print('Basic version of MPE = ',mpe1)
        print('Optimalized version of MPE =',mpe2)
        self.histogram_costs(time_intervals, costs.values(), time_intervals)
        self.histogram_charging_evs(time_intervals, charging_evs.values(), time_intervals, interval_num_evs)


    def LLF(self,
            arrival_times,
            departure_times,
            energy_requests,
            peak_charging_rate,
            t,
            T,
            chargers_per_time):
        '''
        arrival_times = a_t(j) j = 1...N
        departure_times = d_t(j) j = 1...N
        energy requested = e_t(j) j = 1...N
        t = current time from {1,...,T} at each time t je provide energy
        T = time horizon
        chargers_per_time = available chargers at each time t
        '''
        # assert len(djs) == len(ejs), 'Three arrays djs, ejs, rjs must have same length.'
        n = len(departure_times)
        lax_arr = []
        for car_id in range(n):
            # laxity is even minus number possibly, but mostly from R^+
            laxity = departure_times[car_id] - energy_requests[car_id] / peak_charging_rate[car_id]
            lax_arr.append([car_id, laxity, arrival_times[car_id], departure_times[car_id],energy_requests[car_id]])
        sorted_lax_array = sorted(lax_arr, key=lambda x: x[1])
        schedule = {}
        energy_delivered = 0
        current_time = t
        charging_ev = 0
        while sorted_lax_array:
            car_id, least_laxity_task, arrival_time, departure_time, energy_request = sorted_lax_array.pop(0)
            # if car has not arrived yet
            if arrival_time > current_time:
                schedule[car_id] = 0
                continue
            # if car has already left
            if departure_time < current_time:
                schedule[car_id] = 0
                continue
            # if user of car got all energy requested
            if energy_request == 0:
                schedule[car_id] = None
                continue
            # if there is no charger for car left
            if chargers_per_time[current_time] <= 0:
                schedule[car_id] = 0
                continue

            start_time = t
            end_time = departure_time
            schedule[car_id] = [start_time, end_time]
            energy_delivered += energy_request
            charging_ev += 1
            chargers_per_time[current_time] -= 1
            # schedule.append([car_id, start_time, end_time])

        return schedule, energy_delivered, charging_ev

    def EDF(self,
            arrival_times,
            departure_times,
            energy_requests,
            peak_charging_rate,
            t,
            T,
            chargers_per_time
            ):
        # assert len(djs) == len(ejs), 'Three arrays djs, ejs, rjs must have same length.'
        n = len(departure_times)
        deadline_arr = []
        for car_id in range(n):
            deadline_arr.append([car_id, arrival_times[car_id], departure_times[car_id], energy_requests[car_id]])
        sorted_deadline_arr = sorted(deadline_arr,key=lambda x: x[1])
        current_time = t
        schedule = {}
        charging_ev = 0
        energy_delivered = 0
        while sorted_deadline_arr:
            car_id, arrival_time, departure_time, energy_request = deadline_arr.pop(0)
            if arrival_time > current_time:
                schedule[car_id] = 0
                continue
            if departure_time > current_time:
                schedule[car_id] = 0
                continue
            if energy_request == 0:
                schedule[car_id] = 0
                continue
            if chargers_per_time[current_time] <= 0:
                schedule[car_id] = 0
                continue
            start_time = t
            end_time = departure_time
            energy_delivered += energy_request
            charging_ev += 1
            schedule[car_id] = [start_time, end_time]
            chargers_per_time[current_time] -= 1

        return schedule, energy_delivered, charging_ev

