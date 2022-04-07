#!/usr/bin/env python
from sqlite3 import enable_shared_cache
from ekf_python2.ekf_py2 import EKF
from ekf_python2.dynamicmodels_py2 import landmark_gate
from ekf_python2.measurementmodels_py2 import NED_range_bearing
import numpy as np
from numpy import cos
from numpy import sin
from random import randint
from ekf_python2.gaussparams_py2 import MultiVarGaussian
import matplotlib.pyplot as plt

rad2deg = 180/np.pi
deg2rad = np.pi/180

## Sim Parameters ##

N = 1000
dt = 0.1

def rot_mat (euler_angles):

    phi, theta, psi = euler_angles[0], euler_angles[1], euler_angles[2]
    
    rot_mat = np.array([[], [], []])

def test(N, dt):

    sigma_sensor = np.array([5, 5, 3, 0.5])
    sigma_process = np.array([0.005, 0.005, 0.005, 0.1])

    pw_wb = np.array([3, 2, 1.5, 0])

    gate_gt = np.array([3, 3, 0, np.pi])

    gate_gt_c = gate_gt - pw_wb

    Rot_wb = np.eye(3)

    sensor_model = NED_range_bearing(sigma_sensor, pw_wb, Rot_wb)
    gate_model = landmark_gate(sigma_process)

    my_filter = EKF(gate_model, sensor_model)

    z_set, Ts_set = generate_measurement_data(gate_gt_c, N, dt)

    x_hat0 = z_set[0]
    P_hat0 = np.diag(sigma_sensor)

    hats = []
    hats.append(MultiVarGaussian(x_hat0, P_hat0))

    t = [0]

    for i in range(len(z_set)-1):
        t.append(t[i] + Ts_set[i])

        hat = my_filter.step(hats[i], z_set[i+1], Ts_set[i])
        hats.append(hat)

    return t, hats, z_set 


def generate_measurement_data(x_gt_c, N, dt):

    Ts = dt

    noise = np.array([0.05, 0.05, 0.05, 0.001])
    noise_bad = np.array([1.5, 1.5, 1.5, np.pi/8])

    z_set = []
    Ts_set = []

    z_set.append(x_gt_c + np.random.normal(0, noise**2))

    for i in range(N):
        check_detect = randint(0,100)
        check_bad_measurement = randint(0, 100)
        if check_detect > 20:

            Ts_set.append(Ts)
            Ts = dt

            if check_bad_measurement > 10:
                z_n = x_gt_c + np.random.normal(0, noise**2)
                z_set.append(z_n)
            else:
                z_n = x_gt_c + np.random.normal(0, noise_bad**2)
                z_set.append(z_n)
        else:
            Ts += dt

    return z_set , Ts_set

def plotting(t, hats, z_set):

    x_hats = [hats[i].mean for i in range(len(hats))]
    x_pos = [x_hats[i][0] for i in range(len(x_hats))]
    y_pos = [x_hats[i][1] for i in range(len(x_hats))]

    print(x_hats[-1])

    plt.figure(1)
    for i in range(len(z_set)):
        plt.scatter(z_set[i][0], z_set[i][1], 2, "r")
        plt.scatter(x_hats[i][0], x_hats[i][1], 2, "b")

    plt.show()

    plt.figure(2)
    plt.plot(t, x_pos)
    plt.show()
    
    plt.figure(3)
    plt.plot(t, y_pos)
    plt.show()
    
    #fig = plt.figure(figsize = (8,8))
    #fig = fig.scatter(xy_measurements)




t, hats, z_data = test(N, dt)


plotting(t, hats, z_data)