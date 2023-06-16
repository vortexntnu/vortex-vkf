#!/usr/bin/env python
## Addapted for use for Vortex NTNU from the course TTK4250. Credit for the underlying code goes to:
## @author: Lars-Christian Tokle, lars-christian.n.tokle@ntnu.no ##
"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z are multiple measurements so that z = Z[k] at a given time step k
v is the innovation z - h(x)
S is the innovation covariance
"""

## EKF Algorith notation:
# x_prev = mean of previous state posterior pdf
# P_prev = covariance of previous state posterior pdf

# x_pred = kinematic prediction through dynamic model. Also called x_bar in literature
# P_pred = predicted prior covariance. Also called P_bar in the literature

import numpy as np
import scipy.linalg as la

from config import DEBUG
from ekf_python3.dynamicmodels_py2 import DynamicModel
from ekf_python3.measurementmodels_py2 import MeasurementModel
from ekf_python3.gaussparams_py2 import MultiVarGaussian

# The EKF


class EKF:

    def __init__(self, DM, MM):

        self.dynamic_model = DM
        self.sensor_model = MM

    def predict(self, state_upd_prev_gauss, Ts):
        """Predict the EKF state Ts seconds ahead."""
        x_prev, P_prev = state_upd_prev_gauss

        Q = self.dynamic_model.Q(x_prev, Ts)
        F = self.dynamic_model.F(x_prev, Ts)

        x_pred = self.dynamic_model.f(x_prev, Ts)

        P_pred = np.matmul(F, np.matmul(P_prev, F.T)) + Q

        state_pred_gauss = MultiVarGaussian(x_pred, P_pred)

        return state_pred_gauss

    def update(self, z, state_pred_gauss):
        """Given the prediction and measurement, find innovation then 
        find the updated state estimate."""

        x_pred, P = state_pred_gauss

        n = len(x_pred)

        H = self.sensor_model.H(x_pred)
        R = self.sensor_model.R(x_pred)

        z_pred = self.sensor_model.h(x_pred)
        S = np.matmul(H, np.matmul(P, H.T)) + R

        inov = z - z_pred
        W = np.matmul(P, np.matmul(H.T, np.linalg.inv(S)))

        x_upd = x_pred + np.matmul(W, inov)
        P_upd = np.matmul((np.eye(n) - np.matmul(W, H)), P)

        measure_pred_gauss = MultiVarGaussian(z_pred, S)
        state_upd_gauss = MultiVarGaussian(x_upd, P_upd)

        return state_upd_gauss, measure_pred_gauss

    def step_with_info(self, state_upd_prev_gauss, z, Ts):
        """
        Predict ekfstate Ts units ahead and then update this prediction with z.

        Returns:
            state_pred_gauss: The state prediction
            measurement_pred_gauss: 
                The measurement prediction after state prediction
            state_upd_gauss: The predicted state updated with measurement
        """

        state_pred_gauss = self.predict(state_upd_prev_gauss, Ts)

        state_upd_gauss, measure_pred_gauss = self.update(z, state_pred_gauss)

        return state_pred_gauss, measure_pred_gauss, state_upd_gauss

    def step(self, state_upd_prev_gauss, z, Ts):

        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss, z,
                                                    Ts)
        return state_upd_gauss
