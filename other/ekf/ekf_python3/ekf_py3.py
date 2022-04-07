
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


from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la

from config import DEBUG
from dynamicmodels_py3 import DynamicModel
from measurementmodels_py3 import MeasurementModel
from gaussparams_py3 import MultiVarGaussian

# The EKF
@dataclass
class EKF:
    dynamic_model: DynamicModel
    sensor_model: MeasurementModel

    def predict(self,
                state_upd_prev_gauss: MultiVarGaussian,
                Ts: float,
                ) -> MultiVarGaussian:
        """Predict the EKF state Ts seconds ahead."""
        x_prev, P_prev = state_upd_prev_gauss

        Q = self.dynamic_model.Q(x_prev, Ts)
        F = self.dynamic_model.F(x_prev, Ts)

        x_pred = self.dynamic_model.f(x_prev, Ts)
        P_pred = F @ P_prev @ F.T + Q

        state_pred_gauss = MultiVarGaussian(x_pred, P_pred)

        return state_pred_gauss

    def update(self,
               z: np.ndarray,
               state_pred_gauss: MultiVarGaussian,
               ) -> MultiVarGaussian:
        """Given the prediction and measurement, find innovation then 
        find the updated state estimate."""

        x_pred, P = state_pred_gauss

        n = len(x_pred)

        #if measurement_gauss is None:
            #measurement_gauss = self.predict_measurement(state_pred_gauss)

        H = self.sensor_model.H(x_pred)
        R = self.sensor_model.R(x_pred)

        z_pred = self.sensor_model.h(x_pred)
        S = H @ P @ H.T + R 

        inov = z - z_pred
        W = P @ H.T @ np.linalg.inv(S)

        x_upd = x_pred + W @ inov
        P_upd = (np.eye(n) -W @ H)@ P

        measure_pred_gauss = MultiVarGaussian(z_pred, S)
        state_upd_gauss = MultiVarGaussian(x_upd, P_upd)


        return state_upd_gauss, measure_pred_gauss

    def step_with_info(self,
                       state_upd_prev_gauss: MultiVarGaussian,
                       z: np.ndarray,
                       Ts: float,
                       ) -> tuple([MultiVarGaussian,
                                  MultiVarGaussian,
                                  MultiVarGaussian]):
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

    def step(self,
             state_upd_prev_gauss: MultiVarGaussian,
             z: np.ndarray,
             Ts: float,
             ) -> MultiVarGaussian:

        _, _, state_upd_gauss = self.step_with_info(state_upd_prev_gauss,
                                                    z, Ts)
        return state_upd_gauss
