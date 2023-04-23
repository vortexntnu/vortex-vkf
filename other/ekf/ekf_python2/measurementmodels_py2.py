#!/usr/bin/env python

# Imports
import numpy as np
from numpy import ndarray
from scipy.linalg import block_diag

# Measurement models interface declaration


class MeasurementModel:

    def h(self, x, **kwargs):
        """Calculate the noise free measurement location at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def H(self, x, **kwargs):
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def R(self, x, **kwargs):
        """Calculate the measurement covariance matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError


#@dataclass
#class CartesianPosition2D(MeasurementModel):
#    sigma_z: float
#
#    def h(self, x: ndarray) -> ndarray:
#        """Calculate the noise free measurement location at x in sensor_state.
#        """
#
#        # TODO replace this with your own code
#        first_row = [1, 0, 0, 0]
#        second_row = [0, 1, 0, 0]
#
#        H = np.array([first_row, second_row])
#        x_h = H @ x
#
#        return x_h
#
#    def H(self, x: ndarray) -> ndarray:
#        """Calculate the measurement Jacobian matrix at x in sensor_state."""
#
#        # TODO replace this with your own code
#        first_row = [1, 0, 0, 0]
#        second_row = [0, 1, 0, 0]
#
#        H = np.array([first_row, second_row])
#
#        return H
#
#    def R(self, x: ndarray) -> ndarray:
#        """Calculate the measurement covariance matrix at x in sensor_state."""
#
#        # TODO replace this with your own code
#        identity = np.array([[1, 0], [0, 1]])
#        R = (self.sigma_z**2) * identity
#
#        return R


class NED_range_bearing(MeasurementModel):

    def __init__(self, sigma_sensor, pos, Rot):
        self.sigma_z = sigma_sensor
        self.p_wb = pos
        self.Rot_wb = Rot

    def h(self, x):
        """Predict measurement through the non-linear vector field h given the
        state x
        x = [pw_wg, gamma_wg] ^ T
        z = [pb_bg, gamma_wg]
        """

        #z = self.Rot_wb.T @ (x[0:3] - self.p_wb[0:3])

        z = np.matmul(self.Rot_wb, (x[0:3] - self.p_wb[0:3]))
        z = np.append(z, x[3])

        return z

    def H(self, x):
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        
        H_4x4 = [Rot_3x3, 0 
                 0, 0, 0, 1]
        """

        H = block_diag(self.Rot_wb, 1)
        return H

    def R(self, x):
        """Calculate the measurement covariance matrix at x in sensor_state."""

        # TODO replace this with your own code
        n = len(self.sigma_z)

        R = (self.sigma_z**2) * np.eye(n)

        return R
