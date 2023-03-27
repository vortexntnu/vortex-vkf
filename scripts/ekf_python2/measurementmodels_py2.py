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

        z = np.matmul(self.Rot_wb.T, (x[0:3] - self.p_wb[0:3]))
        z = np.append(z, x[3])

        return z

    def H(self, x):
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        
        H_4x4 = [Rot_3x3, 0 
                 0, 0, 0, 1]
        """

        H = block_diag(self.Rot_wb.T, 1)
        return H

    def R(self, x):
        """Calculate the measurement covariance matrix at x in sensor_state."""

        n = len(self.sigma_z)

        R = (self.sigma_z**2) * np.eye(n)

        return R


class measurement_linear_landmark(MeasurementModel):
    """The model is a sensor agnostic landmark pose measurement model
    state:  x = [pw_wl, eulers_lw] ^ T   a 6x1 vector of position and euler angles
    measurement:    z = [ps_sl, eulers_ls] a 6x1 vector, measurement of full state in sensor frame
    h = euler_from_matrix (R_sl)
    H = diag(R_sw dh / deulers, a 6x6 matrix. Use lambda function Jac_lam to evaluate numerically.
    
    """

    def __init__(self, sigma_sensor):

        self.sigma_z = sigma_sensor

    def h(self, x):
        """Predict measurement through the non-linear vector field h given the
        state prediction x
        x = [pw_wg, gamma_wg] ^ T
        z = [pb_bg, gamma_wg]
        """
        n = np.shape(x)[0]
        z = np.matmul(np.eye(n), x)
        return z

    def H(self, x):
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        
        x = [pos_x, pos_y , pos_z , phi_x , theta_y , psi_z]
        H_6x6 = [Rot_3x3, 0 
                  0, matrix2euler_jacobian]
        """

        n = np.shape(x)[0]
        H = np.eye(n)
        return H

    def R(self, x):
        """Calculate the measurement covariance matrix at x in sensor_state."""

        n = np.shape(self.sigma_z)[0]
        R = (self.sigma_z**2) * np.eye(n)

        return R


class LTV_full_measurement_model(MeasurementModel):

    def __init__(self, sigma_sensor, pos, Rot):
        self.sigma_z = sigma_sensor
        self.p_wc = pos
        self.Rot_wc = Rot

    def h(self, x):
        """Predict measurement through the non-linear vector field h given the
        state x
        x = [pw_wg, eulers_world] ^ T
        z = [pb_bg, eulers_world]
        """

        z = np.matmul(self.Rot_wc.T, (x[0:3] - self.p_wc[0:3]))

        return np.append(z, x[3:])

    def H(self, x):
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        
        H_3x3 = Rot_3x3
        
        """
        n = len(x)
        H = np.eye(n)
        H[:3, :3] = self.Rot_wc.T

        return H

    def R(self, x):
        """Calculate the measurement covariance matrix at x in sensor_state."""

        return np.diag(self.sigma_z**2)
