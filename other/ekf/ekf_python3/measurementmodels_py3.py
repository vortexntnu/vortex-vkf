# Imports

from typing import Any, Dict
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from scipy.linalg import block_diag

# Measurement models interface declaration


@dataclass
class MeasurementModel:

    def h(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the noise free measurement location at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def H(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError

    def R(self, x: ndarray, **kwargs) -> ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state.
        Args:
            x (ndarray): state
        """
        raise NotImplementedError


@dataclass
class CartesianPosition2D(MeasurementModel):
    sigma_z: float

    def h(self, x: ndarray) -> ndarray:
        """Calculate the noise free measurement location at x in sensor_state.
        """

        # TODO replace this with your own code
        first_row = [1, 0, 0, 0]
        second_row = [0, 1, 0, 0]

        H = np.array([first_row, second_row])
        x_h = H @ x

        return x_h

    def H(self, x: ndarray) -> ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state."""

        # TODO replace this with your own code
        first_row = [1, 0, 0, 0]
        second_row = [0, 1, 0, 0]

        H = np.array([first_row, second_row])

        return H

    def R(self, x: ndarray) -> ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state."""

        # TODO replace this with your own code
        identity = np.array([[1, 0], [0, 1]])
        R = (self.sigma_z**2) * identity

        return R


@dataclass
class NED_range_bearing(MeasurementModel):
    sigma_z: float
    pw_wc: ndarray
    Rot_wb: ndarray

    def h(self, x: ndarray) -> ndarray:
        """Predict measurement through the non-linear vector field h given the
        state x
        x = [pw_wg, gamma_wg] ^ T
        z = [pb_bg, gamma_wg]
        """

        z = np.array([self.Rot_wb.T @ (x[0] - self.pw_wc), x[1]])

        return z

    def H(self, x: ndarray) -> ndarray:
        """Calculate the measurement Jacobian matrix at x in sensor_state.
        
        H_4x4 = [Rot_3x3, 0 
                 0, 0, 0, 1]
        """

        H = block_diag(self.Rot_wb, 1)
        return H

    def R(self, x: ndarray) -> ndarray:
        """Calculate the measurement covariance matrix at x in sensor_state."""

        # TODO replace this with your own code
        n = len(self.sigma_z)

        R = (self.sigma_z**2) * np.eye(n)

        return R
