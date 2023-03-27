#!/usr/bin/env python
import numpy as np
from numpy import ndarray
import scipy.linalg as la
from gaussparams_py2 import MultiVarGaussian
from config import DEBUG
#from typing import Sequence


def get_NIS(z_pred_gauss, z):
    """Calculate the normalized innovation squared (NIS), this can be seen as 
    the normalized measurement prediction error squared. 
    See (4.66 in the Sensor Fusion book.) 
    Tip: use the mahalanobis_distance method of z_pred_gauss, (3.2) in the book

    Args:
        z_pred_gauss (MultiVarGaussian): predigted measurement gaussian
        z (ndarray): measurement

    Returns:
        NIS (float): normalized innovation squared
    """

    z_pred, S = z_pred_gauss

    v = z - z_pred

    NIS = np.matmul(v.T, np.matmul(np.linalg.inv(S), v))

    return NIS


def get_NEES(x_gauss, x_gt):
    """Calculate the normalized estimation error squared (NEES)
    See (4.65 in the Sensor Fusion book). 
    Tip: use the mahalanobis_distance method of x_gauss, (3.2) in the book

    Args:
        x_gauss (MultiVarGaussian): state estimate gaussian
        x_gt (ndarray): true state

    Returns:
        NEES (float): normalized estimation error squared
    """

    x_hat, P_hat = x_gauss

    err = x_hat - x_gt

    NEES = np.matmul(err.T, np.matmul(np.linalg.inv(P_hat), err))

    return NEES


def get_ANIS(z_pred_gauss_data, z_data):
    """Calculate the average normalized innovation squared (ANIS)
    Tip: use get_NIS

    Args:
        z_pred_gauss_data (Sequence[MultiVarGaussian]): Sequence (List) of 
            predicted measurement gaussians
        z_data (Sequence[ndarray]): Sequence (List) of true measurements

    Returns:
        ANIS (float): average normalized innovation squared
    """

    NIS_arr = np.array([])

    for i in range(len(z_data)):

        NIS = get_NIS(z_pred_gauss_data[i], z_data[i])
        np.append(NIS_arr, NIS)

    ANIS = np.average(NIS_arr)

    return ANIS


def get_ANEES(x_upd_gauss_data, x_gt_data):
    """Calculate the average normalized estimation error squared (ANEES)
    Tip: use get_NEES

    Args:
        x_upd_gauss_data (Sequence[MultiVarGaussian]): Sequence (List) of 
            state estimate gaussians
        x_gt_data (Sequence[ndarray]): Sequence (List) of true states

    Returns:
        ANEES (float): average normalized estimation error squared
    """
    NEES_arr = np.array([])

    for i in range(len(x_gt_data)):

        NEES = get_NEES(x_upd_gauss_data[i], x_gt_data[i])
        np.append(NEES_arr, NEES)

    ANEES = np.average(NEES_arr)

    return ANEES


# def get_RMSE(x_upd_gauss_data: Sequence[MultiVarGaussian],
#              x _gt_data: Sequence[ndarray]):

#              #TODO
