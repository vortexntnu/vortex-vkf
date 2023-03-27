from pathlib import Path
import numpy as np
from scipy.io import loadmat

from utils.sample_CT_trajectory import sample_CT_trajectory

data_path = Path(__file__).parents[2].joinpath("data/data_for_ekf.mat")


def load_data(usePregen=True, data_path=data_path, seed=None):
    if usePregen:
        loaded_data: dict = loadmat(str(data_path))
        N_data: int = int(loaded_data["K"])  # The number of time steps
        Ts: float = float(loaded_data["Ts"])  # The sampling time
        x_gt_data: np.ndarray = loaded_data["Xgt"].T  # grounexutd truth
        z_data: np.ndarray = loaded_data["Z"].T  # the measurements

    else:
        if seed:
            np.random.seed(seed)  # random seed can be set for repeatability

        # inital state distribution
        x0 = np.array([0, 0, 1, 1, 0])
        P0 = np.diag([50, 50, 10, 10, np.pi / 4])**2

        # model parameters
        sigma_a_true = 0.25
        sigma_omega_true = np.pi / 15
        sigma_z_true = 3

        # sampling interval a lenght
        N_data = 1000
        Ts = 0.1

        # get data
        x_gt_data, z_data = sample_CT_trajectory(N_data, Ts, x0, P0,
                                                 sigma_a_true,
                                                 sigma_omega_true,
                                                 sigma_z_true)
    return x_gt_data, z_data, Ts, N_data
