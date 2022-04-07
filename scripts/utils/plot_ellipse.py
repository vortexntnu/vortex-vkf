from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from typing import Union
from matplotlib.transforms import Affine2D


def plot_cov_ellipse2d(
    ax: plt.Axes,
    mean: np.ndarray = np.zeros(2),
    cov: np.ndarray = np.eye(2),
    n_sigma: float = 1,
    *,
    edgecolor: Union[tuple, str] = "C0",
    facecolor: Union[tuple, str] = "none",
    **kwargs,  # extra Ellipse keyword arguments
) -> Ellipse:
    """Plot a n_sigma covariance ellipse centered in mean into ax."""
    ell_trans_mat = np.zeros((3, 3))
    ell_trans_mat[:2, :2] = np.linalg.cholesky(cov)
    ell_trans_mat[:2, 2] = mean
    ell_trans_mat[2, 2] = 1

    ell = Ellipse(
        (0.0, 0.0),
        2.0 * n_sigma,
        2.0 * n_sigma,
        edgecolor=edgecolor,
        facecolor=facecolor,
        **kwargs,
    )
    trans = Affine2D(ell_trans_mat)
    ell.set_transform(trans + ax.transData)
    return ax.add_patch(ell)
