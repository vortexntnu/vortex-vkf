import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
import scipy
import numpy as np

plt.close("all")
mpl.rcParams['axes.grid'] = True

c_gt = 'g'
c_measurement = 'purple'
c_estimate = 'red'


def plot_trajectory_with_measurements(x_gt_series, z_series):
    fig, ax = plt.subplots()
    ax.scatter(*z_series.T, c=c_measurement, s=10, marker=".", label=r"$z$")
    ax.plot(*x_gt_series.T[:2], c=c_gt, alpha=0.9, label=r"$x_{gt}$")
    ax.set_title("Data")
    ax.legend()
    # show turnrate
    fig2, ax2 = plt.subplots(num=2, clear=True)
    ax2.plot(x_gt_series.T[4], c=c_gt)
    ax2.set_xlabel("time step")
    ax2.set_ylabel("turn rate")


def plot_ekf_trajectory(x_gt_series, x_hat_series, RMSE_pred, RMSE_upd,
                        sigma_a, sigma_z):
    fig, ax = plt.subplots(num=3, clear=True)
    ax.plot(*x_hat_series.T[:2], c=c_estimate, label=r"$\hat x$")
    ax.plot(*x_gt_series.T[:2], c=c_gt, alpha=0.9, label=r"$x_{gt}$")
    RMSEs_str = ", ".join(f"{v:.2f}" for v in (*RMSE_pred, *RMSE_upd))
    ax.set_title(f"Output from EKF with "
                 rf"$\sigma_a = {sigma_a}$, $\sigma_z= {sigma_z}$"
                 "\n"
                 f"RMSE(p_p, p_v, u_p, u_v) = ({RMSEs_str})")
    ax.legend()


def plot_NIS_NEES_data(sigma_a_low, sigma_a_high, sigma_a_list, sigma_z_low,
                       sigma_z_high, sigma_z_list, ANIS_data, CINIS,
                       ANEES_pred_data, ANEES_upd_data, CINEES):
    # %% interpolate ANEES/NIS
    ANIS_spline = scipy.interpolate.RectBivariateSpline(
        sigma_a_list, sigma_z_list, ANIS_data)
    ANEES_pred_spline = scipy.interpolate.RectBivariateSpline(
        sigma_a_list, sigma_z_list, ANEES_pred_data)
    ANEES_upd_spline = scipy.interpolate.RectBivariateSpline(
        sigma_a_list, sigma_z_list, ANEES_upd_data)

    n_eval = 100
    mesh_a, mesh_z = np.meshgrid(
        np.linspace(sigma_a_low, sigma_a_high, n_eval),
        np.linspace(sigma_z_low, sigma_z_high, n_eval))
    ANIS_eval = ANIS_spline(mesh_a.ravel(), mesh_z.ravel(),
                            grid=False).reshape(mesh_a.shape)
    ANEES_pred_eval = ANEES_pred_spline(mesh_a.ravel(),
                                        mesh_z.ravel(),
                                        grid=False).reshape(mesh_a.shape)
    ANEES_upd_eval = ANEES_upd_spline(mesh_a.ravel(),
                                      mesh_z.ravel(),
                                      grid=False).reshape(mesh_a.shape)

    # %% find confidence regions for NIS and plot
    # %% confidence plots

    # plot
    fig4 = plt.figure(4, clear=True)
    ax4 = fig4.add_subplot(1, 1, 1, projection="3d")
    z_max = 10
    ax4.plot_surface(mesh_a, mesh_z, np.clip(ANIS_eval, 0, z_max), alpha=0.9)
    ax4.contour(mesh_a, mesh_z, ANIS_eval, [*CINIS],
                offset=0)  # , extend3d=True, colors='yellow')
    ax4.set_xlabel(r"$\sigma_a$")
    ax4.set_ylabel(r"$\sigma_z$")
    ax4.set_zlabel("ANIS")
    ax4.set_zlim(0, z_max)
    ax4.view_init(30, 20)

    # %% find confidence regions for NEES and plot

    # plot
    fig5 = plt.figure(5, clear=True)
    z_max = 50
    ax5s = [
        fig5.add_subplot(1, 2, 1, projection="3d"),
        fig5.add_subplot(1, 2, 2, projection="3d"),
    ]
    ax5s[0].plot_surface(mesh_a,
                         mesh_z,
                         np.clip(ANEES_pred_eval, 0, z_max),
                         alpha=0.9)
    ax5s[0].contour(
        mesh_a,
        mesh_z,
        ANEES_pred_eval,
        [*CINEES],
        offset=0,
    )
    ax5s[0].set_xlabel(r"$\sigma_a$")
    ax5s[0].set_ylabel(r"$\sigma_z$")
    ax5s[0].set_zlabel("ANEES_pred")

    ax5s[0].set_zlim(0, z_max)
    ax5s[0].view_init(40, 30)

    ax5s[1].plot_surface(mesh_a,
                         mesh_z,
                         np.clip(ANEES_upd_eval, 0, z_max),
                         alpha=0.9)
    ax5s[1].contour(
        mesh_a,
        mesh_z,
        ANEES_upd_eval,
        [*CINEES],
        offset=0,
    )
    ax5s[1].set_xlabel(r"$\sigma_a$")
    ax5s[1].set_ylabel(r"$\sigma_z$")
    ax5s[1].set_zlabel("ANEES_upd")
    ax5s[1].set_zlim(0, z_max)
    ax5s[1].view_init(40, 30)

    # %% see the intersection of NIS and NEESes
    fig6, ax6 = plt.subplots(num=6, clear=True)
    cont_upd = ax6.contour(mesh_a,
                           mesh_z,
                           ANEES_upd_eval,
                           CINEES,
                           colors=["C0", "C1"])
    cont_pred = ax6.contour(mesh_a,
                            mesh_z,
                            ANEES_pred_eval,
                            CINEES,
                            colors=["C2", "C3"])
    cont_nis = ax6.contour(mesh_a,
                           mesh_z,
                           ANIS_eval,
                           CINIS,
                           colors=["C4", "C5"])

    for cs, l in zip([cont_upd, cont_pred, cont_nis],
                     ["NEESupd", "NEESpred", "NIS"]):
        for c, hl in zip(cs.collections, ["low", "high"]):
            c.set_label(l + "_" + hl)
    ax6.legend()
    ax6.set_xlabel(r"$\sigma_a$")
    ax6.set_ylabel(r"$\sigma_z$")
