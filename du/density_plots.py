import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

model_dir = "du/models/vd0407_lr5em4/"


def results(model_dir):
    tf.keras.backend.set_floatx("float64")
    plt.style.use("./misc/experiments.mplstyle")
    plt.ioff()

    figures_dir = model_dir + "figures/"
    # trained_model_file = model_dir + "checkpoint_epoch_10"
    trained_model_file = model_dir + "trained_nn"

    # Load neural network and Gaussian mixture layer.
    with open(model_dir + "gm.pickle", "rb") as f:
        gm = pickle.load(f)
    NN = tf.keras.models.load_model(
        trained_model_file, custom_objects={"nll_reg": gm.nll_reg}
    )
    gm.neural_net = NN

    scalers = []
    scalers_str = ["Xscaler", "Yscaler"]
    for i in range(len(scalers_str)):
        with open(model_dir + scalers_str[i] + ".pickle", "rb") as f:
            scalers.append(pickle.load(f))

    [Xscaler, Yscaler] = scalers

    # Plot p(du_l | r) for some value of r.
    rks = np.array([64.0, 16.0, 4.0]).reshape((-1, 1))
    rs = 2 * np.pi / rks
    # rs = np.array([0.05, 0.25, 1.]).reshape((-1, 1))
    gms_ = gm.get_gms_from_x(Xscaler.standardise(rs))
    cov = Yscaler.invert_standardisation_cov(gms_.covariance()).numpy()
    N = 10000

    c = ["b", "g", "r"]
    dul_n = np.linspace(-30.0, 30.0, N)

    plt.figure()
    for r in range(len(rs)):
        dul = dul_n * cov[r, 0, 0] ** 0.5
        pdul = (
            Yscaler.invert_standardisation_prob(
                gm.log_marg_density(Xscaler.standardise(rs[r, :]), dul, 0).numpy()
            )
            * cov[r, 0, 0] ** 0.5
        )
        plt.plot(dul, pdul, color=c[r], label=rf"$r = 2\pi/{rks[r, 0]:.0f}$")
    plt.plot(
        dul_n, -0.5 * np.log(2 * np.pi) - 0.5 * dul_n**2, "k--", label=r"Gaussian"
    )
    plt.xlabel(r"$\delta u_L / \sigma_{\delta u_L | r}$")
    plt.ylabel(r"$\ln p(\delta u_L  / \sigma_{\delta u_L | r}| r)$")
    plt.xlim(-7.5, 7.5)
    plt.ylim(-10.5, 1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(figures_dir + "pdul.png", dpi=576)

    plt.figure()
    for r in range(len(rs)):
        dut = dul_n * cov[r, 1, 1] ** 0.5
        pdut = (
            Yscaler.invert_standardisation_prob(
                gm.log_marg_density(Xscaler.standardise(rs[r, :]), dul, 0).numpy()
            )
            * cov[r, 1, 1] ** 0.5
        )
        plt.plot(dut, pdut, color=c[r], label=rf"$r = 2\pi/{rks[r, 0]:.0f}$")
    plt.plot(
        dul_n, -0.5 * np.log(2 * np.pi) - 0.5 * dul_n**2, "k--", label=r"Gaussian"
    )
    plt.xlabel(r"$\delta u_T / \sigma_{\delta u_T | r}$")
    plt.ylabel(r"$\ln p(\delta u_T  / \sigma_{\delta u_T | r}| r)$")
    plt.xlim(-7.5, 7.5)
    plt.ylim(-10.5, 1.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig(figures_dir + "pdut.png", dpi=576)
