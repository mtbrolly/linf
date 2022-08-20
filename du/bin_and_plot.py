import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
plt.style.use('./misc/experiments.mplstyle')
cp = sns.color_palette("husl", 8)

DATA_DIR = "data/GLAD/"

r = np.load(DATA_DIR + "r.npy")
du = np.load(DATA_DIR + "du.npy")

# DATA_DIR = "data/GLAD/DB_data/"  # !!!

# r2 = np.load(DATA_DIR + "r.npy")
# du2 = np.load(DATA_DIR + "du.npy")

# re = np.logspace(np.log10(r.min()), np.log10(r.max()), 20)
# re = np.logspace(np.log10(10), np.log10(1e6), 17)
base = 1.5
re = np.logspace(np.log(10.) / np.log(base), np.log(1e6) / np.log(base), 20,
                 base=base)
rc = 0.5 * (re[1:] + re[:-1])  # !!!
ind = np.digitize(r, re)[:, 0]


# --- HISTOGRAM OF r FIGURE ---

hist_r, _ = np.histogram(r, bins=re)

plt.figure()
plt.plot(rc, hist_r, 'k')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$r$')
plt.ylabel(r'Number of pairs')
plt.show()


# --- S2 FIGURE ---

S2 = np.zeros(rc.shape + (2, ))
S2a = np.zeros(rc.shape + (2, ))
for i in range(len(rc)):
    S2[i, :] = np.var(du[ind == i, :], axis=0)
    S2a[i, :] = np.mean(du[ind == i, :] ** 2, axis=0)


SF2l = S2[:, 0]
SF2t = S2[:, 1]

plt.figure()
plt.loglog(rc, SF2l + SF2t, 'k-*', label=r'$S_2(r)$')
plt.plot(rc, SF2l, '-*', color=cp[5], label=r'$S_2^{(L)}(r)$')
plt.plot(rc, SF2t, '-*', color=cp[3], label=r'$S_2^{(T)}(r)$')
ylims = plt.ylim()
plt.ylim(*ylims)
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'Second-order structure functions')
plt.legend()
plt.tight_layout()
plt.show()


# --- S3 FIGURE ---

S3 = np.zeros(rc.shape + (2, ))
for i in range(len(rc)):
    S3[i, 0] = np.mean(du[ind == i, 0] ** 3)
    S3[i, 1] = np.mean(du[ind == i, 0] * du[ind == i, 1] ** 2)


S3l = S3[:, 0]
S3t = S3[:, 1]

plt.figure()
plt.loglog(rc, S3l + S3t, 'k-*', label=r'$S_3(r)$')
plt.loglog(rc, -(S3l + S3t), 'k--*', label=r'$-S_3(r)$')
# plt.plot(rc, S3l, '-*', color=cp[5], label=r'$S_3^{(L)}(r)$')
# plt.plot(rc, -S3l, '--*', color=cp[5], label=r'$-S_3^{(L)}(r)$')
# plt.plot(rc, S3t, '-*', color=cp[3], label=r'$S_3^{(T)}(r)$')
# plt.plot(rc, -S3t, '--*', color=cp[3], label=r'$-S_3^{(T)}(r)$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'Third-order structure functions')
plt.legend()
plt.xlim(10., 1000000.)
plt.tight_layout()
plt.show()


# --- DENSITIES FIGURE ---

rs = np.array([10 ** 3, 10 ** 4, 10 ** 5])
rs_pm = np.array([[rs - rs / 2., rs + rs / 2.] for rs in rs])

fig, ax = plt.subplots(3, 2, figsize=(10, 10))

for i in range(len(rs)):
    # Data histogram
    r_ind = ((r > rs_pm[i, 0]) & (r < rs_pm[i, 1]))[:, 0]
    du_r = du[r_ind, :]
    hist, bin_e = np.histogram(du_r[:, 0], bins=50, density=True)
    ax[i, 0].plot((bin_e[1:] + bin_e[:-1]) / 2, np.log(hist), 'k')
    hist, bin_e = np.histogram(du_r[:, 1], bins=50, density=True)
    ax[i, 1].plot((bin_e[1:] + bin_e[:-1]) / 2, np.log(hist), 'k')
    ylims0 = ax[i, 0].set_ylim()
    ylims1 = ax[i, 1].set_ylim()

    # Gaussian pdf with same variance
    dul = np.linspace(*ax[i, 0].set_xlim(), 100)
    dut = np.linspace(*ax[i, 1].set_xlim(), 100)
    lp_nl = norm.logpdf(dul, loc=np.mean(du_r[:, 0]), scale=np.std(du_r[:, 0]))
    lp_nt = norm.logpdf(dut, loc=np.mean(du_r[:, 1]), scale=np.std(du_r[:, 1]))
    ax[i, 0].plot(dul, lp_nl, '--', color='0.5')
    ax[i, 1].plot(dut, lp_nt, '--', color='0.5')

    ax[i, 0].set_title(rf"$r = {rs[i]:.0f}$")
    ax[i, 1].set_title(rf"$r = {rs[i]:.0f}$")
    ax[i, 0].set_xlabel(r"$\delta u_{L}$")
    ax[i, 1].set_xlabel(r"$\delta u_{T}$")
    ax[i, 0].set_ylabel(r"$\log p(\delta u_{L}|r)$")
    ax[i, 1].set_ylabel(r"$\log p(\delta u_{T}|r)$")
    ax[i, 0].grid(True)
    ax[i, 1].grid(True)
    ax[i, 0].set_xlim(-np.max(np.abs(ax[i, 0].set_xlim())),
                      np.max(np.abs(ax[i, 0].set_xlim())))
    ax[i, 1].set_xlim(-np.max(np.abs(ax[i, 1].set_xlim())),
                      np.max(np.abs(ax[i, 1].set_xlim())))
    ax[i, 0].set_ylim(*ylims0)
    ax[i, 1].set_ylim(*ylims1)

fig.tight_layout()
fig.show()
