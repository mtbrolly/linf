import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/git/linf/figures/experiments.mplstyle')

results_dir = (
    "/home/s1511699/git/linf/data/du/real_binned_estimates/new_data/")

r = np.load(results_dir + "r.npy")
SF2l = np.load(results_dir + "SF2l.npy")
SF2t = np.load(results_dir + "SF2t.npy")
SF3l = np.load(results_dir + "SF3l.npy")
SF3t = np.load(results_dir + "SF3t.npy")


# Check correspondence between SF2l and SF2t.
SF2t_rc = np.gradient(r * SF2l, r[1] - r[0])

plt.figure()
plt.loglog(r, SF2t, 'k')
plt.plot(r, SF2t_rc, 'b--')
plt.show()


# Check correspondence between SF3l and SF3t.
dSF3l_dr = np.gradient(SF3l, r[1] - r[0])

plt.figure()
plt.plot(r, SF3t, 'k')
# plt.plot(r[:-1], r[:-1] / 3 * dSF3l_dr, 'r--')
plt.plot(r, r / 3 * dSF3l_dr, 'b--')
plt.show()

plt.figure()
plt.loglog(r, np.abs(SF3t), 'k')
# plt.plot(r[:-1], r[:-1] / 3 * dSF3l_dr, 'r--')
plt.loglog(r, np.abs(r / 3 * dSF3l_dr), 'b--')
plt.show()


# Plot third-order structure functions
plt.figure()
plt.semilogx(r, SF3l + SF3t, 'k', label=r'$V(r)$')
# plt.semilogx(r, SF3l, 'b', label=r'$S_L(r)$')
# plt.semilogx(r, SF3t, 'g', label=r'$S_T(r)$')
plt.xlabel(r'$r$')
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

# plt.savefig("/home/s1511699/git/linf/du"
#             + "/figures/real_binned_estimates/new_data/S3_all.png", dpi=576)


plt.figure()
plt.loglog(r, SF3l + SF3t, 'k-', label=r'$V(r)$')
plt.loglog(r, -(SF3l + SF3t), 'k--', label=r'$-V(r)$')
# plt.loglog(r, SF3l, 'b-', label=r'$S_{(L)}(r)$')
# plt.loglog(r, -SF3l, 'b--', label=r'$-S_{(L)}(r)$')
# plt.loglog(r, SF3t, 'g-', label=r'$S_{(T)}(r)$')
# plt.loglog(r, -SF3t, 'g--', label=r'$-S_{(T)}(r)$')
plt.plot(np.array([0.01, 0.1]), 1. * np.array([0.01, 0.1]) ** 3, 'g--',
         label=r'$r^{3}$')
plt.plot(np.array([0.12, 0.3]), .01 * np.array([0.12, 0.3]) ** 1, 'b--',
         label=r'$r^{1}$')
plt.vlines(2 * np.pi / 64, 1e-5, 1e-2, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 1e-5, 1e-2, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 1e-5, 1e-2, 'grey', ':',
           label=r'$l_{d}$')
plt.ylim(1e-7, None)
plt.xlabel(r'$r$')
plt.legend()
plt.grid()
plt.show()

# plt.savefig("/home/s1511699/git/linf/du"
#             + "/figures/real_binned_estimates/new_data/S3_all.png", dpi=576)


# Plot second-order structure functions
plt.figure()
plt.loglog(r, SF2l + SF2t, 'k-', label=r'$S_2(r)$')
plt.plot(r, SF2l, 'k--', label=r'$S_2^{(L)}(r)$')
plt.plot(r, SF2t, 'k-.', label=r'$S_2^{(T)}(r)$')
plt.vlines(2 * np.pi / 64, 2e-3, 0.5, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 2e-3, 0.5, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 2e-3, 0.5, 'grey', ':',
           label=r'$l_{d}$')
plt.plot([0.006, 0.04], 50 * np.array([0.006, 0.04]) ** 2., 'g--',
         label=r'$r^{2}$')
plt.plot([0.06, 0.25], 2. * np.array([0.06, 0.25]) ** (1.), 'b--',
         label=r'$r^{1}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.title(r'Second-order structure functions')
plt.legend()
plt.tight_layout()
plt.show()

# plt.savefig("/home/s1511699/git/linf/du"
#             + "/figures/real_binned_estimates/new_data/S2_all.png", dpi=576)
