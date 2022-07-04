import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/git/linf/figures/experiments.mplstyle')


r = np.load("./r.npy")
SF2l = np.load("./SF2l.npy")
SF2t = np.load("./SF2t.npy")
SF3l = np.load("./SF3l.npy")
SF3t = np.load("./SF3t.npy")

# Check correspondence between SF2l and SF2t.
SF2t_rc = np.gradient(r * SF2l, r[1] - r[0])

plt.figure()
plt.plot(r, SF2t, 'k')
plt.plot(r, SF2t_rc, 'b--')
plt.show()


# Check correspondence between SF3l and SF3t.
dSF3l_dr = (SF3l[1:] - SF3l[:-1]) / (r[1] - r[0])
dSF3l_dr_g = np.gradient(SF3l, r[1] - r[0])

plt.figure()
plt.plot(r, SF3t, 'k')
# plt.plot(r[:-1], r[:-1] / 3 * dSF3l_dr, 'r--')
plt.plot(r, r / 3 * dSF3l_dr_g, 'b--')
plt.show()

plt.figure()
plt.loglog(r, np.abs(SF3t), 'k')
# plt.plot(r[:-1], r[:-1] / 3 * dSF3l_dr, 'r--')
plt.loglog(r, np.abs(r / 3 * dSF3l_dr_g), 'b--')
plt.show()


# Plot third-order structure functions
plt.figure()
plt.semilogx(r, SF3l / r + SF3t / r, 'k', label=r'$S_L(r) + S_T(r)$')
plt.semilogx(r, SF3l / r, 'b', label=r'$S_L(r)$')
plt.semilogx(r, SF3t / r, 'g', label=r'$S_T(r)$')
plt.xlabel(r'$r$')
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.loglog(r, np.abs(SF3l), 'k*-')
plt.plot(np.array([0.01, 0.05]), 100 * np.array([0.01, 0.05]) ** 3, 'r--',
         label=r'$r^{3}$')
plt.plot(np.array([0.12, 0.3]), .1 * np.array([0.12, 0.3]) ** 1, 'b--',
         label=r'$r^{1}$')
plt.vlines(2 * np.pi / 64, 1e-3, 0.1, 'darkgrey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 1e-3, 0.1, 'grey', '-.',
           label=r'$l_{lsf}$')
plt.ylabel(r'$|S_L(r)|$')
plt.xlabel(r'$r$')
plt.legend()
plt.grid()
plt.show()


# Plot second-order structure functions
plt.figure()
plt.loglog(r, SF2l + SF2t, 'k-', label=r'$S_2^{(L)}+S_2^{(T)}(r)$')
plt.loglog(r, SF2l, 'b-', label=r'$S_2^{(L)}(r)$')
plt.loglog(r, SF2t, 'g-', label=r'$S_2^{(T)}(r)$')

plt.vlines(2 * np.pi / 64, 4e-2, 0.5, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 4e-2, 0.5, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 4e-2, 0.5, 'grey', ':',
           label=r'$l_{d}$')
plt.plot([0.01, 0.08], 124 * np.array([0.01, 0.08]) ** 2., 'r--',
         label=r'$r^{2}$')
plt.plot([0.11, 0.25], np.array([0.11, 0.25]) ** (1.), 'm--',
         label=r'$r^{1}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.title(r'Longitudinal and transverse second-order structure function')
plt.legend()
plt.tight_layout()
plt.show()
