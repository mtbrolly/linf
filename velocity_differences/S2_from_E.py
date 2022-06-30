"""
Calculate S_2(r) from E(k).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv as J
from scipy.integrate import simpson
plt.style.use('~/github/linf/figures/experiments.mplstyle')

k = np.load("/home/s1511699/github/linf/data/sf_ml_snapshots/kr.npy")
E = np.load("/home/s1511699/github/linf/data/sf_ml_snapshots/iso_E_spec.npy")

S2 = np.zeros_like(k)
r = 2 * np.pi / k[::-1]


def integrand(r):
    return E * (1 - J(0, k * r))


for i in range(r.size):
    S2[i] = 4 * simpson(integrand(r[i]), k)

np.save("/home/s1511699/github/linf/data/sf_ml_snapshots/S2_from_E.npy", S2)

rm = np.load(
    "/home/s1511699/github/linf/data/sf_ml_snapshots/rm.npy")
S2_from_increments = np.load(
    "/home/s1511699/github/linf/data/sf_ml_snapshots/S2.npy")


plt.figure()
plt.loglog(r, S2, 'k-', label='From energy spectrum')
plt.loglog(rm, S2_from_increments, 'r--', label='From increments')
plt.vlines(2 * np.pi / 64, 0.1, 0.5, 'grey', '-.',
           label=r'$l_f$')
plt.vlines(2 * np.pi / 6, 0.1, 0.5, 'grey', '--',
           label=r'$l_{lsf}$')
plt.vlines(2 * np.pi / 350, 0.1, 0.5, 'grey', ':',
           label=r'$l_{d}$')
plt.plot([0.04, 0.08], 40 * np.array([0.04, 0.08]) ** 2., 'g--',
         label=r'$r^{2}$')
plt.plot([0.11, 0.25], 1.5 * np.array([0.11, 0.25]) ** (2. / 3.), 'b--',
         label=r'$r^{2/3}$')
plt.grid(True)
plt.xlabel(r'$r$')
plt.ylabel(r'$S_2(r)$')
plt.title(r'Isotropic second-order structure function')
plt.legend()
plt.tight_layout()
plt.show()

# plt.savefig("./figures/S2_from_E.png", format='png', dpi=576)
# plt.savefig("./figures/S2_from_E_and_increments.png", format='png', dpi=576)
