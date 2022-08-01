import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/git/linf/figures/experiments.mplstyle')

dul = np.load("/home/s1511699/git/linf/data/du/dul_1snapshot.npy")

NS = np.logspace(3, 6.3, 4)

plt.figure()
for i in range(len(NS)):
    plt.hist(dul[:int(NS[i])], bins=100, density=True, histtype='step',
             label=f'N = {int(NS[i]):.0f}')
plt.yscale('log')
plt.legend()
plt.show()
plt.savefig("/home/s1511699/git/linf/data/du/dul_conv.png", dpi=288)
