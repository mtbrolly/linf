import numpy as np
import matplotlib.pyplot as plt
# import time
plt.style.use('~/github/linf/figures/experiments.mplstyle')


# Define some constants and load data.
Nx = 1024
Ny = 1024
x = np.arange(0.5, Nx, 1.) / Nx * (2 * np.pi)
y = np.arange(0.5, Ny, 1.) / Ny * (2 * np.pi)
xx, yy = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

u_all = np.load(
    "/home/s1511699/github/linf/data/velocity_differences/u_all.npy")[:2, ...]
v_all = np.load(
    "/home/s1511699/github/linf/data/velocity_differences/v_all.npy")[:2, ...]

hSuuu = np.zeros((Ny, Nx), dtype=np.complex128)
hSuuv = np.zeros((Ny, Nx), dtype=np.complex128)
hSuvv = np.zeros((Ny, Nx), dtype=np.complex128)
hSvvv = np.zeros((Ny, Nx), dtype=np.complex128)

# Quantities used for angle average.
xc = x - x.max() / 2.
yc = y - y.max() / 2.

dr = xc[1] - xc[0]
rmax = xc.max()

RR = np.sqrt((xx - x.max() / .2) ** 2 + (yy - y.max() / 2.))


u_all = np.load(
    "/home/s1511699/github/linf/data/velocity_differences/u_all.npy")[:2, ...]
v_all = np.load(
    "/home/s1511699/github/linf/data/velocity_differences/v_all.npy")[:2, ...]

for t in range(len(u_all)):
    u = u_all[t, ...]
    v = v_all[t, ...]

    hu = np.fft.fft2(u)
    hv = np.fft.fft2(v)

    huu = np.fft.fft2(u ** 2)
    hvv = np.fft.fft2(v ** 2)
    huv = np.fft.fft2(u * v)

    hSuuu += -3. * huu * np.conj(hu) + 3. * np.conj(huu) * hu
    hSuuv += (-2. * huv * np.conj(hu) + 2. * np.conj(huv) * hu
              - huu * np.conj(hv) + np.conj(huu) * hv)
    hSuvv += (-2. * huv * np.conj(hv) + 2. * np.conj(huv) * hv
              - hvv * np.conj(hu) + np.conj(hvv) * hu)
    hSvvv += -3. * hvv * np.conj(hv) + 3. * np.conj(hvv) * hv

hSuuu /= len(u_all)
hSuuv /= len(u_all)
hSuvv /= len(u_all)
hSvvv /= len(u_all)

Suuu = np.real(np.fft.ifft2(hSuuu) / (Nx * Ny))
Suuv = np.real(np.fft.ifft2(hSuuu) / (Nx * Ny))
Suvv = np.real(np.fft.ifft2(hSuuu) / (Nx * Ny))
Svvv = np.real(np.fft.ifft2(hSuuu) / (Nx * Ny))


def Centre_matrix(A, Nx, Ny):
    B = np.tile(A, (2, 2))
    return B[int(Ny / 2): int(Ny * 3 / 2), int(Nx / 2): int(Nx * 3 / 2)]


Suuu_cen = Centre_matrix(Suuu, Nx, Ny)
Suuv_cen = Centre_matrix(Suuv, Nx, Ny)
Suvv_cen = Centre_matrix(Suvv, Nx, Ny)
Svvv_cen = Centre_matrix(Svvv, Nx, Ny)

r = x

plt.figure()
plt.loglog(r[: int(Nx / 2)], Suuu_cen[int(Nx / 2), int(Nx / 2):])
plt.loglog(r[: int(Nx / 2)], Suuv_cen[int(Nx / 2), int(Nx / 2):])
plt.loglog(r[: int(Nx / 2)], Suvv_cen[int(Nx / 2), int(Nx / 2):])
plt.loglog(r[: int(Nx / 2)], Svvv_cen[int(Nx / 2), int(Nx / 2):])
plt.show()

# mup2u = np.fft.irfft2(np.fft.rfft2(u ** 2) * np.conj((np.fft.rfft2(u))))
# mupu2 = np.fft.irfft2(np.fft.rfft2(u) * np.conj((np.fft.rfft2(u ** 2))))
# mdu3 = -3. * mup2u + 3. * mupu2
# r_xy = np.sqrt(xx ** 2 + yy ** 2)
# dr = np.sqrt(dx ** 2 + dy ** 2)
# r = np.arange(dr / 2., np.pi + dr, dr)

# mdu3r = np.zeros(r.size)

# for i in range(r.size):
#     fr = (r_xy >= r[i] - dr / 2) & (r_xy <= r[i] + dr / 2)
#     mdu3r[i] = mdu3[fr].mean() * 2 * np.pi * r[i] * dr / 1024

# plt.figure()
# plt.loglog(r, np.abs(mdu3r), 'k')
# plt.show()

# =============================================================================
# for i in range(kr.size):
#     fkr = (model.wv >= kr[i] - dkr / 2) & (model.wv <= kr[i] + dkr / 2)
#     dtk = pi / (fkr.sum() - 1)
#     spec_iso[i] = spec2D[fkr].sum() * kr[i] * dtk
# =============================================================================

# =============================================================================
# r = x[1: int(len(u) / 2)] - x[0]
# SF3l = np.zeros(np.shape(r))
# SF3t = np.zeros(np.shape(r))
#
# for i in range(len(r)):
#     print(i)
#     t0 = time.time()
#     up = np.roll(u, -(i + 1), axis=0)
#     up2 = up ** 2
#     up2h = np.fft.rfft2(up2)
#     uhc = np.conj(np.fft.rfft2(u))
#     mup2u = np.mean(np.fft.irfft2(up2h * uhc))
#     u2hc = np.conj(np.fft.rfft2(u ** 2))
#     uph = np.fft.rfft2(up)
#     mupu2 = np.mean(np.fft.irfft2(uph * u2hc))
#
#     SF3l[i] = -3. * mup2u + 3 * mupu2
#     T = time.time() - t0
#     # print(f'Time taken: {T:.1f} seconds.')
#
# plt.figure()
# plt.plot(r, SF3l, 'k')
# plt.show()
# =============================================================================
