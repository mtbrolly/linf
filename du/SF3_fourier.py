import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/git/linf/figures/experiments.mplstyle')


# Define some constants and load data.
Nx = 1024
Ny = 1024
x = np.arange(0.5, Nx, 1.) / Nx * (2 * np.pi)
y = np.arange(0.5, Ny, 1.) / Ny * (2 * np.pi)
xx, yy = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

u_all = np.load(
    "/home/s1511699/git/linf/data/du/u_some.npy")
v_all = np.load(
    "/home/s1511699/git/linf/data/du/v_some.npy")

hSuuu = np.zeros((Ny, Nx), dtype=np.complex128)
hSuuv = np.zeros((Ny, Nx), dtype=np.complex128)
hSuvv = np.zeros((Ny, Nx), dtype=np.complex128)
hSvvv = np.zeros((Ny, Nx), dtype=np.complex128)

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
Suuv = np.real(np.fft.ifft2(hSuuv) / (Nx * Ny))
Suvv = np.real(np.fft.ifft2(hSuvv) / (Nx * Ny))
Svvv = np.real(np.fft.ifft2(hSvvv) / (Nx * Ny))


def Centre_matrix(A, Nx, Ny):
    B = np.tile(A, (2, 2))
    return B[int(Ny / 2): int(Ny * 3 / 2), int(Nx / 2): int(Nx * 3 / 2)]


Suuu_cen = Centre_matrix(Suuu, Nx, Ny)
Suuv_cen = Centre_matrix(Suuv, Nx, Ny)
Suvv_cen = Centre_matrix(Suvv, Nx, Ny)
Svvv_cen = Centre_matrix(Svvv, Nx, Ny)

r = x


plt.figure()
plt.loglog(r[: int(Nx / 2)], -Suuu_cen[int(Nx / 2), int(Nx / 2):],
           label=r'$S_{LLL}$')
plt.loglog(r[: int(Nx / 2)], -Suuv_cen[int(Nx / 2), int(Nx / 2):],
           label=r'$S_{LLT}$')
plt.loglog(r[: int(Nx / 2)], -Suvv_cen[int(Nx / 2), int(Nx / 2):],
           label=r'$S_{LTT}$')
plt.loglog(r[: int(Nx / 2)], -Svvv_cen[int(Nx / 2), int(Nx / 2):],
           label=r'$S_{TTT}$')
plt.xlabel(r'$r$')
plt.ylabel(r'$V(x)$')
plt.title(r'Third-order structure functions')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.loglog(r[: int(Nx / 2)],
           (-(Suuu_cen[int(Nx / 2), int(Nx / 2):]
              + Suvv_cen[int(Nx / 2), int(Nx / 2):])),
           label=r'$V(r)$')
plt.loglog(r[: int(Nx / 2)],
           (-(Suuu_cen[int(Nx / 2), int(Nx / 2):])),
           label=r'$S_L(r)$')
plt.loglog(r[: int(Nx / 2)],
           (-(Suvv_cen[int(Nx / 2), int(Nx / 2):])),
           label=r'$S_T(r)$')
plt.loglog(r[: int(Nx / 2)], r[: int(Nx / 2)] ** 3., '--',
           label=r'$r^3$')
plt.loglog(r[: int(Nx / 2)], r[: int(Nx / 2)] ** 1., '--',
           label=r'$r^1$')
plt.xlabel(r'$r$')
plt.legend()
plt.title(r'Third-order structure functions')
plt.tight_layout()
plt.show()

plt.figure()
plt.loglog(r[: int(Nx / 2)],
           ((Suuu_cen[int(Nx / 2), int(Nx / 2):]
             + Suvv_cen[int(Nx / 2), int(Nx / 2):])),
           label=r'$V(r)$')
plt.loglog(r[: int(Nx / 2)],
           ((Suuu_cen[int(Nx / 2), int(Nx / 2):])),
           label=r'$S_L(r)$')
plt.loglog(r[: int(Nx / 2)],
           ((Suvv_cen[int(Nx / 2), int(Nx / 2):])),
           label=r'$S_T(r)$')
plt.xlabel(r'$r$')
plt.legend()
plt.title(r'Third-order structure functions')
plt.tight_layout()
plt.show()

plt.figure()
plt.loglog(r[: int(Nx / 2)],
           (-(Svvv_cen[int(Nx / 2), int(Nx / 2):]
              + Suuv_cen[int(Nx / 2), int(Nx / 2):])),
           label=r'$S_{TTT} + S_{LLT}$')
plt.xlabel(r'$r$')
plt.ylabel(r'$V(x)$')
plt.title(r'Third-order structure functions')
plt.legend()
plt.tight_layout()
plt.show()


plt.figure()
plt.plot(r[: int(Nx / 2)], Suuu_cen[int(Nx / 2), int(Nx / 2):],
         label=r'$S_{LLL}$')
plt.plot(r[: int(Nx / 2)], Suuv_cen[int(Nx / 2), int(Nx / 2):],
         label=r'$S_{LLT}$')
plt.plot(r[: int(Nx / 2)], Suvv_cen[int(Nx / 2), int(Nx / 2):],
         label=r'$S_{LTT}$')
plt.plot(r[: int(Nx / 2)], Svvv_cen[int(Nx / 2), int(Nx / 2):],
         label=r'$S_{TTT}$')
plt.xlabel(r'$r$')
plt.ylabel(r'$V(x)$')
plt.title(r'Third-order structure functions')
plt.legend()
plt.tight_layout()
plt.show()
