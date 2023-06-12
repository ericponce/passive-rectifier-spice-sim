import matplotlib.pyplot as plt
import numpy as np

import sim


def plot_delta(ax, x, y, alpha=1.0, linefmt='C0-', basefmt='C3-'):
    if alpha is not None:
        minValue = alpha * np.average(np.abs(y))
        px = x[np.abs(y) > minValue]
        py = y[np.abs(y) > minValue]
        print(minValue)
    else:
        px = x
        py = y

    print(px.shape, py.shape)

    ax[0].stem(px, 20*np.log10(np.abs(py)), bottom=20*np.log10(minValue), linefmt=linefmt, basefmt=basefmt)
    ax[1].stem(px, np.angle(py) * 180 / np.pi, bottom=0, linefmt=linefmt, basefmt=basefmt)

# [Note] Could be optimized by having only a single access to the file
def plot_rectifier_ports(fig, ax, model, interpolate_dt, f_max, f_mix=60, incremental=None):

    # Get mixing signal from data
    t, mix = sim.extract_mixing_signal(model, interpolate_dt=interpolate_dt)
    t_sl = slice(np.argmin(np.abs(t - 1)), None)
    t = t[t_sl]
    mix = mix[t_sl]
    N = len(t)

    sp_mix = np.fft.fft(mix) / N
    f = np.fft.fftfreq(N, t[1] - t[0])

    sp_mix = np.fft.fftshift(sp_mix)
    f = np.fft.fftshift(f)

    # Get ac and dc voltages and currents
    t, vac, iac, vdc, idc = sim.get_voltage_and_current(model, interpolate_dt=interpolate_dt)

    t = t[t_sl]
    vac = vac[t_sl]
    iac = iac[t_sl]
    vdc = vdc[t_sl]
    idc = idc[t_sl]

    N = len(t)
    f = np.fft.fftshift(np.fft.fftfreq(N, t[1] - t[0]))
    sp_vdc = np.fft.fftshift(np.fft.fft(vdc) / N)
    sp_vac = np.fft.fftshift(np.fft.fft(vac) / N)
    sp_idc = np.fft.fftshift(np.fft.fft(idc) / N)
    sp_iac = np.fft.fftshift(np.fft.fft(iac) / N)

    def compute_thd(sp):
        idx = np.argmin(np.abs(f - f_mix))
        thd = (2 * sp)
        V1 = thd[idx]
        thd[idx] = 0
        return np.sqrt(np.sum(thd[N//2:]**2)) / V1
    
    if incremental is not None:
        print("incrementing")
        sp_mix -= incremental[0, :]
        sp_vdc -= incremental[1, :]
        sp_idc -= incremental[2, :]
        sp_vac -= incremental[3, :]
        sp_iac -= incremental[4, :]
    

    f_sl = slice(np.argmin(np.abs(f + 0)), np.argmin(np.abs(f - f_max)))
    magCutoff = 1e-3

    gs0 = ax[0, 0].get_gridspec()
    gs0.update(hspace=0)

    for axs in ax[:, 0]:
        axs.remove()
    axbig0 = fig.add_subplot(gs0[0:2, 0])
    axbig1 = fig.add_subplot(gs0[3:5, 0])
    axbig2 = fig.add_subplot(gs0[6:8, 0])

    ax[2, 0].set_visible(False)
    ax[5, 0].set_visible(False) 

    axbig0.plot(t, vac)
    axbig0.twinx().plot(t, iac)
    axbig0.set_title("THD(V)=%.2f%%, THD(I)=%.2f%%" % (100*compute_thd(sp_vac), 100*compute_thd(sp_iac)))

    axbig1.plot(t, mix)

    axbig2.plot(t, vdc)
    axbig2.twinx().plot(t, idc)

    ax[0][1].plot(f[f_sl], 20 * np.log10(np.abs(sp_vac[f_sl])))
    ax[0][1].plot(f[f_sl], 20 * np.log10(np.abs(sp_iac[f_sl])))
    ax[0][1].xaxis.set_tick_params(labelbottom=False)
    ax[0][1].set_xticks([])

    sp_vac[f_sl][np.abs(sp_vac[f_sl]) < magCutoff] = 0
    sp_iac[f_sl][np.abs(sp_iac[f_sl]) < magCutoff] = 0
    ax[1][1].plot(f[f_sl], 180 / np.pi * np.angle(sp_vac[f_sl]))
    ax[1][1].plot(f[f_sl], 180 / np.pi * np.angle(sp_iac[f_sl])) 

    ax[2, 1].set_visible(False)

    ax[3][1].plot(f[f_sl], 20 * np.log10(np.abs(sp_mix[f_sl])))
    ax[3][1].xaxis.set_tick_params(labelbottom=False)
    ax[3][1].set_xticks([])

    sp_mix[f_sl][np.abs(sp_mix[f_sl]) < magCutoff] = 0
    ax[4][1].plot(f[f_sl], 180 / np.pi * np.angle(sp_mix[f_sl]))

    ax[5, 1].set_visible(False)

    ax[6][1].plot(f[f_sl], 20 * np.log10(np.abs(sp_vdc[f_sl])))
    ax[6][1].plot(f[f_sl], 20 * np.log10(np.abs(sp_idc[f_sl])))
    ax[6][1].xaxis.set_tick_params(labelbottom=False)
    ax[6][1].set_xticks([])

    sp_vdc[f_sl][np.abs(sp_vdc[f_sl]) < magCutoff] = 0
    sp_idc[f_sl][np.abs(sp_idc[f_sl]) < magCutoff] = 0
    ax[7][1].plot(f[f_sl], 180 / np.pi * np.angle(sp_vdc[f_sl]))
    ax[7][1].plot(f[f_sl], 180 / np.pi * np.angle(sp_idc[f_sl]))

    return np.vstack((sp_mix, sp_vdc, sp_idc, sp_vac, sp_iac))

    




