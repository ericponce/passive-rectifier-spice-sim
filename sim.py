import os
import filecmp
from shutil import copyfile

# from PyLTSpice import RawWRead
from PyLTSpice.LTSpice_RawRead import RawRead

import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import quad

import argparse

# write a parameter file for the sim
# only write to the actual paramter file if it is different
# (so make doesn't needlessly compile)
def write_param_file(filename, params):
    tmp_filename = filename + '_'

    with open(tmp_filename,"w") as f:
        for key in params:
          f.write(".param {:s} {:E}\n".format(key, params[key]))
        f.write("\n")
        f.close()

    if os.path.isfile(filename) and filecmp.cmp(tmp_filename,filename):
        # os.remove(tmp_filename)
        pass
    else:
        copyfile(tmp_filename, filename)
        os.remove(tmp_filename)

def execute_spice(model):
    os.system('make build/' + model + '.raw')

def read_raw_file(simname):
    return RawRead("build/{:s}.raw".format(simname))

def get_voltage_and_current(model, interpolate_dt=None):
    ltr = read_raw_file(model)
    t = np.abs(ltr.get_trace('time').get_wave())
    vac = ltr.get_trace('V(ac_p)').get_wave() \
                - ltr.get_trace('V(ac_n)').get_wave()
    iac = ltr.get_trace('I(R4)').get_wave()

    vdc = ltr.get_trace('V(out)').get_wave()
    idc = ltr.get_trace('I(R2)').get_wave()

    if interpolate_dt is not None:
        N = int((t[-1] - t[0])/interpolate_dt)
        x = np.linspace(t[0], t[-1], N)
        xp = t

        vac = np.interp(x, xp, vac)
        iac = np.interp(x, xp, iac)

        vdc = np.interp(x, xp, vdc)
        idc = np.interp(x, xp, idc)

        t = x

    return t, vac, iac, vdc, idc

def get_minimum_timestep(model):
    ltr = read_raw_file(model)
    t = np.abs(ltr.get_trace('time').get_wave())
    return np.min(np.diff(t))

def extract_mixing_signal(model, interpolate_dt=None):
    ltr = read_raw_file(model)
    t = np.abs(ltr.get_trace('time').get_wave())
    iPos = ltr.get_trace('I(D5)').get_wave()
    iNeg = ltr.get_trace('I(D6)').get_wave()

    if interpolate_dt is None:
        mix = np.zeros_like(iPos)
        mix[iPos > 0] = 1
        mix[iNeg > 0] = -1

        return t, mix
    else: # interpolate
        # Digitize Diode Current
        iPosD = np.zeros_like(iPos)
        iNegD = np.zeros_like(iNeg)
        iPosD[iPos > 0] = 1
        iNegD[iNeg > 0] = 1

        # Find transistion points using the derivative of the square wave
        diffPosD = np.diff(iPosD)
        diffNegD = np.diff(iNegD)

        pos_open_idxs = np.where(diffPosD > 0)[0] + 1
        pos_close_idxs = np.where(diffPosD < 0)[0]

        neg_open_idxs = np.where(diffNegD > 0)[0] + 1
        neg_close_idxs = np.where(diffNegD < 0)[0]

        N = int((t[-1] - t[0])/interpolate_dt)
        t_interp = np.linspace(t[0], t[-1], N)
        mix_interp = np.ones_like(t_interp) * (1 if iPos[0] > 0 else (-1 if iNeg[1] > 0 else 0))

        for i in pos_open_idxs:
            t_idx = np.argmin(np.abs(t[i] - t_interp))
            mix_interp[t_idx:] += 1

        for i in pos_close_idxs:
            t_idx = np.argmin(np.abs(t[i] - t_interp))
            mix_interp[t_idx:] -= 1

        for i in neg_open_idxs:
            t_idx = np.argmin(np.abs(t[i] - t_interp))
            mix_interp[t_idx:] -= 1

        for i in neg_close_idxs:
            t_idx = np.argmin(np.abs(t[i] - t_interp))
            mix_interp[t_idx:] += 1

        return t_interp, mix_interp

# def compute_fourier_coefficient(t, x, omega):
#     # multiply by complex exponential
#     integrand = x * np.exp(-1j * omega * t)

#     # integrate over period(s)
#     return (2 / t[-1]) * np.trapz(integrand, t)

# def get_fourier_decomposition(t, x, omega):
#     fc = compute_fourier_coefficient(t, x, omega)

#     return np.abs(fc) * np.exp(-1j * omega * t + np.angle(fc))

# very very simple fourier transform
# sp = fourier_transform(time, data, frequencies (rad/s))
def fourier_transform(t, x, w):
    P = t[-1] - t[0] # integration period
    omega = np.expand_dims(w, 1)
    t = np.expand_dims(t, 0)
    x = np.expand_dims(x, 1)

    return (1 / P) * np.exp(-1j * omega @ t) @ x

if __name__ == "__main__":
    model = 'base_RLC'

    gen_params = {
        'fl':60,
        'fp':21,
        'Ldc':50e-3,
        'Cdc':1000e-6,
        'Rdc':10,
        'transtop':2,
        'transtart':1,
        'timestep':1e-6
    }

    write_param_file(model+'.gen', gen_params)

    execute_spice(model)

    ltr = read_raw_file(model)

    # the simulation will only have one step, 0
    step = 0
    t = ltr.get_trace('time').get_wave()
    t = np.abs(t) # fix sign error 
    vac = ltr.get_trace('V(ac_p)').get_wave() \
            - ltr.get_trace('V(ac_n)').get_wave()
    iac = ltr.get_trace('I(R4)').get_wave()

    dt_avg = np.average(np.gradient(t)[10:-10])

    # get grid fundamental to subtract from vac
    omega_grid = 2 * np.pi * gen_params['fl']
    omega_pert = 2 * np.pi * gen_params['fp']
    exp_grid = np.exp(-1j * omega_grid * t)
    exp_pert = np.exp(-1j * omega_pert * t)

    fc = np.trapz(vac * exp_grid, t) * 2 / t[-1]; # compute fourier coefficient
    vac_grid = np.abs(fc) * np.exp(-1j * omega_grid * t + np.angle(fc))
    vac_grid = np.real(vac_grid)

    fc = np.trapz(iac * exp_grid, t) * 2 / t[-1]; # compute fourier coefficient
    iac_grid = np.abs(fc) * np.exp(-1j * omega_grid * t + np.angle(fc))
    iac_grid = np.real(iac_grid)

    fc = np.trapz(vac * exp_pert, t) * 2 / t[-1];
    vac_pert = np.abs(fc) * np.exp(-1j * omega_pert * t + np.angle(fc))
    vac_pert = np.real(vac_pert)

    fc = np.trapz(iac * exp_pert, t) * 2 / t[-1];
    iac_pert = np.abs(fc) * np.exp(-1j * omega_pert * t + np.angle(fc))
    iac_pert = np.real(iac_pert)

    plt.figure()
    ax = plt.subplot(3, 2, 1)
    ax.set_title("$V_{ac}$")
    ax.plot(t, vac)
    ax.plot(t, vac_grid + vac_pert)
    ax.twinx().plot(t, vac - vac_grid - vac_pert, color='black', ls='--', alpha=0.8)

    ax = plt.subplot(3, 2, 2)
    ax.set_title("$I_{ac}$")
    ax.plot(t, iac)
    ax.plot(t, iac_grid + iac_pert)
    ax.twinx().plot(t, iac - iac_grid - iac_pert, color='black', ls='--', alpha=0.8)

    ax = plt.subplot(3, 2, 3)
    ax.set_title("$V_{ac,grid}$")
    ax.plot(t, vac - vac_pert)
    ax.plot(t, vac_grid)
    ax.twinx().plot(t, vac_grid - vac + vac_pert, color='black', ls='--', alpha=0.8)

    ax = plt.subplot(3, 2, 4)
    ax.set_title("$I_{ac,grid}$")
    ax.plot(t, iac - iac_pert)
    ax.plot(t, iac_grid)
    ax.twinx().plot(t, iac_grid - iac + iac_pert, color='black', ls='--', alpha=0.8)

    ax = plt.subplot(3, 2, 5)
    ax.set_title("$V_{ac,pert}$")
    ax.plot(t, vac - vac_grid)
    ax.plot(t, vac_pert)
    ax.twinx().plot(t, vac - vac_grid - vac_pert, color='black', ls='--', alpha=0.8)

    ax = plt.subplot(3, 2, 6)
    ax.set_title("$I_{ac,pert}$")
    ax.plot(t, iac - iac_grid)
    ax.plot(t, iac_pert)
    # ax.twinx().plot(t, vac - vac_grid - vac_pert, color='black', ls='--', alpha=0.8)

    # ax = plt.subplot(3, 2, 6)
    # ax.set_title("FFT")

    # freqs = np.fft.fftfreq(len(vac), dt_avg)
    # sp = np.fft.fft(vac - vac_grid - vac_pert)
    # freqs = np.fft.fftshift(freqs)
    # sp = np.fft.fftshift(np.abs(sp))

    # plt.semilogy(freqs, sp)

    plt.tight_layout()
    plt.show()

    exit()

    plt.figure()
    plt.plot(t, vac)
    plt.plot(t, vac_grid)
    plt.axhline(np.abs(vl))
    plt.show()


    plt.figure()
    plt.plot(t, vac - vac_grid)
    plt.axvline(1/gen_params['fp'])
    # plt.plot(t, vac_grid / np.abs(vl))
    plt.plot(t, vac_pert)
    plt.show()