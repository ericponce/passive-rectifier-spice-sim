
import sim

import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import quad
from scipy import signal

import argparse

def compute_incremental_impedance(gen_params, freqs, timesteps):
    sp = np.zeros(len(freqs), dtype=np.cdouble)

    for i, f in enumerate(freqs):
        gen_params['fp'] = f
        gen_params['timestep'] = timesteps[i]
        sim.write_param_file('gen.txt', gen_params)

        sim.execute_spice()

        ltr = sim.read_raw_file('base')

        t = ltr.get_trace('time').get_wave()
        t = np.abs(t) # fix sign error 
        vac = ltr.get_trace('V(ac_p)').get_wave() \
                - ltr.get_trace('V(ac_n)').get_wave()
        iac = ltr.get_trace('I(R4)').get_wave()

        # get grid fundamental to subtract from vac
        omega_pert = 2 * np.pi * f
        exp_pert = np.exp(-1j * omega_pert * t)

        vac_pert = np.trapz(vac * exp_pert, t) * 2 / t[-1];

        iac_pert = np.trapz(iac * exp_pert, t) * 2 / t[-1];

        print(vac_pert, iac_pert)

        sp[i] = vac_pert / iac_pert

        print("f=%f, %d of %d complete." % (f, i, len(freqs)))

    return sp

if __name__ == "__main__":
    gen_params = {
        'fl': 60,
        'fp':21,
        'Ldc':50e-3,
        'Cdc':1000e-6,
        'Rdc':10,
        'transtop':2,
        'transtart':1,
        'timestep':10e-6
    }

    f_mix = gen_params['fl']

    Ldc = gen_params['Ldc']
    Cdc = gen_params['Cdc']
    Rdc = gen_params['Rdc']

    # Frequency Domain Estimate
    Ydc_tf = signal.lti([0, Rdc * Cdc, 1], [Rdc*Cdc*Ldc, Ldc, Rdc])

    f_stop = 1000
    Nf = 10000

    w = 2 * np.pi * np.linspace(-3 * f_mix - f_stop, 3 * f_mix + f_stop, Nf)
    w, Ydc = signal.freqresp(Ydc_tf, w)
    Zdc = 1 / Ydc

    mix1_idx = np.argmin(np.abs(w - 2 * np.pi * f_mix))
    roll = mix1_idx - Nf // 2

    # dc term
    Ydc_zero = Ydc[Nf//2]

    # fundamental harmonic
    Ydc_plus = np.roll(Ydc, -roll)
    Ydc_minus = np.roll(Ydc, roll)

    # third harmonic
    Ydc_plus3 = np.roll(Ydc, -3*roll) / 3**2
    Ydc_minus3 = np.roll(Ydc, 3*roll) / 3**2

    # fifth harmonic
    Ydc_plus5 = np.roll(Ydc, -5*roll) / 5**2
    Ydc_minus5 = np.roll(Ydc, 5*roll) / 5**2

    Yac = (4 / np.pi**2) * (Ydc_plus + Ydc_minus + \
                            Ydc_plus3 + Ydc_minus3 + \
                            Ydc_plus5 + Ydc_minus5 + \
                            Ydc_zero)
    Zac = 1 / Yac

    # Spice Simulation
    freqs = np.linspace(1, 200, 10)
    freqs = np.floor(freqs)
    timesteps = 1 / (1000 * freqs)
    timesteps = np.clip(timesteps, None, 1 / (1000 * f_mix))

    sp = compute_incremental_impedance(gen_params, freqs, timesteps)

    # Plot Results
    sl1 = slice(Nf // 2, Nf // 2 + roll * 5)
    x1 = w[sl1] / (2 * np.pi)

    sl2 = slice(None)
    x2 = freqs[sl2]

    y11 = 20 * np.log10(np.abs(Zdc[sl1]))
    y12 = 20 * np.log10(np.abs(Zac[sl1]))
    y13 = 20 * np.log10(np.abs(sp[sl2]))

    y21 = np.angle(Zdc[sl1]) * 180 / np.pi
    y22 = np.angle(Zac[sl1]) * 180 / np.pi
    y23 = np.angle(sp[sl2]) * 180 / np.pi

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.set_title("Magnitude")
    ax.semilogx(x1, y11, label='Zdc')
    ax.semilogx(x1, y12, label='Zac')
    ax.semilogx(x2, y13, 'r+')
    ax.axvline(f_mix)
    ax.legend()

   
    ax = plt.subplot(2, 1, 2)
    ax.semilogx(x1, y21)
    ax.semilogx(x1, y22)
    ax.semilogx(x2, y23, 'r+')
    ax.axvline(f_mix)
    
    plt.show()