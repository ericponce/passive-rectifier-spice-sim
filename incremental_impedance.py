
import sim

import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import quad
from scipy import signal

import argparse

def compute_incremental_impedance(model, gen_params, freqs, timesteps):
    sp = np.zeros(len(freqs), dtype=np.cdouble)

    for i, f in enumerate(freqs):
        gen_params['fp'] = f
        gen_params['timestep'] = timesteps[i]
        sim.write_param_file(model+'.gen', gen_params)

        sim.execute_spice(model)

        ltr = sim.read_raw_file(model)

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

        sp[i] = vac_pert / iac_pert

        print("f=%f, %d of %d complete." % (f, i, len(freqs)))

    return sp

def compute_conduction_times(model, gen_params, timestep):
    gen_params['fp'] = 0
    gen_params['timestep'] = timestep

    sim.write_param_file(model+'.gen', gen_params)
    sim.execute_spice(model)
    ltr = sim.read_raw_file(model)

    t = np.abs(ltr.get_trace('time').get_wave())

    iPos = ltr.get_trace('I(D5)').get_wave()
    iNeg = ltr.get_trace('I(D6)').get_wave()
    mix = np.zeros_like(t)


    # Digitize Diode Current
    iPos[iPos > 0] = 1
    iPos[iPos <= 0] = 0
    iNeg[iNeg > 0] = 1
    iNeg[iNeg <= 0] = 0

    # Form mixing signal
    mix[iPos > 0] = 1
    mix[iNeg > 0] = -1

    # Find transistion points using the derivative of the square wave
    diPos = np.diff(iPos)
    diNeg = np.diff(iNeg)


    pos_conduction = np.concatenate((
                        np.where(diPos > 0)[0] + 1,
                        np.where(diPos < 0)[0]))

    pos_conduction = np.sort(pos_conduction)

    neg_conduction = np.concatenate((
                        np.where(diNeg > 0)[0] + 1,
                        np.where(diNeg < 0)[0]))

    neg_conduction = np.sort(neg_conduction)


    return pos_conduction, neg_conduction, t, mix

def reflect_admittance(Ydc_tf, w_stop, num_w, w_mix, num_harmonics=3, debug=False):

    w_start_adj = -w_stop - (2 * num_harmonics + 1) * w_mix
    w_stop_adj = w_stop + (2 * num_harmonics + 1) * w_mix
    num_w_adj = int(num_w *(1 + (2 * num_harmonics + 1) * w_mix / w_stop)) + 1

    w = np.linspace(w_start_adj, w_stop_adj, num_w_adj)
    Nf = len(w)
    _, Ydc = signal.freqresp(Ydc_tf, w)
    Zdc = 1 / Ydc

    mix1_idx = np.argmin(np.abs(w - w_mix))
    roll = mix1_idx - Nf // 2

    # dc term
    # Ydc_zero = Ydc[Nf//2]
    # Ydc_one = -(Ydc[Nf // 2 + 2 * roll] + Ydc[Nf // 2 - 2 * roll]) / 3

    if debug:
        fig, ax = plt.subplots(2, 1)
        ax[0].set_xscale('symlog')
        ax[1].set_xscale('symlog')
        ax[0].plot(w, np.abs(Ydc))
        ax[1].plot(w, np.unwrap(np.angle(Ydc)))

    Yac = np.zeros_like(Ydc)

    for i in range(num_harmonics):
        # component from conduction angle modulation
        if i == 0:
            comp = Ydc[Nf // 2]
        else:
            comp = -1 / (4 * i**2 - 1) * (Ydc[Nf // 2 + 2 * i * roll] + Ydc[Nf // 2 - 2 * i * roll])


        plus = np.roll(Ydc, -(2*i + 1) * roll) / (2*i + 1)**2
        minus = np.roll(Ydc, (2*i + 1) * roll) / (2*i + 1)**2

        Yac += comp + plus + minus

        if debug:
            ax[0].axhline(np.abs(comp), color='black', ls=':', alpha=0.5)
            ax[0].plot(w, np.abs(plus))
            ax[0].plot(w, np.abs(minus))

            ax[1].axhline(np.angle(comp), color='black', ls=':', alpha=0.5)
            ax[1].plot(w, np.unwrap(np.angle(plus)))
            ax[1].plot(w, np.unwrap(np.angle(minus)))

    Yac *= (4 / np.pi**2)

    if debug:
        plt.show()

    # cleanup
    w_stop1_idx = np.argmin(np.abs(w - (-w_stop)))
    w_stop2_idx = np.argmin(np.abs(w - w_stop))

    sl = slice(w_stop1_idx, w_stop2_idx)
    w = w[sl]
    Ydc = Ydc[sl]
    Yac = Yac[sl]

    return w, Ydc, Yac

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Passive Rectifier Circuits")
    # parser.add_argument('model', choices=['rlc', 'pfc_cpl'], 
    #                     help='model to use')

    parser.add_argument('--debug', action='store_true')

    subparsers = parser.add_subparsers(dest='model', help="rlc model")
    parser_rlc = subparsers.add_parser('rlc', help='rlc model')
    parser_rlc.add_argument('--R', type=float, default=10,
                            help='Resistance in Ohms')
    parser_rlc.add_argument('--L', type=float, default=50e-3,
                            help='Inductance in H')
    parser_rlc.add_argument('--C', type=float, default=1000e-6,
                            help='Capacitance in F')
    parser_rlc.add_argument('--f', type=int, default=60,
                            help='Line (Grid) Frequency in Hz')

    parser_pfc_cpl = subparsers.add_parser('pfc_cpl', help='pfc cpl model')
    parser_pfc_cpl.add_argument('--Cdc', type=float, default=1e-6,
                                help='Input Capacitor in F')
    parser_pfc_cpl.add_argument('--fcpl', type=float, default=2,
                                help='CPL Bandwidth in Hz')
    parser_pfc_cpl.add_argument('--Ydc', type=float, default=0.1,
                                help='CPL Admittance in S')
    parser_pfc_cpl.add_argument('--f', type=int, default=60,
                            help='Line (Grid) Frequency in Hz')


    args = parser.parse_args()
    model = args.model
    # model = args.model
    # params = args.params 

    if model == 'rlc':
        print("Simple Passive RLC Model")
        model = 'base_RLC'

        gen_params = {
            'fl':args.f,
            'fp':0,
            'Ldc':args.L,
            'Cdc':args.C,
            'Rdc':args.R,
            'transtop':2,
            'transtart':1,
            'timestep':1e-6
        }

        f_mix = gen_params['fl']

        L = gen_params['Ldc']
        C = gen_params['Cdc']
        R = gen_params['Rdc']

        Ydc_tf = signal.lti([0, C*R, 1], [L*C*R, L, R])

    elif model == 'pfc_cpl':
        print("Passive PFC CPL Model")
        model = 'base_PFC_CPL'

        w_cpl = args.fcpl * 2 * np.pi
        Cdc = args.Cdc 
        Ydc = args.Ydc
        f_mix = args.f

        gen_params = {
            'fl':f_mix,
            'fp':0,
            'Ydc':Ydc,
            'Cdc':Cdc,
            'wCPL':w_cpl,
            'transtop':2,
            'transtart':1,
            'timestep':1e-6
        }

        Ydc_tf = signal.lti([Cdc, Cdc*w_cpl + Ydc, -Ydc * w_cpl], [0, 1, w_cpl])

    # Analytic Estimation
    f_stop = 500 # will be adjusted to remove edge effects
    Nf = 10000
    w, Ydc, Yac = reflect_admittance(Ydc_tf, 2*np.pi*f_stop, Nf, 2*np.pi*f_mix, num_harmonics=2, debug=args.debug)
    Zdc, Zac = 1 / Ydc, 1 / Yac
    Nf = len(w)

    Z1 = Zdc[Nf//2]
    mix1_idx = np.argmin(np.abs(w - 2*np.pi*f_mix))
    Z2 = Zdc[mix1_idx]
    mix1_idx = np.argmin(np.abs(w + 2*np.pi*f_mix))
    Z3 = Zdc[mix1_idx]

    print(1 / Z1, 1 / Z2, 1 / Z3)
    print((1 / Z1 + 1 / Z2 + 1 / Z3))
    print(np.abs((1 / Z1 + 1 / Z2 + 1 / Z3)))
    print(np.abs(1 / (1 / Z1 + 1 / Z2 + 1 / Z3)))

    # Plot Analytic Estimate
    sl1 = slice(Nf // 2, None)
    x1 = w[sl1] / (2 * np.pi)

    y11 = 20 * np.log10(np.abs(Zdc[sl1]))
    y12 = 20 * np.log10(np.abs(Zac[sl1]))

    y21 = np.unwrap(np.angle(Zdc[sl1])) * 180 / np.pi
    y22 = np.unwrap(np.angle(Zac[sl1])) * 180 / np.pi

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.set_title("Magnitude")
    ax.semilogx(x1, y11, label='Zdc')
    ax.semilogx(x1, y12, label='Zac')
    ax.axvline(f_mix)
    ax.legend()

   
    ax = plt.subplot(2, 1, 2)
    ax.semilogx(x1, y21)
    ax.semilogx(x1, y22)
    ax.axvline(f_mix)
    plt.show()

    # Spice Simulation
    freqs = np.linspace(1, 200, 30)
    freqs = np.floor(freqs)
    timesteps = 1 / (1000 * freqs)
    timesteps = np.clip(timesteps, None, 1 / (1000 * f_mix))

    # Simulate in SPICE with no pertubation to getsteady state conduction times
    pos_conduction_idx, neg_conduction_idx, t, mix = compute_conduction_times(model, gen_params, timesteps[0])

    # Simulate in SPICE to get incremental impedance
    sp = compute_incremental_impedance(model, gen_params, freqs, timesteps)

    # Plot Results
    sl1 = slice(Nf // 2, None)
    x1 = w[sl1] / (2 * np.pi)

    sl2 = slice(None)
    x2 = freqs[sl2]

    y11 = 20 * np.log10(np.abs(Zdc[sl1]))
    y12 = 20 * np.log10(np.abs(Zac[sl1]))
    y13 = 20 * np.log10(np.abs(sp[sl2]))

    y21 = (np.angle(Zdc[sl1])) * 180 / np.pi
    y22 = (np.angle(Zac[sl1])) * 180 / np.pi
    y23 = (np.angle(sp[sl2])) * 180 / np.pi

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