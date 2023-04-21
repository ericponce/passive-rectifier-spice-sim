
import sim

import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import quad
from scipy import signal
from scipy import interpolate

import argparse

def print_seperator(title):
    s = str(title)
    n = 40 - len(s)//2 - 1
    print('#'*n + ' ' + str(title) + ' ' + '#'*n)

def get_vac(ltr):
    return ltr.get_trace('V(ac_p)').get_wave() - ltr.get_trace('V(ac_n)').get_wave()

def get_iac(ltr):
    return ltr.get_trace('I(R4)').get_wave()

def get_vdc(ltr):
    return ltr.get_trace('V(out)').get_wave()

def get_idc(ltr):
    return ltr.get_trace('I(R2)').get_wave()


def compute_incremental_impedance(model, gen_params, freqs, timesteps, fv=get_vac, fi=get_iac, debug=False):
    sp = np.zeros(len(freqs), dtype=np.cdouble)

    for i, f in enumerate(freqs):
        print("Finding impedance at f=%f, dt=%e, %d of %d" % (f, timesteps[i], i, len(freqs)))
        gen_params['fp'] = f
        gen_params['timestep'] = timesteps[i]
        sim.write_param_file(model+'.gen', gen_params)

        sim.execute_spice(model)

        # Get Data
        ltr = sim.read_raw_file(model)
        t = ltr.get_trace('time').get_wave()
        t = np.abs(t) # fix sign error
        voltage = fv(ltr)
        current = fi(ltr)


        # print(t.shape, voltage.shape, current.shape)

        # plt.figure()
        # plt.plot(t, voltage)
        # plt.plot(t, current)

        # Interpolate
        N = int((t[-1] - t[0])/timesteps[i])
        x = np.linspace(t[0], t[-1], N)
        
        voltage = np.interp(x, t, voltage)
        current = np.interp(x, t, current)
        t = x

        # print(t.shape, voltage.shape, current.shape)
        # plt.plot(t, voltage, 'r+')
        # plt.plot(t, current, 'g*')
        # plt.show()

        # compute complex fourier coefficient
        omega = 2 * np.pi * f
        exp = np.exp(-1j * omega * t)
        voltage_coefficient = np.trapz(voltage * exp, t) * 2 / t[-1];
        current_coefficient = np.trapz(current * exp, t) * 2 / t[-1];

        # compute impedance
        sp[i] = voltage_coefficient / current_coefficient

        if debug:
            t = ltr.get_trace('time').get_wave()
            t = np.abs(t) # fix sign error
            dt = timesteps[i]
            print(dt, np.max(np.diff(t)), np.min(np.diff(t)))
            N = int((t[-1] - t[0])/dt)
            x = np.linspace(t[0], t[-1], N)
            xp = t

            vac = np.interp(x, xp, fv(ltr))[:-1]
            iac = np.interp(x, xp, fi(ltr))[:-1]
            vdc = np.interp(x, xp, get_vdc(ltr))[:-1]
            idc = np.interp(x, xp, get_idc(ltr))[:-1]
            t_interp = x[:-1]
            N = N - 1

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(t, fv(ltr))
            plt.plot(t_interp, vac, 'r+')
            plt.plot(t_interp, iac)
            plt.subplot(2, 1, 2)
            plt.plot(t_interp, vdc)
            plt.plot(t_interp, idc)


            # Take FFT which will be used to find coefficients
            sp_v = np.fft.fft(vac) / N
            sp_v[np.abs(sp_v) < 1e-6] = 0
            sp_i = np.fft.fft(iac) / N
            sp_i[np.abs(sp_i) < 1e-6] = 0
            sp_vdc = np.fft.fft(vdc) / N
            sp_vdc[np.abs(sp_vdc) < 1e-6] = 0
            sp_idc = np.fft.fft(idc) / N
            sp_idc[np.abs(sp_idc) < 1e-6] = 0

            fx = np.fft.fftfreq(N, dt)

            sp_v = np.fft.fftshift(sp_v)
            sp_i = np.fft.fftshift(sp_i)
            sp_vdc = np.fft.fftshift(sp_vdc)
            sp_idc = np.fft.fftshift(sp_idc)
            fx = np.fft.fftshift(fx)

            idx = np.argmin(np.abs(fx - f))

            # f = np.arange(0, 200)
            # w = f * 2 * np.pi
            # sp_mix = sim.fourier_transform(t, mix, w) / len(t)
            # sp_peaks, sp_peak_props = signal.find_peaks(np.abs(sp), height=1e-3, distance=int(30/df))

            plt.figure()
            plt.title("AC")
            plt.subplot(2, 1, 1)
            plt.semilogy(fx, np.abs(sp_v))
            plt.semilogy(fx, np.abs(sp_i))
            plt.plot(fx[idx], np.abs(sp_v[idx]), 'r+')
            plt.plot(fx[idx], np.abs(sp_i[idx]), 'r+')
            # plt.axvline(f)

            plt.subplot(2, 1, 2)
            plt.plot(fx, np.angle(sp_v))
            plt.plot(fx, np.angle(sp_i))
            plt.plot(fx[idx], np.angle(sp_v[idx]), 'r+')
            plt.plot(fx[idx], np.angle(sp_i[idx]), 'r+')
            # plt.axvline(f)

            plt.figure()
            plt.title("DC")
            plt.subplot(2, 1, 1)
            plt.semilogy(fx, np.abs(sp_vdc))
            plt.semilogy(fx, np.abs(sp_idc))
            plt.plot(fx[idx], np.abs(sp_vdc[idx]), 'r+')
            plt.plot(fx[idx], np.abs(sp_idc[idx]), 'r+')
            # plt.axvline(f)

            plt.subplot(2, 1, 2)
            plt.plot(fx, np.angle(sp_vdc))
            plt.plot(fx, np.angle(sp_idc))
            plt.plot(fx[idx], np.angle(sp_vdc[idx]), 'r+')
            plt.plot(fx[idx], np.angle(sp_idc[idx]), 'r+')
            # plt.axvline(f)
            plt.show()


        print("\t Done. |Z(%f)|=%f, <Z(%f)=%f" % (f, np.abs(sp[i]), f, np.angle(sp[i])*180/np.pi))
        # print("\t       |Z(-%f)|=%f, <Z(-%f)=%f" % (f, np.abs(sp_neg), f, np.angle(sp_neg)*180/np.pi))

    return sp

# for the resistive case, it is pretty important that you have many harmonics!
def reflect_admittance(Ydc_tf, w_stop, num_w, w_mix, order=3, debug=False):
    w_start_adj = -w_stop - (2 * order + 1) * w_mix
    w_stop_adj = w_stop + (2 * order + 1) * w_mix
    num_w_adj = int(num_w *(1 + (2 * order + 1) * w_mix / w_stop))
    num_w_adj += (1 if num_w_adj % 2 == 1 else 0)

    w = np.linspace(0, w_stop_adj, num_w_adj//2, endpoint=False)
    w = np.concatenate((np.flip(-w[1:]), w))

    # w = np.linspace(w_start_adj, w_stop_adj, num_w_adj, endpoint=False)
    Nf = len(w)
    _, Ydc = signal.freqresp(Ydc_tf, w)

    mix1_idx = np.argmin(np.abs(w - w_mix))
    roll = mix1_idx - Nf // 2

    if debug:
        x = w / (2 * np.pi)
        fig, ax = plt.subplots(2, 1)
        ax[0].set_xscale('symlog')
        ax[1].set_xscale('symlog')
        ax[0].plot(x, np.abs(Ydc))
        ax[1].plot(x, (np.angle(Ydc)))

    Yac = np.zeros_like(Ydc)

    for i in range(order):
        # component from conduction angle modulation
        # sum Ydc(2nf_mix)*-1/(4n^2-1) from n=-inf to inf
        # first order = Ydc[0]; m = 0
        # second order = Ydc[0] - 1/3 (Ydc[2f_mix] + Ydc[-2f_mix]); m= +-1
        if i == 0:
            comp = Ydc[Nf // 2]
            # print(w[Nf//2], Ydc[Nf//2])
            if debug:
                ax[0].plot(x[Nf//2], np.abs(Ydc[Nf//2]), 'r*')
                ax[1].plot(x[Nf//2], np.angle(Ydc[Nf//2]), 'r*')
        else:
            pidx = Nf // 2 + 2 * i * roll
            nidx = Nf // 2 - 2 * i * roll
            comp = -1 / (4 * i**2 - 1) * (Ydc[pidx] + Ydc[nidx])
            # print(w[pidx]/2/np.pi, w[nidx]/2/np.pi, Ydc[pidx] + Ydc[nidx])
            if debug:
                ax[0].plot(x[pidx], np.abs(Ydc[pidx]), 'r*')
                ax[0].plot(x[nidx], np.abs(Ydc[nidx]), 'g*')
                ax[1].plot(x[pidx], np.angle(Ydc[pidx]), 'r*')
                ax[1].plot(x[nidx], np.angle(Ydc[nidx]), 'g*')


        # components from mixing signal
        # sum 1/(2i+1)^2(Ydc[f - (2m+1)f_mix] + Ydc[f + (2m+1)f_mix]) from m=-inf to inf
        # first order (m = 0, -1) = Ydc[f-f_mix] + Ydc[f+f_mix]
        # second order (m = -2, 1) = f.o. + 1/3^2(Ydc[f+3f_mix] + Ydc[f-3f_mix])
        plus = np.roll(Ydc, -(2*i + 1) * roll) / (2*i + 1)**2
        minus = np.roll(Ydc, (2*i + 1) * roll) / (2*i + 1)**2 ###### wrong i #######
        # print(w[Nf // 2+(2*i + 1) * roll]/2/np.pi, w[Nf // 2-(2*i + 1) * roll]/2/np.pi)
        Yac += comp + plus + minus

        if debug:
            ax[0].axhline(np.abs(comp), color='black', ls=':', alpha=0.5)
            ax[0].plot(x, np.abs(plus))
            ax[0].plot(x, np.abs(minus))

            ax[1].axhline(np.angle(comp), color='black', ls=':', alpha=0.5)
            ax[1].plot(x, np.unwrap(np.angle(plus)))
            ax[1].plot(x, np.unwrap(np.angle(minus)))

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

    parser_rlc = subparsers.add_parser('r', help='resistive model')
    parser_rlc.add_argument('--R', type=float, default=10,
                            help='Resistance in Ohms')
    parser_rlc.add_argument('--f', type=int, default=60,
                            help='Line (Grid) Frequency in Hz')

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
    parser_pfc_cpl.add_argument('--fcpl', type=float, default=10,
                                help='CPL Bandwidth in Hz')
    parser_pfc_cpl.add_argument('--Ydc', type=float, default=0.1,
                                help='CPL Admittance in S')
    parser_pfc_cpl.add_argument('--f', type=int, default=60,
                            help='Line (Grid) Frequency in Hz')

    parser_dc_cpl = subparsers.add_parser('dc_cpl', help='dc cpl model (time domain)')
    parser_dc_cpl.add_argument('--Cdc', type=float, default=1e-6,
                                help='Input Capacitor in F')
    parser_dc_cpl.add_argument('--fcpl', type=float, default=10,
                                help='CPL Bandwidth in Hz')
    parser_dc_cpl.add_argument('--Ydc', type=float, default=0.1,
                                help='CPL Admittance in S')

    parser_dc_ac_cpl = subparsers.add_parser('dc_ac_cpl', help='dc cpl model (ac analysis)')
    parser_dc_ac_cpl.add_argument('--Cdc', type=float, default=1e-6,
                                help='Input Capacitor in F')
    parser_dc_ac_cpl.add_argument('--fcpl', type=float, default=10,
                                help='CPL Bandwidth in Hz')
    parser_dc_ac_cpl.add_argument('--Ydc', type=float, default=0.1,
                                help='CPL Admittance in S')


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

    elif model == 'r':
        print("Simple Passive resistor Model")
        model = 'base_R'

        gen_params = {
            'fl':args.f,
            'fp':0,
            'Rdc':args.R,
            'transtop':2,
            'transtart':1,
            'timestep':1e-6
        }

        f_mix = gen_params['fl']
        R = gen_params['Rdc']

        Ydc_tf = signal.lti([1/R], [1])

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
            'timestep':1e-6,
            'Vg':100,
            'Vp':10
        }

        Ydc_tf = signal.lti([Cdc, Cdc*w_cpl + Ydc, -Ydc * w_cpl], [0, 1, w_cpl])

    elif model == 'dc_cpl':
        print("DC CPL Model")
        model = 'dc_PFC_CPL'

        w_cpl = args.fcpl * 2 * np.pi
        Cdc = args.Cdc 
        Ydc = args.Ydc

        gen_params = {
            'fp':0,
            'Ydc':Ydc,
            'Cdc':Cdc,
            'wCPL':w_cpl,
            'transtop':2,
            'transtart':1,
            'timestep':1e-6
        }

        Ydc_tf = signal.lti([Cdc, Cdc*w_cpl + Ydc, -Ydc * w_cpl], [0, 1, w_cpl])
        # Spice Simulation
        freqs = np.linspace(1, 200, 30)
        freqs = np.floor(freqs)
        timesteps = 1 / (100 * freqs)

        # Simulate in SPICE to get incremental impedance
        sp = compute_incremental_impedance('dc_PFC_CPL', gen_params, freqs, timesteps, get_vdc, get_idc)

        # Analytic Estimation
        f_stop = 500 # will be adjusted to remove edge effects
        Nf = 10000
        w = np.linspace(-2*np.pi*f_stop, 2*np.pi*f_stop, Nf)
        Nf = len(w)
        _, Ydc = signal.freqresp(Ydc_tf, w)
        Zdc = 1 / Ydc

        sl1 = slice(Nf // 2, None)
        x1 = w[sl1] / (2 * np.pi)

        sl2 = slice(None)
        x2 = freqs[sl2]

        y11 = 20 * np.log10(np.abs(Zdc[sl1]))
        # y12 = 20 * np.log10(np.abs(Zac[sl1]))
        y13 = 20 * np.log10(np.abs(sp[sl2]))

        y21 = (np.angle(Zdc[sl1])) * 180 / np.pi
        # y22 = (np.angle(Zac[sl1])) * 180 / np.pi
        y23 = (np.angle(sp[sl2])) * 180 / np.pi

        plt.figure()
        ax = plt.subplot(2, 1, 1)
        ax.set_title("Magnitude")
        ax.semilogx(x1, y11, label='Zdc')
        # ax.semilogx(x1, y12, label='Zac')
        ax.semilogx(x2, y13, 'r+')
        ax.legend()

       
        ax = plt.subplot(2, 1, 2)
        ax.semilogx(x1, y21)
        # ax.semilogx(x1, y22)
        ax.semilogx(x2, y23, 'r+')
        
        plt.show()

        exit()
    # This doesn't work very well
    elif model == 'dc_ac_cpl':
        print("DC CPL Model (AC Analysis)")
        model = 'dc_ac_PFC_CPL'

        fstart = 1
        fstop = 200
        Nf = 30

        gen_params = {
            'fp':0,
            'Ydc':args.Ydc,
            'Cdc':args.Cdc,
            'wCPL':args.fcpl * 2 * np.pi,
            'fstop':fstop,
            'fstart':fstart,
            'num_pts':Nf
        }

        # Simulate
        sim.write_param_file(model+'.gen', gen_params)
        sim.execute_spice(model)
        ltr = sim.read_raw_file(model)

        exit()

    # Analytic Estimation
    print_seperator('Analytical Estimation')
    f_stop = 5000 # will be adjusted to remove edge effects
    Nf = 10000
    w, Ydc, Yac = reflect_admittance(Ydc_tf, 2*np.pi*f_stop, Nf, 2*np.pi*f_mix, order=100, debug=args.debug)
    Zdc, Zac = 1 / Ydc, 1 / Yac
    Nf = len(w)

    # Plot Results
    sl1 = slice(Nf // 2, None)
    sl1 = slice(None)
    x1 = w[sl1] / (2 * np.pi)

    y11 = 20 * np.log10(np.abs(Zdc[sl1]))
    y12 = 20 * np.log10(np.abs(Zac[sl1]))

    y21 = (np.angle(Zdc[sl1])) * 180 / np.pi
    y22 = (np.angle(Zac[sl1])) * 180 / np.pi

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.set_title("Magnitude")
    ax.set_xscale('symlog')
    ax.plot(x1, y11, label='Zdc')
    ax.plot(x1, y12, label='Zac')
    ax.axvline(f_mix)
    ax.legend()
 
    ax = plt.subplot(2, 1, 2)
    ax.set_xscale('symlog')
    ax.plot(x1, y21)
    ax.plot(x1, y22)
    ax.axvline(f_mix)    
    plt.show()

    print_seperator('Mixing Signal Analysis')
    # Simulate in SPICE with no pertubation to getsteady state mixing signal
    # it is *super important that this number i slarger enough to account for expect
    #       very *tiny* deviations in conduction window (delta t from sun paper)
    gen_params['fp'] = 10
    gen_params['timestep'] = (gen_params['Vp'] / (2*np.pi*(gen_params['Vg']*f_mix + gen_params['Vp']*gen_params['fp']))) / 10
    sim.write_param_file(model+'.gen', gen_params)
    sim.execute_spice(model)

    interpolate_dt = gen_params['timestep']
    t, mix = sim.extract_mixing_signal(model, interpolate_dt=interpolate_dt)
    mix_unperturbed = np.cos(2 * np.pi * f_mix * t)/np.abs(np.cos(2 * np.pi * f_mix * t))
    N = len(t)

    sp_mix = np.fft.fft(mix) / N
    f = np.fft.fftfreq(N, t[1] - t[0])

    sp_mix = np.fft.fftshift(sp_mix)
    f = np.fft.fftshift(f)

    # f_max = 500
    # f_sl = slice(np.argmin(np.abs(f + f_max)), np.argmin(np.abs(f - f_max)))

    # plt.figure()
    # plt.subplot(4, 1, 1)
    # plt.plot(t, mix)
    # plt.subplot(4, 1, 2)
    # plt.plot(t, mix - mix_unperturbed)
    # plt.subplot(4, 1, 3)
    # plt.semilogy(f[f_sl], np.abs(sp_mix[f_sl]))
    # plt.axhline(gen_params['Vp'] / (gen_params['Vg'] * np.pi))
    # plt.subplot(4, 1, 4)
    # plt.plot(f[f_sl], np.angle(sp_mix[f_sl])*180/np.pi)
    # plt.show()

    

    t, vac, iac, vdc, idc = sim.get_voltage_and_current(model, interpolate_dt=interpolate_dt)
    N = len(t)
    f = np.fft.fftshift(np.fft.fftfreq(N, t[1] - t[0]))
    sp_vdc = np.fft.fftshift(np.fft.fft(vdc) / N)
    sp_vac = np.fft.fftshift(np.fft.fft(vac) / N)
    sp_idc = np.fft.fftshift(np.fft.fft(idc) / N)
    sp_iac = np.fft.fftshift(np.fft.fft(iac) / N)

    f_max = 1000
    f_sl = slice(np.argmin(np.abs(f + f_max)), np.argmin(np.abs(f - f_max)))

    # # f = f1
    # # sp_vdc = sim.fourier_transform(t, vdc, w) / N
    # # sp_vac = sim.fourier_transform(t, vac, w) / N
    # # sp_idc = sim.fourier_transform(t, idc, w) / N
    # # sp_iac = sim.fourier_transform(t, iac, w) / N

    plt.figure()
    plt.subplot(3, 2, 1)
    plt.plot(t, vac)
    plt.twinx().plot(t, iac)
    plt.subplot(3, 2, 2)
    plt.plot(f[f_sl], np.abs(sp_vac[f_sl]))
    plt.plot(f[f_sl], np.abs(sp_iac[f_sl]))

    plt.subplot(3, 2, 3)
    plt.plot(t, mix)
    plt.subplot(3, 2, 4)
    plt.plot(f[f_sl], np.abs(sp_mix[f_sl]))

    plt.subplot(3, 2, 5)
    plt.plot(t, vdc)
    plt.twinx().plot(t, idc)
    plt.subplot(3, 2, 6)
    plt.plot(f[f_sl], np.abs(sp_vdc[f_sl]))
    plt.plot(f[f_sl], np.abs(sp_idc[f_sl]))
    plt.tight_layout()
    plt.show()


    # import pdb; pdb.set_trace()
    print_seperator('Simulation')
    # Spice Simulation
    freqs = np.linspace(1, 200, 30)
    freqs = np.floor(freqs)
    timesteps = (gen_params['Vp'] / (2*np.pi*(gen_params['Vg']*f_mix + gen_params['Vp']*freqs))) / 10

    # Simulate in SPICE to get incremental impedance
    sp = compute_incremental_impedance(model, gen_params, freqs, timesteps, debug=args.debug)

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