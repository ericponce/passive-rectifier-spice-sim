
import sim
import analytic
import plot_utils

import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import quad
from scipy import signal
from scipy import interpolate

from scipy.interpolate import interp1d

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

def compute_incremental_impedance(model, gen_params, freqs, timesteps, stoptimes, fv=get_vac, fi=get_iac, debug=False):
    sp = np.zeros(len(freqs), dtype=np.cdouble)

    for i, f in enumerate(freqs):
        print("Finding impedance at f=%f, dt=%e, %d of %d" % (f, timesteps[i], i, len(freqs)))
        gen_params['fp'] = f
        gen_params['timestep'] = timesteps[i]
        gen_params['transtop'] = stoptimes[i]
        sim.write_param_file(model+'.gen', gen_params)

        sim.execute_spice(model)

        # Get Data
        ltr = sim.read_raw_file(model)
        t = ltr.get_trace('time').get_wave()
        t = np.abs(t) # fix sign error
        voltage = fv(ltr)
        current = fi(ltr)

        print('Sim Time: %fs to %fs, %d pts' % (t[0], t[-1], len(t)))




        # Interpolate
        N = 10 * len(t)
        N = np.max((N, 250000))
        x, dx = np.linspace(1, t[-1], N, endpoint=False, retstep=True)
        print('Interpolating time: 1s to %fs, %d pts' % (t[-1], len(x)))
        
        sl = slice(np.argmin(np.abs(t-1))-1, None)
        v_interp = interpolate.interp1d(t[sl], voltage[sl], 'quadratic')
        i_interp = interpolate.interp1d(t[sl], current[sl], 'quadratic')
        voltage = v_interp(x)
        current = i_interp(x)

        if debug:
            vdc_interp = interpolate.interp1d(t[sl], get_vdc(ltr)[sl], 'quadratic')
            idc_interp = interpolate.interp1d(t[sl], get_idc(ltr)[sl], 'quadratic')
            vdc = vdc_interp(x)
            idc = idc_interp(x)


        t = x
        dt = dx

        # compute complex fourier coefficient of impedance
        omega = 2 * np.pi * f
        exp = np.exp(-1j * omega * t)
        sp[i] = np.trapz(voltage * exp, t) / np.trapz(current * exp, t)

        t_tmp = ltr.get_trace('time').get_time_axis()[sl]
        exp = np.exp(-1j * omega * t_tmp)
        sp[i] = np.trapz(fv(ltr)[sl] * exp, t_tmp) / np.trapz(fi(ltr)[sl] * exp, t_tmp)

        if debug:
            v_coeff = np.trapz(fv(ltr)[sl] * exp, t_tmp) * 2 / (t_tmp[-1] - t_tmp[0])
            i_coeff = np.trapz(fi(ltr)[sl] * exp, t_tmp) * 2 / (t_tmp[-1] - t_tmp[0])

            vac = voltage
            iac = current

            plt.figure()
            plt.suptitle("Voltage and Current")
            plt.subplot(2, 1, 1)
            plt.title("Measurement Side")
            plt.plot(ltr.get_trace('time').get_time_axis(), fv(ltr))
            plt.plot(t, voltage, 'r+')
            plt.plot(ltr.get_trace('time').get_time_axis(), fi(ltr))
            plt.plot(t, current, 'g*')
            plt.subplot(2, 1, 2)
            plt.title("DC Side")
            plt.plot(t, vdc)
            plt.plot(t, idc)
            plt.tight_layout()

            # Take FFT which will be used to find coefficients
            sp_v = np.fft.fft(voltage) / N
            sp_i = np.fft.fft(current) / N
            sp_vdc = np.fft.fft(vdc) / N
            sp_idc = np.fft.fft(idc) / N

            fx = np.fft.fftfreq(N, dt)

            sl = slice(0, np.argmin(np.abs(fx-1300)))

            sp_v = sp_v[sl]
            sp_i = sp_i[sl]
            sp_vdc = sp_vdc[sl]
            sp_idc = sp_idc[sl]
            fx = fx[sl]

            idx = np.argmin(np.abs(fx - f))

            plt.figure()
            
            plt.subplot(2, 1, 1)
            plt.title("Measurement Side Spectrum Magnitude")
            plt.semilogx(fx, 20*np.log10(np.abs(sp_v)))
            plt.semilogx(fx, 20*np.log10(np.abs(sp_i)))
            plt.plot(fx[idx], 20*np.log10(np.abs(sp_v[idx])), 'r+')
            plt.plot(fx[idx], 20*np.log10(np.abs(sp_i[idx])), 'g*')
            plt.plot(fx[idx], 20*np.log10(np.abs(v_coeff/2)), 'b*')
            plt.plot(fx[idx], 20*np.log10(np.abs(i_coeff/2)), 'y+')
            # plt.axvline(f)

            plt.subplot(2, 1, 2)
            plt.title("Measurement Side Spectrum Phase")
            plt.semilogx(fx, np.angle(sp_v))
            plt.semilogx(fx, np.angle(sp_i))
            plt.plot(fx[idx], np.angle(sp_v[idx]), 'r+')
            plt.plot(fx[idx], np.angle(sp_i[idx]), 'g*')
            # plt.axvline(f)

            plt.figure()
            plt.subplot(2, 1, 1)
            plt.title("DC Side Spectrum Magnitude")
            plt.semilogx(fx, 20*np.log10(np.abs(sp_vdc)))
            plt.semilogx(fx, 20*np.log10(np.abs(sp_idc)))

            plt.subplot(2, 1, 2)
            plt.title("DC Side Spectrum Phase")
            plt.semilogx(fx, np.angle(sp_vdc))
            plt.semilogx(fx, np.angle(sp_idc))
            plt.show()


        print("\t Done. |Z(%f)|=%f, <Z(%f)=%f" % (f, np.abs(sp[i]), f, np.angle(sp[i])*180/np.pi))
    return sp

def compute_incremental_impedance2(model, gen_params, freqs, timestep, fv=get_vac, fi=get_iac):
    # General Configuration
    gen_params['timestep'] = timestep
    gen_params['timestart'] = 0
    gen_params['timestop'] = 2

    # Allocate Array for final result
    sp = np.zeros(len(freqs), dtype=np.cdouble)
    sp_mirror = np.zeros(len(freqs), dtype=np.cdouble)

    print("Beginning unperturbed simulation")
    Vp = gen_params['Vp'] 
    gen_params['Vp'] = 0
    gen_params['fp'] = 0
    sim.write_param_file(model+'.gen', gen_params)
    sim.execute_spice(model)

    # Get Data
    ltrUnperturbed = sim.read_raw_file(model)

    print("Beginning perturbed simulations")
    for i, f in enumerate(freqs):
        print("Finding impedance at f=%f, %d of %d" % (f, i, len(freqs)))
        gen_params['fp'] = f
        gen_params['Vp'] = Vp
        sim.write_param_file(model+'.gen', gen_params)
        sim.execute_spice(model)

        # Get Data
        ltr = sim.read_raw_file(model)

        # compute complex fourier coefficient of impedance (perturbed and unperturbed)
        omega = 2 * np.pi * f

        tUnperturbed = ltrUnperturbed.get_trace('time').get_time_axis()
        tPerturbed = ltr.get_trace('time').get_time_axis()

        slU = slice(np.argmin(np.abs(tUnperturbed-1)) - 1, None)
        slP = slice(np.argmin(np.abs(tPerturbed-1)) - 1, None)

        tU = tUnperturbed[slU]
        tP = tPerturbed[slP]

        expU = np.exp(-1j * omega * tU)
        expP = np.exp(-1j * omega * tP)

        spV = np.trapz(fv(ltr)[slP] * expP, tP) - np.trapz(fv(ltrUnperturbed)[slU] * expU, tU)
        spI = np.trapz(fi(ltr)[slP] * expP, tP) - np.trapz(fi(ltrUnperturbed)[slU] * expU, tU)
        sp[i] = spV / spI

        # Find coefficient for fundamental-mirrored perturbation frequency
        omega = 2 * np.pi * (2 * gen_params['fl'] - f)
                             
        expU = np.exp(-1j * omega * tU)
        expP = np.exp(-1j * omega * tP)
        # spV = np.trapz(fv(ltr)[slP] * expP, tP) - np.trapz(fv(ltrUnperturbed)[slU] * expU, tU) # do not recompute the voltage, jsut the current
        spI = np.trapz(fi(ltr)[slP] * expP, tP) - np.trapz(fi(ltrUnperturbed)[slU] * expU, tU)
        sp_mirror[i] = spV / spI

        print("\t Done. |Z(%f)|=%f, <Z(%f)=%f" % (f, np.abs(sp[i]), f, np.angle(sp[i])*180/np.pi))

    return sp, sp_mirror

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
            comp = (Ydc[Nf // 2]) # why does putting abs here this help pfc
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
            ax[1].plot(x, (np.angle(plus)))
            ax[1].plot(x, (np.angle(minus)))

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

    parser_cpl = subparsers.add_parser('cpl', help='cpl model')
    parser_cpl.add_argument('--Cdc', type=float, default=1e-6,
                                help='Input Capacitor in F')
    parser_cpl.add_argument('--Ydc', type=float, default=0.1,
                                help='CPL Admittance in S')
    parser_cpl.add_argument('--f', type=int, default=60,
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

    nonlinear = False

    gen_params = {
            'fl':args.f,
            'fp':0,
            'transtop':2,
            'transtart':0,
            'timestep':50e-6,
            'Vg':100,
            'Vp':1
    }

    if model == 'r':
        print("Passive full-bridge recitifier with resistive load")
        model = 'ss_models/base_R'
        gen_params['Rdc'] = args.R

        Ydc_tf = signal.lti([1/gen_params['Rdc']], [1])

    elif model == 'rlc':
        print("Simple Passive RLC Model")
        model = 'ss_models/base_RLC'

        gen_params['Ldc'] = args.L
        gen_params['Cdc'] = args.C
        gen_params['Rdc'] = args.R

        gen_params['timestep'] = 10e-6

        Ydc_tf = signal.lti([0, args.C*args.R, 1], 
                            [args.L*args.C*args.R, args.L, args.R])

    elif model == 'pfc_cpl':
        print("Passive PFC CPL Model")
        model = 'ss_models/base_PFC_CPL'

        w_cpl = args.fcpl * 2 * np.pi

        gen_params['Ydc'] = args.Ydc
        gen_params['Cdc'] = args.Cdc
        gen_params['wCPL'] = w_cpl

        Ydc_tf = signal.lti([args.Cdc, args.Cdc*w_cpl + args.Ydc, -args.Ydc * w_cpl], [0, 1, w_cpl])


    # Analytic Estimation
    print_seperator('Analytical Estimation')
    f_stop = 2500 # will be adjusted to remove edge effects
    Nf = 50000

    w_stop = 2 * np.pi * f_stop
    w_mix = 2 * np.pi * gen_params['fl']
    if model == 'base_PFC_CPL':
        # w, Ydc, Yac = analytic.CBCPL_SimpleYac2(Ydc_tf, w_stop, Nf, w_mix, order=100, debug=args.debug)
        # w, Ydc, Yac = analytic.CBCPL_SimpleYac(Ydc_tf, w_stop, Nf, w_mix, order=1, debug=args.debug)
        w, Ydc, Yac = analytic.CBCPL_ApproximateYac(args.Ydc, args.fcpl, f_stop, Nf, gen_params['fl'])

    else:
        w, Ydc, Yac = reflect_admittance(Ydc_tf, w_stop, Nf, w_mix, order=100, debug=args.debug)

    
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
    ax.axvline(gen_params['fl'])
    ax.legend()
 
    ax = plt.subplot(2, 1, 2)
    ax.set_xscale('symlog')
    ax.plot(x1, y21)
    ax.plot(x1, y22)
    ax.axvline(gen_params['fl'])    
    plt.show()

    mixingSignalAnalysis = True
    incrementalMixingSignalAnalysis = False
    if mixingSignalAnalysis:
        print_seperator('Mixing Signal Analysis')

        Vp = gen_params['Vp']
        interpolate_dt = 1e-6
        if incrementalMixingSignalAnalysis:
            # First, without a pertubation
            gen_params['Vp'] = 0
            gen_params['fp'] = 0
            
            # simulate
            sim.write_param_file(model+'.gen', gen_params)
            sim.execute_spice(model)

            fig, ax = plt.subplots(8, 2)
            incremental = plot_utils.plot_rectifier_ports(fig, ax, model, interpolate_dt, 1000)

            plt.tight_layout()
            plt.show()

        # Simulate in SPICE with a pertubation
        gen_params['fp'] = 30
        gen_params['Vp'] = Vp

        # Compute DC Side Terms
        an_f_vdc, an_sp_vdc = analytic.CCPR_Vdc(lineVoltage=gen_params['Vg'], 
                                    lineFrequency=gen_params['fl'], 
                                    perturbationVoltage=gen_params['Vp'], 
                                    perturbationFrequency=gen_params['fp'],
                                    maximumMixingOrder = 10, 
                                    positiveFrequenciesOnly=False,
                                    combineLikeTerms=True,
                                    includeSecondOrder=False)
        
        an_sp_vdc = an_sp_vdc - 1j * 1e-16

        an_sp_idc_linear = analytic.LL_Idc(an_f_vdc, an_sp_vdc, Ydc_tf)

        if model == 'base_PFC_CPL':
            bias = 2 * gen_params['Vg']/ np.pi
            power = bias**2 * gen_params['Ydc']
            cutoff = gen_params['wCPL']
            an_sp_idc_nonlinear = analytic.CBCPL_Idc2(cutoff, an_f_vdc, an_sp_vdc, power, bias)
            # an_sp_idc_nonlinear = analytic.CBCPL_Idc2(cutoff, Ydc_tf, an_f_vdc, an_sp_vdc, power, bias)
            # an_sp_idc_nonlinear = analytic.CBCPL_Idc3(cutoff, Ydc_tf, an_f_vdc, an_sp_vdc, gen_params['fl'])
        elif model == 'base_CPL':
            bias = 2 * gen_params['Vg']/ np.pi
            power = bias**2 * gen_params['Ydc']
            an_sp_idc_nonlinear = analytic.CPL_Idc(an_f_vdc, an_sp_vdc, power, bias)

        

        # slice
        sl = slice(np.argmin(np.abs(an_f_vdc)), np.argmin(np.abs(an_f_vdc - 1000)))
        an_f_vdc = an_f_vdc[sl]
        an_sp_vdc = an_sp_vdc[sl]
        an_sp_idc_linear = an_sp_idc_linear[sl]

        if model == 'base_PFC_CPL' or model == 'base_CPL':
            an_sp_idc_nonlinear = an_sp_idc_nonlinear[sl]

        # print(an_f_vdc[:10])
        # print(np.abs(an_sp_idc_nonlinear / an_sp_idc_linear)[:10])
        # print(np.angle(an_sp_idc_nonlinear / an_sp_idc_linear)[:10] * 180 / np.pi)

        # simulate
        sim.write_param_file(model+'.gen', gen_params)
        sim.execute_spice(model)

        fig, ax = plt.subplots(8, 2)
        if incrementalMixingSignalAnalysis:
            plot_utils.plot_rectifier_ports(fig, ax, model, interpolate_dt, 1000, incremental=incremental)
        else:
            plot_utils.plot_rectifier_ports(fig, ax, model, interpolate_dt, 1000)

        ax[6][1].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_vdc)), 'r+')
        ax[6][1].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_idc_linear)), 'g*')
        if model == 'base_CPL' or model == 'base_PFC_CPL':
            ax[6][1].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_idc_nonlinear)), 'bx')

        ax[7][1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_vdc), 'r+')
        ax[7][1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_idc_linear), 'g*')
        if model == 'base_CPL' or model == 'base_PFC_CPL':
            ax[7][1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_idc_nonlinear), 'bx')
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1)

        # Get ac and dc voltages and currents
        t, vac, iac, vdc, idc = sim.get_voltage_and_current(model, interpolate_dt=interpolate_dt)
        t_sl = slice(np.argmin(np.abs(t - 1)), None)

        t = t[t_sl]
        vac = vac[t_sl]
        iac = iac[t_sl]
        vdc = vdc[t_sl]
        idc = idc[t_sl]

        N = len(t)
        f = np.fft.fftshift(np.fft.fftfreq(N, t[1] - t[0]))
        sp_vac = np.fft.fftshift(np.fft.fft(vac) / N)
        sp_iac = np.fft.fftshift(np.fft.fft(iac) / N)
        
        # if incremental is not None:
        #     print("incrementing")
        #     sp_mix -= incremental[0, :]
        #     sp_vdc -= incremental[1, :]
        #     sp_idc -= incremental[2, :]
        #     sp_vac -= incremental[3, :]
        #     sp_iac -= incremental[4, :]
        
        f_max = 200
        f_sl = slice(np.argmin(np.abs(f + f_max)), np.argmin(np.abs(f - f_max)))
        magCutoff = 1e-2

        sp_vac[np.abs(sp_vac) < magCutoff] = 1e-6
        sp_iac[np.abs(sp_iac) < magCutoff] = 1e-6

        ax.plot(f[f_sl], 20 * np.log10(np.abs(sp_vac[f_sl])), lw=2)
        ax.plot(f[f_sl], 20 * np.log10(np.abs(sp_iac[f_sl])), lw=2)

        xticks = [-120, -120 + gen_params['fp'], -60, -gen_params['fp'], 0, gen_params['fp'], 60, 120 - gen_params['fp'], 120]
        xtick_labels = ["$-2f_{L}$", "$-2f_{L} + f_P$", "$-f_{L}$", "$f_{P}$", 0,
                        "$f_{P}$", "$f_L$", "$2f_L - f_P$", "$2f_L$"]
        
        ax.set_xticks(xticks, xtick_labels)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude")
        plt.legend(["$v_{ac}[f]$", "$i_{ac}[f]$"])

        plt.tight_layout()

        fig, ax = plt.subplots(2, 1)

        f_max = 1000
        

        # Get ac and dc voltages and currents
        t, vac, iac, vdc, idc = sim.get_voltage_and_current(model, interpolate_dt=interpolate_dt)
        t_sl = slice(np.argmin(np.abs(t - 1)), None)

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
        
        # if incremental is not None:
        #     print("incrementing")
        #     sp_mix -= incremental[0, :]
        #     sp_vdc -= incremental[1, :]
        #     sp_idc -= incremental[2, :]
        #     sp_vac -= incremental[3, :]
        #     sp_iac -= incremental[4, :]
        

        f_sl = slice(np.argmin(np.abs(f + f_max)), np.argmin(np.abs(f - f_max)))
        magCutoff = 1e-3

        ax[0].plot(f[f_sl], 20 * np.log10(np.abs(sp_vdc[f_sl])))
        ax[0].plot(f[f_sl], 20 * np.log10(np.abs(sp_idc[f_sl])))

        sp_vdc[f_sl][np.abs(sp_vdc[f_sl]) < magCutoff] = 0
        sp_idc[f_sl][np.abs(sp_idc[f_sl]) < magCutoff] = 0
        ax[1].plot(f[f_sl], 180 / np.pi * np.angle(sp_vdc[f_sl]))
        ax[1].plot(f[f_sl], 180 / np.pi * np.angle(sp_idc[f_sl]))

        ax[0].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_vdc)), 'r+')
        ax[0].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_idc_linear)), 'g*')
        if model == 'base_CPL' or model == 'base_PFC_CPL':
            ax[0].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_idc_nonlinear)), 'bx')

        ax[1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_vdc), 'r+')
        ax[1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_idc_linear), 'g*')
        if model == 'base_CPL' or model == 'base_PFC_CPL':
            ax[1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_idc_nonlinear), 'bx')


        plt.tight_layout()
        plt.show()

    print_seperator('Simulation')
    # Spice Simulation
    freqs = np.linspace(1, 119, 60)
    freqs = np.floor(freqs)
    timesteps = np.ones_like(freqs) * 100e-6
    stoptimes = np.ones_like(freqs) * 2

    # Simulate in SPICE to get incremental impedance and incremental impedance for mirror frequency
    # sp = compute_incremental_impedance(model, gen_params, freqs, timesteps, stoptimes, debug=args.debug)
    sp, sp_mirror = compute_incremental_impedance2(model, gen_params, freqs, 10e-6)

    # Plot Results
    sl1 = slice(Nf // 2, None)
    x1 = w[sl1] / (2 * np.pi)

    sl2 = slice(None)
    x2 = freqs[sl2]

    y11 = 20 * np.log10(np.abs(Zdc[sl1]))
    y12 = 20 * np.log10(np.abs(Zac[sl1]))
    y13 = 20 * np.log10(np.abs(sp[sl2]))
    y14 = 20 * np.log10(np.abs(sp_mirror[sl2]))

    y21 = (np.angle(Zdc[sl1])) * 180 / np.pi
    y22 = (np.angle(Zac[sl1])) * 180 / np.pi
    y23 = (np.angle(sp[sl2])) * 180 / np.pi
    y24 = (np.angle(sp_mirror[sl2])) * 180 / np.pi

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.set_title("Magnitude")
    ax.semilogx(x1, y11, label='Zdc', alpha =0.5)
    ax.semilogx(x1, y12, label='Zac')
    ax.semilogx(x2, y13, 'r+')
    ax.semilogx(x2, y14, 'g*')
    ax.axvline(gen_params['fl'])
    ax.legend()
   
    ax = plt.subplot(2, 1, 2)
    ax.semilogx(x1, y21, alpha=0.5)
    ax.semilogx(x1, y22)
    ax.semilogx(x2, y23, 'r+')
    ax.semilogx(x2, y24, 'g*')
    ax.axvline(gen_params['fl'])
    
    plt.show()
    exit()
    # freqs = np.concatenate((np.flip(-freqs), freqs))
    # sp = np.concatenate((np.flip(np.conjugate(sp)), sp))
    # plt.figure()
    # ax = plt.subplot(2, 1, 1)
    # ax.set_title("Simulated")
    # ax.plot(freqs, np.real(sp), label='Real')
    # ax.plot(freqs, np.imag(sp), label='Imag')
    # ax.axvline(gen_params['fl'])
    # ax.legend()
   
    # sl = slice(np.argmin(np.abs(w+2*np.pi*freqs[-1])), np.argmin(np.abs(w-2*np.pi*freqs[-1])))
    # ax = plt.subplot(2, 1, 2)
    # ax.set_title("DC-Side Impedance")
    # ax.plot(w[sl] / (2 * np.pi), np.real(Zdc[sl]), label='Real')
    # ax.plot(w[sl] / (2 * np.pi), np.imag(Zdc[sl]), label='Imag')
    # ax.axvline(gen_params['fl'])
    
    # plt.tight_layout()

    # plt.figure()
    # ax = plt.subplot(2, 1, 1)
    # ax.set_title("Simulated AC-Side Admittance")
    # ax.plot(np.real(1/sp), np.imag(1/sp))
   
    # sl = slice(np.argmin(np.abs(w+2*np.pi*freqs[-1])), np.argmin(np.abs(w-2*np.pi*freqs[-1])))
    # ax = plt.subplot(2, 1, 2)
    # ax.set_title("Analytic DC-Side Admittance")
    # ax.plot(np.real(Ydc[sl]), np.imag(Ydc[sl]))
    
    # plt.tight_layout()
    # plt.show()