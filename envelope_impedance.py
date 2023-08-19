
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

def compute_incremental_envelope_impedance(model, gen_params, freqs, timestep, fv=get_vac, fi=get_iac):
    # General Configuration
    gen_params['timestep'] = timestep
    gen_params['timestart'] = 0
    gen_params['timestop'] = 2

    # Allocate Array for final result
    sp = np.zeros(len(freqs), dtype=np.cdouble)

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
        # omega = 2 * np.pi * f
        fLine = gen_params['fl']
        omegaLine = 2 * np.pi * fLine
        omegaPlus = 2 * np.pi * (fLine + f)
        omegaMinus = 2 * np.pi * (fLine - f)

        tUnperturbed = ltrUnperturbed.get_trace('time').get_time_axis()
        tPerturbed = ltr.get_trace('time').get_time_axis()

        slU = slice(np.argmin(np.abs(tUnperturbed-1)) - 1, None)
        slP = slice(np.argmin(np.abs(tPerturbed-1)) - 1, None)

        tU = tUnperturbed[slU]
        tP = tPerturbed[slP]

        # expU = np.exp(-1j * omega * tU)
        # expP = np.exp(-1j * omega * tP)
        expULine = np.exp(-1j * omegaLine * tU)
        expUPlus = np.exp(-1j * omegaPlus * tU)
        expUMinus = np.exp(-1j * omegaMinus * tU)

        expPLine = np.exp(-1j * omegaLine * tP)
        expPPlus = np.exp(-1j * omegaPlus * tP)
        expPMinus = np.exp(-1j * omegaMinus * tP)

        # spV = np.trapz(fv(ltr)[slP] * expP, tP) - np.trapz(fv(ltrUnperturbed)[slU] * expU, tU)
        # spI = np.trapz(fi(ltr)[slP] * expP, tP) - np.trapz(fi(ltrUnperturbed)[slU] * expU, tU)

        spV  = np.trapz(fv(ltr)[slP] * expPLine, tP)  - np.trapz(fv(ltrUnperturbed)[slU] * expULine, tU)
        spV += np.trapz(fv(ltr)[slP] * expPPlus, tP)  - np.trapz(fv(ltrUnperturbed)[slU] * expUPlus, tU)
        spV += np.trapz(fv(ltr)[slP] * expPMinus, tP) - np.trapz(fv(ltrUnperturbed)[slU] * expUMinus, tU)

        spI  = np.trapz(fi(ltr)[slP] * expPLine, tP) - np.trapz(fi(ltrUnperturbed)[slU] * expULine, tU)
        spI += np.trapz(fi(ltr)[slP] * expPPlus, tP) - np.trapz(fi(ltrUnperturbed)[slU] * expUPlus, tU)
        spI += np.trapz(fi(ltr)[slP] * expPMinus, tP) - np.trapz(fi(ltrUnperturbed)[slU] * expUMinus, tU)
        sp[i] = spV / spI

        print("\t Done. |Z(%f)|=%f, <Z(%f)=%f" % (f, np.abs(sp[i]), f, np.angle(sp[i])*180/np.pi))

    return sp

# for the resistive case, it is pretty important that you have many harmonics!
def compute_envelope_admittance(Ydc_tf, w_stop, num_w, w_mix, order=3, debug=False):
    w_stop_adj = w_stop + (2 * order + 1) * w_mix
    num_w_adj = int(num_w *(1 + (2 * order + 1) * w_mix / w_stop))
    num_w_adj += (1 if num_w_adj % 2 == 1 else 0)

    w = np.linspace(0, w_stop_adj, num_w_adj//2, endpoint=False)
    w = np.concatenate((np.flip(-w[1:]), w))

    Nf = len(w)
    _, Ydc = signal.freqresp(Ydc_tf, w)

    mix1_idx = np.argmin(np.abs(w - 2 * w_mix))
    roll = mix1_idx - Nf // 2

    if debug:
        x = w / (2 * np.pi)
        fig, ax = plt.subplots(2, 1)
        ax[0].set_xscale('symlog')
        ax[1].set_xscale('symlog')
        ax[0].plot(x, np.abs(Ydc))
        ax[1].plot(x, np.angle(Ydc))

    Yac = np.zeros_like(Ydc)

    orders = np.arange(order + 1)
    orders = np.concatenate((np.flip(-orders[1:]), orders))
    print(orders)
    for i in orders:
        shift = np.roll(Ydc, -2 * i * roll) / (2 * i+1) / (1 - 4 * i**2)
        Yac += shift

        if debug:
            ax[0].plot(x, np.abs(shift))
            ax[1].plot(x, np.angle(shift))

    Yac *= (8 / np.pi**2)

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
        print("Not yet implemented")
        exit()
        print("Passive full-bridge recitifier with resistive load")
        model = 'base_R'

        gen_params['Rdc'] = args.R

        Ydc_tf = signal.lti([1/gen_params['Rdc']], [1])
    elif model == 'rlc':
        print("RLC Model")
        model = 'base_RLC_Envelope'

        gen_params['Ldc'] = args.L
        gen_params['Cdc'] = args.C
        gen_params['Rdc'] = args.R

        gen_params['timestep'] = 1e-6

        Ydc_tf = signal.lti([0, args.C*args.R, 1], 
                            [args.L*args.C*args.R, args.L, args.R])

    elif model == 'pfc_cpl':
        print("Not yet implemented")
        exit()
        print("Passive PFC CPL Model")
        model = 'base_PFC_CPL'

        w_cpl = args.fcpl * 2 * np.pi

        gen_params['Ydc'] = args.Ydc
        gen_params['Cdc'] = args.Cdc
        gen_params['wCPL'] = w_cpl

        Ydc_tf = signal.lti([args.Cdc, args.Cdc*w_cpl + args.Ydc, -args.Ydc * w_cpl], [0, 1, w_cpl])

    # # Analytic Estimation
    print_seperator('Analytical Estimation')
    f_stop = 2500 # will be adjusted to remove edge effects
    Nf = 50000

    w_stop = 2 * np.pi * f_stop
    w_mix = 2 * np.pi * gen_params['fl']
    # if model == 'base_PFC_CPL':
    #     # w, Ydc, Yac = analytic.CBCPL_SimpleYac2(Ydc_tf, w_stop, Nf, w_mix, order=100, debug=args.debug)
    #     # w, Ydc, Yac = analytic.CBCPL_SimpleYac(Ydc_tf, w_stop, Nf, w_mix, order=1, debug=args.debug)
    #     w, Ydc, Yac = analytic.CBCPL_ApproximateYac(args.Ydc, args.fcpl, f_stop, Nf, gen_params['fl'])

    # else:
    w, Ydc, Yac = compute_envelope_admittance(Ydc_tf, w_stop, Nf, w_mix, order=1, debug=args.debug)
    Zdc, Zac = 1 / Ydc, 1 / Yac
    Nf = len(w)

    # Plot Results
    # sl1 = slice(Nf // 2, None)
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

    print_seperator('Mixing Signal Analysis')
    Vp = gen_params['Vp']
    interpolate_dt = 1e-6

    # Simulate in SPICE with a pertubation
    gen_params['fp'] = 5
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
        print("Not yet implemented")
        exit()
        bias = 2 * gen_params['Vg']/ np.pi
        power = bias**2 * gen_params['Ydc']
        cutoff = gen_params['wCPL']
        an_sp_idc_nonlinear = analytic.CBCPL_Idc2(cutoff, an_f_vdc, an_sp_vdc, power, bias)


    

    # slice
    sl = slice(np.argmin(np.abs(an_f_vdc)), np.argmin(np.abs(an_f_vdc - 1000)))
    an_f_vdc = an_f_vdc[sl]
    an_sp_vdc = an_sp_vdc[sl]
    an_sp_idc_linear = an_sp_idc_linear[sl]

    if model == 'base_PFC_CPL' or model == 'base_CPL':
        an_sp_idc_nonlinear = an_sp_idc_nonlinear[sl]

    # simulate
    sim.write_param_file(model+'.gen', gen_params)
    sim.execute_spice(model)

    fig, ax = plt.subplots(8, 2)
    plot_utils.plot_rectifier_ports(fig, ax, model, interpolate_dt, 1000)

    # ax[6][1].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_vdc)), 'r+')
    # ax[6][1].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_idc_linear)), 'g*')
    # if model == 'base_CPL' or model == 'base_PFC_CPL':
    #     ax[6][1].plot(an_f_vdc, 20*np.log10(np.abs(an_sp_idc_nonlinear)), 'bx')

    # ax[7][1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_vdc), 'r+')
    # ax[7][1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_idc_linear), 'g*')
    # if model == 'base_CPL' or model == 'base_PFC_CPL':
    #     ax[7][1].plot(an_f_vdc, 180 / np.pi * np.angle(an_sp_idc_nonlinear), 'bx')
    plt.tight_layout()
    plt.show()

    print_seperator('Simulation')
    # Spice Simulation
    freqs = np.linspace(1, 59, 30)
    freqs = np.floor(freqs)
    timesteps = np.ones_like(freqs) * 100e-6
    stoptimes = np.ones_like(freqs) * 2

    # Simulate in SPICE to get incremental impedance
    # sp = compute_incremental_impedance(model, gen_params, freqs, timesteps, stoptimes, debug=args.debug)
    sp = compute_incremental_envelope_impedance(model, gen_params, freqs, 100e-6)

    # Plot Results
    # sl1 = slice(Nf // 2, None)
    # x1 = w[sl1] / (2 * np.pi)

    sl2 = slice(None)
    x2 = freqs[sl2]

    # y11 = 20 * np.log10(np.abs(Zdc[sl1]))
    # y12 = 20 * np.log10(np.abs(Zac[sl1]))
    y13 = 20 * np.log10(np.abs(sp[sl2]))

    # y21 = (np.angle(Zdc[sl1])) * 180 / np.pi
    # y22 = (np.angle(Zac[sl1])) * 180 / np.pi
    y23 = (np.angle(sp[sl2])) * 180 / np.pi

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    ax.set_title("Magnitude")
    # ax.semilogx(x1, y11, label='Zdc')
    # ax.semilogx(x1, y12, label='Zac')
    ax.semilogx(x2, y13, 'r+')
    ax.axvline(gen_params['fl'])
    ax.legend()
   
    ax = plt.subplot(2, 1, 2)
    ax.set_title("Phase")
    # ax.semilogx(x1, y21)
    # ax.semilogx(x1, y22)
    ax.semilogx(x2, y23, 'r+')
    ax.axvline(gen_params['fl'])
    
    plt.show()
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