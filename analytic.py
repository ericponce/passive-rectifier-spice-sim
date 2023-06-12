import numpy as np
from scipy import signal

##############################################################################
#                  		          Linear Loads		   	  		             #
##############################################################################

def LL_Idc(freqs, vals, Y_tf):
    return vals * signal.freqresp(Y_tf, 2*np.pi*freqs)[1]

# for the resistive case, it is pretty important that you have many harmonics!
def LL_Yac(Ydc_tf, w_stop, num_w, w_mix, order=3, retYdc=True):
    w_start_adj = -w_stop - (2 * order + 1) * w_mix
    w_stop_adj = w_stop + (2 * order + 1) * w_mix
    num_w_adj = int(num_w *(1 + (2 * order + 1) * w_mix / w_stop))
    num_w_adj += (1 if num_w_adj % 2 == 1 else 0)

    w = np.linspace(0, w_stop_adj, num_w_adj//2, endpoint=False)
    w = np.concatenate((np.flip(-w[1:]), w))

    Nf = len(w)
    _, Ydc = signal.freqresp(Ydc_tf, w)

    mix1_idx = np.argmin(np.abs(w - w_mix))
    roll = mix1_idx - Nf // 2

    Yac = np.zeros_like(Ydc)

    for i in range(order):
        # component from conduction angle modulation
        # sum Ydc(2nf_mix)*-1/(4n^2-1) from n=-inf to inf
        # first order = Ydc[0]; m = 0
        # second order = Ydc[0] - 1/3 (Ydc[2f_mix] + Ydc[-2f_mix]); m= +-1
        if i == 0:
            comp = (Ydc[Nf // 2]) # why does putting abs here this help pfc
        else:
            pidx = Nf // 2 + 2 * i * roll
            nidx = Nf // 2 - 2 * i * roll
            comp = -1 / (4 * i**2 - 1) * (Ydc[pidx] + Ydc[nidx])


        # components from mixing signal
        # sum 1/(2i+1)^2(Ydc[f - (2m+1)f_mix] + Ydc[f + (2m+1)f_mix]) from m=-inf to inf
        # first order (m = 0, -1) = Ydc[f-f_mix] + Ydc[f+f_mix]
        # second order (m = -2, 1) = f.o. + 1/3^2(Ydc[f+3f_mix] + Ydc[f-3f_mix])
        plus = np.roll(Ydc, -(2*i + 1) * roll) / (2*i + 1)**2
        minus = np.roll(Ydc, (2*i + 1) * roll) / (2*i + 1)**2 ###### wrong i #######

        Yac += comp + plus + minus

    Yac *= (4 / np.pi**2)


    # cleanup
    w_stop1_idx = np.argmin(np.abs(w - (-w_stop)))
    w_stop2_idx = np.argmin(np.abs(w - w_stop))

    sl = slice(w_stop1_idx, w_stop2_idx)
    w = w[sl]
    Ydc = Ydc[sl]
    Yac = Yac[sl]

    if retYdc:
        return w, Ydc, Yac
    else:
        return w, Ydc

##############################################################################
#                            Constant Power Loads		     		         #
##############################################################################

"""
power - Power Level (W)
bias - Bias Voltage (V)
level - Pertubation Voltage (V)
"""
def CPL_FourierTerm(n, power, bias, level):
    A = np.sqrt(bias**2 - level**2)
    res = power * level**n
    res /= A * (bias + A)**n
    res *= (-1)**n
    res *= np.where(n > 0, 2*np.ones_like(n), 1*np.ones_like(n))
    return res

# Assumes collected terms
# [TODO] Can't use small signal assumption because some of the vdc terms are not small signals!!
# [TODO] For very large inputs (like harmonic terms in Vdc), the solution is also not reasonable
def CPL_Idc(freqs, Vdc, power, bias, smallsignal=True):
    # Double the input terms (except DC)
    vals = 2 * Vdc
    vals[freqs == 0] /= 2

    Y = power / bias**2

    if smallsignal:
        # n = 0 term is Y * V_T = Y * vals[f=0*i]
        # n = 1 term is Y * V_T * 2 / (2 * V_T) * vals[i] = Y * vals[i]
        # n > 2 terms will be added later
        idc = -vals * Y
    else:
        idc = np.zeros_like(Vdc)
        idc[freqs == 0] = power / np.sqrt(bias**2 - np.sum(vals[freqs != 0]**2))
        idc[freqs != 0] = 2 * power / np.sqrt(bias**2 - vals[freqs != 0]**2) 
        idc[freqs != 0] /= -bias/vals[freqs != 0] - np.sqrt((bias/vals[freqs != 0])**2 - 1)
        # print((-bias/vals[freqs != 0]-np.sqrt((bias/vals[freqs != 0])**2 - 1)))

        # print(np.sqrt((bias/vals[freqs != 0])**2 - 1))

    # print(bias) 
    # print(vals)
    print(idc)

    max_f = np.max(freqs)

    for i, f in enumerate(freqs):
        n = 2
        while np.abs(n*f) < max_f and f != 0:
            # if freqs contains n*f, add some more current
            # print(np.where(freqs == n * f), n, f, n*f)
            # print(freqs)
            # print(len(np.where(freqs == n * f)))
            idx = np.where(freqs == n * f)[0]
            if len(idx) > 0:
                idx = idx[0]
                if smallsignal:
                    adderand = 2 * Y * bias * (-vals[i] / (2 * bias)) ** n
                else:
                    adderand = 2 * power / np.sqrt(bias**2 - vals[i]**2) * (-bias/vals[i]-np.sqrt((bias/vals[i])**2 - 1)) ** -n
                print("hit, (f=%d)*(n=%d)=%f: i(v=%f,n=%d)=%f+%f" % (f, n, freqs[idx], vals[i], n, idc[idx],  adderand))
                idc[idx] += adderand

            n += 1

    # halve the output values (except DC)
    idc[freqs != 0] /= 2
    return idc

# Controllable Bandwidth CPL
def CBCPL_Idc(cutoff, Ydc_tf, freqs, Vdc, power, bias):
    max_f = np.max(freqs)
    bias = Vdc[freqs == 0]
    Y = power / bias**2

    filter = signal.TransferFunction([1], [1/cutoff, 1])
    filterResponse = signal.freqresp(filter, freqs*2*np.pi)[1]
    filterResponse[freqs == 0] = 0

    # n = 0, m = 0 and n = 1 terms
    # vals = 2 * Vdc # Double the input terms (except DC)
    # vals[freqs == 0] /= 2
    idc = Y * Vdc * (1 - 2*filterResponse)

    # n > 2 terms, controllable bandwidth
    # vals = 2 * filteredVdc
    for i, f in enumerate(freqs):
        n = 2
        while np.abs(n*f) < max_f and f != 0:
            idx = np.where(freqs == n * f)[0]
            if len(idx) > 0:
                idx = idx[0]
                adderand = Y * (-Vdc[i] * filterResponse[i] / bias) ** (n - 1) * Vdc[i] * (n - (n + 1) * filterResponse[i])
                # if np.abs(adderand) > 1e-6:
                    # print("hit, (f=%d)*(n=%d)=%f: i(v=%f,n=%d)=%f+%f" % (f, n, freqs[idx], vals[i], n, idc[idx],  adderand))
                    # print(np.abs(vals[i]), 2 * Vdc[i])
                # idc[idx] += 2**(n-1) * adderand # correction from this being a real signal?
                idc[idx] += adderand # correction from this being a real signal?

            n += 1

    # halve the output values (except DC)
    # idc[freqs != 0] /= 2
    return idc

# def CBCPL_SimpleYac(Ydc_tf, w_stop, num_w, w_mix, order=3, retYdc=True):
    w_start_adj = -w_stop - (2 * order + 1) * w_mix
    w_stop_adj = w_stop + (2 * order + 1) * w_mix
    num_w_adj = int(num_w *(1 + (2 * order + 1) * w_mix / w_stop))
    num_w_adj += (1 if num_w_adj % 2 == 1 else 0)

    w = np.linspace(0, w_stop_adj, num_w_adj//2, endpoint=False)
    w = np.concatenate((np.flip(-w[1:]), w))

    Nf = len(w)
    _, Ydc = signal.freqresp(Ydc_tf, w)

    mix1_idx = np.argmin(np.abs(w - w_mix))
    roll = mix1_idx - Nf // 2

    Yac = np.zeros_like(Ydc)

    for i in range(order):
        # component from conduction angle modulation
        # sum Ydc(2nf_mix)*-1/(4n^2-1) from n=-inf to inf
        # first order = Ydc[0]; m = 0
        # second order = Ydc[0] - 1/3 (Ydc[2f_mix] + Ydc[-2f_mix]); m= +-1
        if i == 0:
            comp = np.abs(Ydc[Nf // 2]) # DC term is not negative
        else: # [TODO] figure out what to do here
            pidx = Nf // 2 + 2 * i * roll
            nidx = Nf // 2 - 2 * i * roll
            comp = -1 / (4 * i**2 - 1) * (Ydc[pidx] + Ydc[nidx])


        # components from mixing signal
        # sum 1/(2i+1)^2(Ydc[f - (2m+1)f_mix] + Ydc[f + (2m+1)f_mix]) from m=-inf to inf
        # first order (m = 0, -1) = Ydc[f-f_mix] + Ydc[f+f_mix]
        # second order (m = -2, 1) = f.o. + 1/3^2(Ydc[f+3f_mix] + Ydc[f-3f_mix])
        plus = np.roll(Ydc, -(2*i + 1) * roll) / (2*i + 1)**2
        minus = np.roll(Ydc, (2*i + 1) * roll) / (2*i + 1)**2 

        Yac += comp + plus + minus

    Yac *= (4 / np.pi**2)


    # cleanup
    w_stop1_idx = np.argmin(np.abs(w - (-w_stop)))
    w_stop2_idx = np.argmin(np.abs(w - w_stop))

    sl = slice(w_stop1_idx, w_stop2_idx)
    w = w[sl]
    Ydc = Ydc[sl]
    Yac = Yac[sl]

    if retYdc:
        return w, Ydc, Yac
    else:
        return w, Ydc
    
def CBCPL_SimpleYac2(Ydc_tf, w_stop, num_w, w_mix, order=3, retYdc=True, debug=False):
    w_start_adj = -w_stop - (2 * order + 1) * w_mix
    w_stop_adj = w_stop + (2 * order + 1) * w_mix
    num_w_adj = int(num_w *(1 + (2 * order + 1) * w_mix / w_stop))
    num_w_adj += (1 if num_w_adj % 2 == 1 else 0)

    w = np.linspace(0, w_stop_adj, num_w_adj//2, endpoint=False)
    w = np.concatenate((np.flip(-w[1:]), w))

    Nf = len(w)

    mix1_idx = np.argmin(np.abs(w - w_mix))
    roll = mix1_idx - Nf // 2

    _, Ydc = signal.freqresp(Ydc_tf, w)

    # Small signal pertubations will effectively cause zero current change at harmonics of grid voltage
    # so set Y[n * 120] = 0
    Ydc_ss = np.copy(Ydc)
    n = np.arange(-int(w_stop_adj / w_mix / 2), int(w_stop_adj / w_mix / 2) + 1)
    # Ydc_ss[Nf//2 + 2*n*roll] = 0
    Ydc_ss[Nf//2] = 0
    
    Yac = np.zeros_like(Ydc)

    if debug:
        import matplotlib.pyplot as plt
        x = w / (2 * np.pi)
        fig, ax = plt.subplots(2, 1)
        # ax[0].plot(x, np.abs(Ydc))
        # ax[1].plot(x, (np.angle(Ydc)))

    for i in range(order):
        # component from conduction angle modulation
        # sum Ydc(2nf_mix)*-1/(4n^2-1) from n=-inf to inf
        # first order = Ydc[0]; m = 0
        # second order = Ydc[0] - 1/3 (Ydc[2f_mix] + Ydc[-2f_mix]); m= +-1
        
        if i == 0:
            comp = np.abs(Ydc[Nf // 2])
        else:
            pidx = Nf // 2 + 2 * i * roll
            nidx = Nf // 2 - 2 * i * roll
            comp = -1 / (4 * i**2 - 1) * (Ydc[pidx] + Ydc[nidx])
        # comp = 0 # end up begin zero then Ydc[2mf] = constant

        # components from mixing signal
        # sum 1/(2i+1)^2(Ydc[f - (2m+1)f_mix] + Ydc[f + (2m+1)f_mix]) from m=-inf to inf
        # first order (m = 0, -1) = Ydc[f-f_mix] + Ydc[f+f_mix]
        # second order (m = -2, 1) = f.o. + 1/3^2(Ydc[f+3f_mix] + Ydc[f-3f_mix])
        plus = np.roll(Ydc_ss, -(2*i + 1) * roll) / (2*i + 1)**2
        minus = np.roll(Ydc_ss, (2*i + 1) * roll) / (2*i + 1)**2 

        if debug:
            if i < 2:
                ax[0].axhline(np.abs(comp), color='black', ls=':', alpha=0.5)
                ax[0].plot(x, np.abs(plus))
                ax[0].plot(x, np.abs(minus))

                ax[1].axhline(np.angle(comp), color='black', ls=':', alpha=0.5)
                ax[1].plot(x, (np.angle(plus)))
                ax[1].plot(x, (np.angle(minus)))

        Yac += comp + plus + minus

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

    if retYdc:
        return w, Ydc, Yac
    else:
        return w, Ydc

# Controllable Bandwidth CPL
# def CBCPL_Idc2(cutoff, Ydc_tf, freqs, Vdc, power, bias, smallsignal=True):
    Ydc = np.abs(signal.freqresp(Ydc_tf, np.array([0]))[1])

    Vdc = Vdc * (1 + 0j)

    filter = signal.TransferFunction([1], [1/cutoff, 1])
    filteredVdc = Vdc * signal.freqresp(filter, 2*np.pi*freqs)[1]
    filteredVdc[freqs == 0] = 0 # elimnate dc term

    # w, mag, phase = signal.bode(filter)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.semilogx(w, mag)    # Bode magnitude plot
    # plt.figure()
    # plt.semilogx(w, phase)  # Bode phase plot
    # plt.show()

    idc = Ydc * (Vdc - 2*filteredVdc)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(Vdc))
    plt.plot(freqs, np.abs(filteredVdc), 'r+')

    ax = plt.subplot(2, 1, 2)
    ax.plot(freqs, np.angle(Vdc))
    ax.plot(freqs, np.angle(filteredVdc), 'r+')

    plt.show()

    return idc

# Controllable Bandwidth CPL
# def CBCPL_Idc3(cutoff, Ydc_tf, freqs, Vdc, mixingFrequency):
    Ydc = np.abs(signal.freqresp(Ydc_tf, np.array([0]))[1]) # equivalent admittance

    Vdc = Vdc * (1 + 0j)

    filter = signal.TransferFunction([1], [1/cutoff, 1])
    filterResponse = signal.freqresp(filter, freqs*2*np.pi)[1]

    filterResponse[freqs == 0] = 0 # elimnate dc term
    

    # w, mag, phase = signal.bode(filter)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(freqs, np.abs(1 - 2*filterResponse))
    plt.subplot(2, 1, 2)
    plt.stem(freqs, 180/np.pi*np.angle(1 - 2*filterResponse))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.stem(freqs, np.abs(filterResponse))
    plt.subplot(2, 1, 2)
    plt.stem(freqs, 180/np.pi*np.angle(filterResponse))

    
    idc = Ydc * Vdc * (1 - 2*filterResponse)

    import matplotlib.pyplot as plt
    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(freqs, np.abs(Vdc))
    # plt.plot(freqs, np.abs(vdcFilter), 'r+')

    # ax = plt.subplot(2, 1, 2)
    # ax.plot(freqs, np.angle(Vdc))
    # ax.plot(freqs, np.angle(filteredVdc), 'r+')

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(Ydc * Vdc))
    plt.plot(freqs, np.abs(-Ydc * Vdc))
    plt.plot(freqs, np.abs(idc))

    ax = plt.subplot(2, 1, 2)
    plt.plot(freqs, np.angle(Ydc * Vdc))
    plt.plot(freqs, np.angle(-Ydc * Vdc))
    plt.plot(freqs, np.angle(idc))

    plt.show()

    return idc
##############################################################################
#                     Constant Conduction Passive Rectifier		  	       	 #
##############################################################################
def CCPR_Vdc(lineVoltage, lineFrequency, 
             pertubationVoltage, pertubationFrequency,
             maximumMixingOrder, positiveFrequenciesOnly=True,
             combineLikeTerms=True, includeSecondOrder=True):

    fL = lineFrequency
    fP = pertubationFrequency

    vL = lineVoltage
    vP = pertubationVoltage

    if positiveFrequenciesOnly:
        m = np.arange(maximumMixingOrder+1)
    else:
        m = np.arange(-maximumMixingOrder, maximumMixingOrder+1)

    freqs = np.array([])
    vals = np.array([])

    # First Term: M[(2m+1) lineFrequency], Vac[+-lineFrequency]
    freqs = np.append(freqs, (2 * m + 1) * fL + fL)
    freqs = np.append(freqs, (2 * m + 1) * fL - fL)
    vals = np.append(vals, np.tile((-1)**np.abs(m) * vL/(np.pi*(2*m + 1)), 2))
    
    # Second Term: M[(2m+1) lineFrequency], Vac[+-pertubationFrequency]
    freqs = np.append(freqs, (2 * m + 1) * fL + fP)
    freqs = np.append(freqs, (2 * m + 1) * fL - fP)
    vals = np.append(vals, np.tile((-1)**np.abs(m) * vP/(np.pi*(2*m + 1)), 2))

    # Third Term: M[2m*lineFrequency +-pertubationFrequency], Vac[+-fL]
    freqs = np.append(freqs, 2*m*fL + fP + fL)
    freqs = np.append(freqs, 2*m*fL + fP - fL)
    freqs = np.append(freqs, 2*m*fL - fP + fL)
    freqs = np.append(freqs, 2*m*fL - fP - fL)
    vals = np.append(vals, np.tile((-1)**np.abs(m) * vP/(2 * np.pi), 4))

    if includeSecondOrder:
        # Fourth Term: M[2m*lineFrequency +-pertubationFrequency], Vac[+-fP]
        freqs = np.append(freqs, 2*m*fL)
        vals = np.append(vals, (-1)**np.abs(m) * vP**2/(2 * np.pi * vL))

        # Fifth Term: M[2m*lineFrequency +-pertubationFrequency], Vac[+-fP]
        freqs = np.append(freqs, 2*m*fL + 2 * fP)
        freqs = np.append(freqs, 2*m*fL - 2 * fP)
        vals = np.append(vals, np.tile((-1)**np.abs(m) * vP**2/(2 * np.pi * vL), 2)) # this term is off by ~2 in resistive case

    if combineLikeTerms:
        freqs, idx = np.unique(freqs, return_inverse=True)
        vals = np.bincount(idx, weights=vals)

    return freqs, vals

if __name__ == "__main__":
    CCPR_Vdc(120, 60, 1, 7, 3, True)