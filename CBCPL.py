import math
import numpy as np
from scipy import signal

# Controllable Bandwidth CPL
def getDCCurrentSpectrum(cutoff, admittance, frequencies, voltageSpectrum):
    V = voltageSpectrum
    Y = admittance

    freqs = frequencies
    w = 2 * np.pi * freqs # rad / s

    max_f = np.max(freqs)
    bias = V[freqs == 0]

    filter = signal.TransferFunction([1], [1/cutoff, 1]) # 1 / (RCs + 1)
    filterResponse = signal.freqresp(filter, w)[1]

    filterResponse[freqs == 0] = 0
    idc = Y * V * (1 - 2*filterResponse)

    # (n, m) = n > 2, 0 (here the distinction between n and m is inconsequential)
    #   basically, add harmonics generated by (n, m) -> (n + m)
    for i, f in enumerate(freqs):
        n = 2
        while np.abs(n*f) < max_f and f != 0: # for each frequency f, and n > 2, check if n*f is in freqs
            idx = np.where(freqs == n * f)[0]
            if len(idx) == 1: # if so, add the harmonic to this term
                idx = idx[0]
                adderand = Y * (-V[i] * filterResponse[i] / bias) ** (n - 1) * V[i] * (n - (n + 1) * filterResponse[i])
                idc[idx] += adderand 
            elif len(idx) > 1: # error
                print("FATAL ERROR: Input Vdc terms not combined")

            n += 1

    # note that this will double count combinations of frequencies!!!
    for i, f1 in enumerate(freqs):
        for j, f2 in enumerate(freqs):
            # if f1 * f2 > 0 and f1 != f2 and f1 != 0 and f2 != 0:
            if f1 != f2 and f1 != 0 and f2 != 0:
                m = 1
                n = 1
                while np.abs(n * f1 + m * f2) < max_f and (n + m) < 4:
                    idx = np.where(freqs == n * f1 + m * f2)[0]
                    if len(idx) > 1:
                        print("FATAL ERROR: Input Vdc terms not combined")
                    elif len(idx) == 1:
                        # print("hit: n=%d, f1=%d, m=%d, f2=%d, n*f1+m*f2=%d" % (n, f1, m, f2, n * f1 + m * f2))
                        idx = idx[0]
                        adderand =  bias * (n + m + 1) * V[i] * filterResponse[i] * V[j] * filterResponse[j] / bias**2
                        adderand -= V[i] * m * V[j] * filterResponse[j] / bias
                        adderand -= V[j] * n * V[i] * filterResponse[i] / bias
                        adderand -= np.conjugate(V[i]) * (n + m + 2) * (n + m + 1) / (m + 1) * (V[i] * filterResponse[i])**2 * (V[j] * filterResponse[j]) / bias**3
                        adderand -= np.conjugate(V[j]) * (n + m + 2) * (n + m + 1) / (n + 1) * (V[i] * filterResponse[i]) * (V[j] * filterResponse[j])**2 / bias**3

                        adderand *= (-1) ** (m + n) * Y
                        adderand *= (V[i] * filterResponse[i] / bias) ** (m - 1)
                        adderand *= (V[j] * filterResponse[j] / bias) ** (n - 1)
                        adderand *= math.factorial(n + m) / (math.factorial(m) * math.factorial(n))
                        adderand /= 2 # adjust for double counting

                        idc[idx] += adderand
                    m += 1
                    if np.abs(n * f1 + m * f2) >= max_f:
                        m = 1
                        n += 1

    return idc