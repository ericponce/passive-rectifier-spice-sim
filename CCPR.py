import numpy as np

class CCPR():
    def __init__(self, lineVoltage, lineFrequency, perturbationVoltage, maximumMixingOrder):
        self.vL = lineVoltage
        self.fL = lineFrequency

        self.vP = perturbationVoltage
        self.order = maximumMixingOrder

    def getMixingSpectrum(self, perturbationFrequency):
        fL = self.fL
        vL = self.vL
        fP = perturbationFrequency
        vP = self.vP

        m = np.arange(-self.order, self.order + 1)

        freqs = np.array([])
        vals = np.array([], dtype=np.complex128)

        # line rectification harmonics
        freqs = np.append(freqs, (2 * m + 1) * fL)
        vals = np.append(vals, (-1.0)**m * 2 / (2 * m + 1) / np.pi)

        if perturbationFrequency != 0:
            # perturbation modulation harmonics
            freqs = np.append(freqs, 2 * m * fL + fP)
            vals = np.append(vals, (-1.0)**m * vP / vL / np.pi)
            freqs = np.append(freqs, 2 * m * fL - fP)
            vals = np.append(vals, (-1.0)**m * np.conj(vP) / vL / np.pi)

        return freqs, vals

    # When perturbed by an envelope, there is no window modulation harmonics
    def getEnvelopeMixingSpectrum(self):
        fL = self.fL

        m = np.arange(-self.order, self.order + 1)

        freqs = np.array([])
        vals = np.array([], dtype=np.complex128)

        # line rectification harmonics
        freqs = np.append(freqs, (2 * m + 1) * fL)
        vals = np.append(vals, (-1.0)**m * 2 / (2 * m + 1) / np.pi)

        return freqs, vals
    
    def getDCVoltageSpectrum(self, perturbationFrequency):
        fL = self.fL
        vL = self.vL
        fP = perturbationFrequency
        vP = self.vP

        m = np.arange(-self.order, self.order + 1)

        freqs = np.array([])
        vals = np.array([], dtype=np.complex128)

        # First Term: M[(2m+1) lineFrequency], Vac[+-lineFrequency]
        freqs = np.append(freqs, 2 * m * fL)
        vals = np.append(vals, (-1.0)**m * 2 * vL/ (1 - 4*m**2) / np.pi)

        if perturbationFrequency != 0:
            # Second Term: M[(2m+1) lineFrequency], Vac[+-pertubationFrequency]
            freqs = np.append(freqs, (2 * m + 1) * fL + fP)
            vals = np.append(vals, (-1.0)**m * vP / (2*m + 1) / np.pi)
            freqs = np.append(freqs, (2 * m + 1) * fL - fP)
            vals = np.append(vals, (-1.0)**m * np.conj(vP) / (2*m + 1) / np.pi)

        return freqs, vals
    
    def getEnvelopeDCVoltageSpectrum(self, perturbationFrequency):
        fL = self.fL
        vL = self.vL
        fP = perturbationFrequency
        vP = self.vP

        m = np.arange(-self.order, self.order + 1)

        freqs = np.array([])
        vals = np.array([], dtype=np.complex128)

        # First Term: M[(2m+1) lineFrequency], Vac[+-lineFrequency]
        freqs = np.append(freqs, 2 * m * fL)
        vals = np.append(vals, (-1.0)**m * 2 * vL/ (1 - 4*m**2) / np.pi)
        
        if perturbationFrequency != 0:
            # Second Term: M[(2m+1) lineFrequency], Vac[+-(lineFrequency +- pertubationFrequency)]
            freqs = np.append(freqs, 2 * m * fL + fP)
            vals = np.append(vals, (-1.0)**m * 2 * vP/ (1 - 4*m**2) / np.pi)
            freqs = np.append(freqs, 2 * m * fL - fP)
            vals = np.append(vals, (-1.0)**m * 2 * np.conj(vP) / (1 - 4*m**2) / np.pi)

        return freqs, vals

    def getACCurrentSpectrum(self, perturbationFrequency, dcFrequencies, dcCurrents):
        mixFrequencies, mixSpectrum = self.getMixingSpectrum(perturbationFrequency)

        # as of right now this only looks at the positive perturbation frequency
        perturbationCurrent = 0
        mirrorCurrent = 0

        if (perturbationFrequency == 0):
            dcCurrent = 0
            lineCurrent = 0
            doubleLineCurrent = 0

        for i, mf in enumerate(mixFrequencies):
            if perturbationFrequency == 0: # unperturbed case
                match_idx = np.where(mf == 0 - dcFrequencies)[0]
                if len(match_idx) > 1:
                    print("FATAL ERROR: Multiple indexes found")
                    exit()
                elif len(match_idx) == 1:
                    dcCurrent += mixSpectrum[i] * dcCurrents[match_idx]

                match_idx = np.where(mf == self.fL - dcFrequencies)[0]
                if len(match_idx) > 1:
                    print("FATAL ERROR: Multiple indexes found")
                    exit()
                elif len(match_idx) == 1:
                    lineCurrent += mixSpectrum[i] * dcCurrents[match_idx]

                # Find similar for fundamental-mirrored perturbation frequency
                match_idx = np.where(mf == 2 * self.fL - dcFrequencies)[0]
                if len(match_idx) > 1:
                    print("FATAL ERROR: Multiple indexes found")
                    exit()
                elif len(match_idx) == 1:
                    doubleLineCurrent += mixSpectrum[i] * dcCurrents[match_idx]
            else:
                # Find terms that add into perturbation frequency
                match_idx = np.where(mf == perturbationFrequency - dcFrequencies)[0]
                if len(match_idx) > 1:
                    print("FATAL ERROR: Multiple indexes found")
                    exit()
                elif len(match_idx) == 1:
                    perturbationCurrent += mixSpectrum[i] * dcCurrents[match_idx]

                # Find similar for fundamental-mirrored perturbation frequency
                match_idx = np.where(mf == (2 * self.fL - perturbationFrequency) - dcFrequencies)[0]
                if len(match_idx) > 1:
                    print("FATAL ERROR: Multiple indexes found")
                    exit()
                elif len(match_idx) == 1:
                    mirrorCurrent += mixSpectrum[i] * dcCurrents[match_idx]
        
        if perturbationFrequency == 0:
            return dcCurrent, lineCurrent, doubleLineCurrent
        else:
            return perturbationCurrent, mirrorCurrent # note that in unpertrubed case we will return the line and 2*line frequency
    
    def getEnvelopeACCurrentSpectrum(self, perturbationFrequency, dcFrequencies, dcCurrents):
        mixFrequencies, mixSpectrum = self.getMixingSpectrum(perturbationFrequency)

        # as of right now this only looks at the positive perturbation frequency
        lineCurrent = 0
        positiveSidebandCurrent = 0
        negativeSidebandCurrent = 0

        for i, mf in enumerate(mixFrequencies):
            # Find terms that add into the line frequency
            matchFrequency = self.fL
            match_idx = np.where(mf == matchFrequency - dcFrequencies)[0]
            if len(match_idx) > 1:
                print("FATAL ERROR: Multiple indexes found")
                exit()
            elif len(match_idx) == 1:
                lineCurrent += mixSpectrum[i] * dcCurrents[match_idx]

            # Find terms that add into postive sideband frequency
            matchFrequency = self.fL + perturbationFrequency
            match_idx = np.where(mf == matchFrequency - dcFrequencies)[0]
            if len(match_idx) > 1:
                print("FATAL ERROR: Multiple indexes found")
                exit()
            elif len(match_idx) == 1:
                positiveSidebandCurrent += mixSpectrum[i] * dcCurrents[match_idx]

            # Find terms that add into negative sideband frequency
            matchFrequency = self.fL - perturbationFrequency
            match_idx = np.where(mf == matchFrequency - dcFrequencies)[0]
            if len(match_idx) > 1:
                print("FATAL ERROR: Multiple indexes found")
                exit()
            elif len(match_idx) == 1:
                negativeSidebandCurrent += mixSpectrum[i] * dcCurrents[match_idx]
  
        return lineCurrent, positiveSidebandCurrent, negativeSidebandCurrent

if __name__ == "__main__":
    from scipy import signal
    import matplotlib.pyplot as plt

    import sim
    import plot_utils

    import argparse
    parser = argparse.ArgumentParser(description="Simulate Passive Rectifier Circuits")
    parser.add_argument('model', default='r', help='Model to test')
    args = parser.parse_args()

    lineVoltage = 100
    lineFrequency = 60
    perturbationVoltage = 1
    perturbationFrequency = 10
    interpolate_dt = 1e-6

    ccpr = CCPR(lineVoltage, lineFrequency, perturbationVoltage, 10)

    # base parameters for all tests
    base_params = {
            'fl': lineFrequency,
            'fp': perturbationFrequency,
            'transtop':2,
            'transtart':0,
            'timestep':50e-6,
            'Vg': lineVoltage,
            'Vp': perturbationVoltage
    }

    def testModel(model, gen_params, Idc):
        import CBCPL
        # Compute mixing signal terms
        mixingFrequencies, mixingSpectrum = ccpr.getMixingSpectrum(perturbationFrequency)

        # Compute DC Side Terms
        dcFrequencies, dcVoltageSpectrum = ccpr.getDCVoltageSpectrum(perturbationFrequency)
        dcCurrentSpectrum = Idc(dcFrequencies, dcVoltageSpectrum)

        # Compute AC Side Current
        pertrubationCurrent, perturbationMirrorCurrent = ccpr.getACCurrentSpectrum(
                                perturbationFrequency,
                                dcFrequencies,
                                dcCurrentSpectrum)
        

        # simulate
        sim.write_param_file(model+'.gen', gen_params)
        sim.execute_spice(model)

        fig, ax = plt.subplots(8, 2)
        plot_utils.plot_rectifier_ports(fig, ax, model, interpolate_dt, 1000)

        # plot ac side currents
        ax[0][1].plot(perturbationFrequency, 20*np.log10(np.abs(pertrubationCurrent)), 'bx')
        ax[0][1].plot(2 * lineFrequency - perturbationFrequency, 20*np.log10(np.abs(perturbationMirrorCurrent)), 'bx')
        ax[1][1].plot(perturbationFrequency, 180 / np.pi * np.angle(pertrubationCurrent), 'bx')    
        ax[1][1].plot(2 * lineFrequency - perturbationFrequency, 180 / np.pi * np.angle(perturbationMirrorCurrent), 'bx')

        # plot mixing signal harmonics
        ax[3][1].plot(mixingFrequencies, 20*np.log10(np.abs(mixingSpectrum)), 'bx')
        ax[4][1].plot(mixingFrequencies, 180 / np.pi * np.angle(mixingSpectrum), 'bx')

        # plot dc side spectrum
        ax[6][1].plot(dcFrequencies, 20*np.log10(np.abs(dcVoltageSpectrum)), 'r+')
        ax[6][1].plot(dcFrequencies, 20*np.log10(np.abs(dcCurrentSpectrum)), 'g*')

        ax[7][1].plot(dcFrequencies, 180 / np.pi * np.angle(dcVoltageSpectrum), 'r+')
        ax[7][1].plot(dcFrequencies, 180 / np.pi * np.angle(dcCurrentSpectrum), 'g*')
        plt.tight_layout()
        plt.show()

    def testEnvelopeModel(model, gen_params, Idc):
        # Compute mixing signal terms
        mixingFrequencies, mixingSpectrum = ccpr.getEnvelopeMixingSpectrum()

        # Compute DC Side Terms
        dcFrequencies, dcVoltageSpectrum = ccpr.getEnvelopeDCVoltageSpectrum(perturbationFrequency)
        dcCurrentSpectrum = dcCurrentSpectrum = Idc(dcFrequencies, dcVoltageSpectrum)

        # Compute AC Side Current
        lineCurrent, posSidebandCurrent, negSidebandCurrent = ccpr.getEnvelopeACCurrentSpectrum(
                                perturbationFrequency,
                                dcFrequencies,
                                dcCurrentSpectrum)
        

        # simulate
        sim.write_param_file(model+'.gen', gen_params)
        sim.execute_spice(model)

        fig, ax = plt.subplots(8, 2)
        plot_utils.plot_rectifier_ports(fig, ax, model, interpolate_dt, 1000)

        # plot ac side currents
        ax[0][1].plot(lineFrequency, 20*np.log10(np.abs(lineCurrent)), 'bx')
        ax[0][1].plot(lineFrequency - perturbationFrequency, 20*np.log10(np.abs(negSidebandCurrent)), 'bx')
        ax[0][1].plot(lineFrequency + perturbationFrequency, 20*np.log10(np.abs(posSidebandCurrent)), 'bx')
        ax[1][1].plot(lineFrequency, 180 / np.pi * np.angle(lineCurrent), 'bx')    
        ax[1][1].plot(lineFrequency - perturbationFrequency, 180 / np.pi * np.angle(negSidebandCurrent), 'bx')
        ax[1][1].plot(lineFrequency + perturbationFrequency, 180 / np.pi * np.angle(posSidebandCurrent), 'bx')

        # plot mixing signal harmonics
        ax[3][1].plot(mixingFrequencies, 20*np.log10(np.abs(mixingSpectrum)), 'bx')
        ax[4][1].plot(mixingFrequencies, 180 / np.pi * np.angle(mixingSpectrum), 'bx')

        # plot dc side spectrum
        ax[6][1].plot(dcFrequencies, 20*np.log10(np.abs(dcVoltageSpectrum)), 'r+')
        ax[6][1].plot(dcFrequencies, 20*np.log10(np.abs(dcCurrentSpectrum)), 'g*')

        ax[7][1].plot(dcFrequencies, 180 / np.pi * np.angle(dcVoltageSpectrum), 'r+')
        ax[7][1].plot(dcFrequencies, 180 / np.pi * np.angle(dcCurrentSpectrum), 'g*')
        plt.tight_layout()
        plt.show()

    if args.model == 'r':
        print("Testing with resistive load")
        R = 10

        model = 'base_R'
        gen_params = base_params.copy()
        gen_params['Rdc'] = R

        Ydc_tf = signal.lti([1/gen_params['Rdc']], [1])
        Idc = lambda f, v: v * signal.freqresp(Ydc_tf, 2 * np.pi * f)[1]

        testModel(model, gen_params, Idc)

        model = 'base_R_Envelope'
        testEnvelopeModel(model, gen_params, Idc)

    elif args.model == 'rlc':
        print("Testing with rlc load")
        R = 10
        L = 50e-3
        C = 1000e-6

        model = 'base_RLC'
        gen_params = base_params.copy()
        gen_params['Ldc'] = L
        gen_params['Cdc'] = C
        gen_params['Rdc'] = R

        gen_params['timestep'] = 10e-6

        Ydc_tf = signal.lti([0, C*R, 1], 
                            [L*C*R, L, R])
        Idc = lambda f, v: v * signal.freqresp(Ydc_tf, 2 * np.pi * f)[1]
        testModel(model, gen_params, Idc)

        model = 'base_RLC_Envelope'
        gen_params['timestep'] = 50e-6
        testEnvelopeModel(model, gen_params, Idc)

    elif args.model == 'cbcpl':
        import CBCPL

        print("Testing with cbcpl load")
        fCPL = 7
        wCPL = fCPL * 2 * np.pi
        Ydc = 0.1
        Cdc = 1e-6
        

        model = 'base_PFC_CPL'
        gen_params = base_params.copy()

        gen_params['Ydc'] = Ydc
        gen_params['Cdc'] = Cdc
        gen_params['wCPL'] = wCPL

        Idc = lambda f, v: CBCPL.getDCCurrentSpectrum(wCPL, Ydc, f, v)
        testModel(model, gen_params, Idc)

        model = 'base_PFC_CPL_Envelope'
        testEnvelopeModel(model, gen_params, Idc)



    