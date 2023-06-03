import numpy as np
from scipy import signal

##############################################################################
#                  		          Linear Loads		   	  		             #
##############################################################################

def LL_Idc(freqs, vals, Y_tf):
	return vals * signal.freqresp(Y_tf, freqs)[1]


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
def CBCPL_Idc(cutoff, Ydc_tf, freqs, Vdc, power, bias, smallsignal=True):
	max_f = np.max(freqs)
	Y = power / bias**2

	filter = signal.TransferFunction([0, 1], [1/cutoff, 1])
	filteredVdc = Vdc * signal.freqresp(filter, freqs)[1]

	# n = 0 and n = 1 terms
	vals = 2 * Vdc # Double the input terms (except DC)
	vals[freqs == 0] /= 2
	idc = -vals * signal.freqresp(Ydc_tf, freqs)[1]

	# n > 2 terms, controllable bandwidth
	vals = 2 * filteredVdc
	for i, f in enumerate(freqs):
		n = 2
		while np.abs(n*f) < max_f and f != 0:
			idx = np.where(freqs == n * f)[0]
			if len(idx) > 0:
				idx = idx[0]
				adderand = 2 * Y * bias * (-vals[i] / (2 * bias)) ** n
				if np.abs(adderand) > 1e-6:
					print("hit, (f=%d)*(n=%d)=%f: i(v=%f,n=%d)=%f+%f" % (f, n, freqs[idx], vals[i], n, idc[idx],  adderand))
					print(np.abs(vals[i]), 2 * Vdc[i])
				idc[idx] += adderand

			n += 1

	# halve the output values (except DC)
	idc[freqs != 0] /= 2
	return idc

##############################################################################
#                     Constant Conduction Passive Rectifier		  	       	 #
##############################################################################
def CCPR_Vdc(lineVoltage, lineFrequency, 
			 pertubationVoltage, pertubationFrequency,
			 maximumMixingOrder, positiveFrequenciesOnly=True,
			 combineLikeTerms=True):

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