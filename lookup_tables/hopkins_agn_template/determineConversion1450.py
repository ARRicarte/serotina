"""
ARR: 02.17.17

Calculate the 1450 Angstrom luminosity density given a bolometric luminosity
"""

import numpy as np
import os
import cPickle as pickle
import constants

def FnuToMAB(Fnu):
	return -2.5 * np.log10(Fnu / (3631 * constants.Jy))

def calculateF1450(logL):
	#First, create a file with the data I want
	os.system('./spec {0} > calculateF1450_output.txt'.format(logL))
	with open('calculateF1450_output.txt', 'r') as myfile:
		data = np.loadtxt(myfile, comments=';')

	nu = 10**data[:-4,0]
	Lnu = 10**data[:-4,1] / nu
	Lnu_1450 = np.interp(constants.c / (1450.0*1e-10), nu, Lnu)
	Fnu_10pc = Lnu_1450 / (4 * np.pi * (10 * constants.pc)**2) * constants.L_sun
	return FnuToMAB(Fnu_10pc)

def createConversionFile(output="LbolToM1450.pkl", logLRange=np.linspace(7,15,100)):
	M1450_list = []
	for logL in logLRange:
		M1450_list.append(calculateF1450(logL))
	M1450_list = np.array(M1450_list)

	outTable = np.vstack((logLRange, M1450_list))

	with open(output, 'w') as myfile:
		pickle.dump(outTable, myfile, protocol=2)

if __name__ == '__main__':
	createConversionFile(output='LbolToM1450.pkl', logLRange=np.linspace(4,20,100))
