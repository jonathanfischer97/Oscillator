
import numpy as np 
import math
import peakutils
import numpy.fft as fft
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
from matplotlib import cm  
from scipy.integrate import odeint 
import scipy.signal as signal 


import pickle 
import random as rand  
from numpy import random
#from dFlipFlop import DFlipFlop
#from acdc import ACDC  
#from repressilator import Repressilator
from sklearn import decomposition 
from deap import creator, base, tools, algorithms 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans  
import os.path 

#from sympy import latex 



class Oscillator: 
	
	def __init__(self, parameter_values, params, initial_conditions, dt = 0.001, mode = 1, sigma = 0.001): 
		self.nParams = len(params)   
		self.params = params #model parameters
		self.parameter_values = parameter_values #allowed parameter ranges  
		self.y0 = initial_conditions 	
		self.dt = dt
		self.sigma = sigma
		self.T = 100 
		self.N = int(self.T/self.dt) 
		self.ts = np.linspace(0, self.T, self.N) 
		self.amp = 300 #[nM] 		
		self.per = self.T/8 	
		self.sample_rate 		= 0.0033333333 #Hz 
		self.samples_per_hour 	= (1/self.dt)		
		self.jump 				= int(self.samples_per_hour/(self.sample_rate*3600))  	 	
		self.ideal = self.amp*(np.sin(math.pi*(self.ts)/self.per - math.pi/2) + 1) 
		#number of samples for FFT		
		self.nS = self.N/self.jump 
		self.dF = self.sample_rate/self.nS  
		self.idealF = self.getFrequencies(self.ideal) 
		self.idealFreq = 0.3 #peaks per second 	
		self.penalty = 0.5 	
		thresholdOne = -(self.nS/2)*100 #10nM -+ from ideal signal harmonics       
		thresholdTwo = 0.1
		#self.maxFreq = 0.5
		#self.maxPeaks = self.maxFreq * self.T  
		self.minAmp = 1
		self.maxAmp = 6000 
		self.mode = mode    			
		self.modes = [self.eval]       
		self.threshold = thresholdOne  
		self.omega = 100 #nm^-1 
		if self.mode == 1:
			self.threshold = thresholdTwo
	
	#gets summed difference of arrayData
	@staticmethod 	
	def getDif(indexes, arrayData):	
		arrLen = len(indexes)
		sum = 0
		for i, ind in enumerate(indexes):
			if i == arrLen - 1:
				break
			sum += arrayData[ind] - arrayData[indexes[i + 1]]
			
		#add last peak - same as substracting it from zero 
		sum += arrayData[indexes[-1:]]  
		return sum   
		
	#gets standard deviation 
	@staticmethod 
	def getSTD(indexes, arrayData, window):
		numPeaks = len(indexes)
		arrLen = len(arrayData)
		sum = 0
		for ind in indexes:
			minInd = max(0, ind - window)
			maxInd = min(arrLen, ind + window)
			sum += np.std(arrayData[minInd:maxInd])  
			
		sum = sum/numPeaks 	
		return sum	 
	
	def getFrequencies(self, y):
		#fft sample rate: 1 sample per 5 minutes
		y = y[0::self.jump]  
		res = abs(fft.rfft(y))
		#normalize the amplitudes 
		res = res/math.ceil(self.nS/2) 
		return res

	def costOne(self, Y): 
		p1 = Y[:,1]   
		fftData = self.getFrequencies(p1)     
		
		diff = fftData - self.idealF         
		cost = -np.dot(diff, diff) 		
		return cost,	
		
	def costTwo(self, Y, getAmplitude = False): 
		p1 = Y[:,5] 
		fftData = self.getFrequencies(p1)      
		fftData = np.array(fftData) 
		
		#find peaks using very low threshold and minimum distance
		#indexes = peakutils.indexes(fftData, thres=0.02/max(fftData), min_dist=1) 
		#find number of peaks in time domain for frequency threshold
		#peaknumber = len(peakutils.indexes(p1, thres=0.02/max(fftData), min_dist=1))  

		indexes = peakutils.indexes(fftData, thres=np.mean(fftData), min_dist=10, thres_abs = True) 
		#find number of peaks in time domain for frequency threshold
		peaknumber = len(peakutils.indexes(p1, thres=np.mean(p1), min_dist=1, thres_abs = True)) 

		#in case of no oscillations return 0 
		if len(indexes) == 0 or peaknumber == 0:
			return 0, 0 
		#if amplitude is greater than 400nM
		amp = np.max(fftData[indexes])
		if amp > self.maxAmp: 
			return 0, 
		fitSamples = fftData[indexes]  			
		std = self.getSTD(indexes, fftData, 1)  
		diff = self.getDif(indexes, fftData)  
		penalty =  ((abs(self.idealFreq - (peaknumber/self.T)))*self.penalty)
		cost = std + diff #- penalty #penalize difference from ideal frequency
		#print(cost)   
		if getAmplitude:
			return cost, amp
		return cost, penalty
		
	def isViableFitness(self, fit):
		return fit >= self.threshold
		
	def isViable(self, point): 
		fitness = self.eval(point, getAmplitude=True)  
		if self.mode == 0:
			return self.isViableFitness(fitness[0]) 
			
		fit = fitness[0] 
		amp = 0
		if fit > 0:
			amp = fitness[1] 
		return self.isViableFitness(fit) #and amp >= self.minAmp and amp <= self.maxAmp   
		
	#evaluates a candidate  
	def eval(self, candidate, getAmplitude = False): 
		Y = np.array(self.simulate(candidate)) 
		if self.mode == 0:
			return self.costOne(Y)  
		else:
			return self.costTwo(Y, getAmplitude = False)     
	
	#simulates a candidate
	def simulate(self, candidate): 
		self.y0 = [candidate[13],0,candidate[14],candidate[15],0,candidate[16],0,0,0,0,0,0]
		return odeint(self.oscillatorModelOde, self.y0, self.ts, args=(candidate,))   		

	def plotModel(self, subject, mode="ode", show=True):     		
		if mode == "ode":
			t = np.linspace(0, self.T, self.N)
			solutions = self.simulate(subject) 			
		#else:
			#ssa simulation
			#ts,Y = self.represilatorStochastic(subject)
			
		fig = plt.figure(figsize=(8,4),dpi=200)

		plt.plot(t,solutions[:,0], color="b", label = 'L')
		plt.plot(t,solutions[:,1], color="g", label = 'Lp')
		plt.plot(t,solutions[:,2], color="gold", label = 'K')
		plt.plot(t,solutions[:,3], color="r", label = 'P')
		plt.plot(t,solutions[:,4], color="cyan", label = 'LK')
		plt.plot(t,solutions[:,5], color="black", label = 'A')
		plt.plot(t,solutions[:,6], color="indigo", label = 'LpA')
		plt.plot(t,solutions[:,7], color="yellow", label = 'LpAK')
		plt.plot(t,solutions[:,8], color="magenta", label = 'LpAP')
		plt.plot(t,solutions[:,9], color="deeppink", label = 'LpAPLp')
		plt.plot(t,solutions[:,10], color="peru", label = 'LpAKL')

		plt.plot(t,solutions[:,11], color="purple", label = 'PLp')


		plt.xlabel('Time (s)')
		plt.ylabel('Concentration (uM)')
		##plt.ylim(top=110, bottom = -10)
		plt.legend(loc = 'upper right', prop={'size': 6})

		plt.show()
			 				

	def getTotalVolume(self):
		vol = 1.0
		for param in self.params:		
			vol = vol*(self.parameter_values[param]["max"] - self.parameter_values[param]["min"])
		return vol 

	def oscillatorModelOde(self, Y, t, can): 

		#V = can[12]
		VA = can[12]
		#A = V/VA
		sigma = self.sigma
		#y1 = VA/(2*sigma)
		
		
		ka1 = can[0]
		kb1 = can[1]
		kcat1 = can[2]

		ka2 = can[3]
		kb2 = can[4]

		ka3 = can[5]
		kb3 = can[6]

		ka4 = can[7]
		kb4 = can[8]

		ka7 = can[9]
		kb7 = can[10]
		kcat7 = can[11]

		ka5 = can[9]
		kb5 = can[10]
		kcat5 = can[11]

		ka6 = can[0]
		kb6 = can[1]
		kcat6 = can[2]

	
		# ka6 = can[12] 
		# kb6 = can[13]
		# kcat6 = can[14]

		
		#initial conditions
		L = Y.item(0)   # make sure in uM
		Lp = Y.item(1) 
		K = Y.item(2)
		P = Y.item(3)
		LK = Y.item(4) 
		A = Y.item(5)
		LpA = Y.item(6) 
		LpAK = Y.item(7) 
		LpAP = Y.item(8) 
		LpAPLp = Y.item(9) 
		LpAKL = Y.item(10) 
		LpP = Y.item(11) 
	
		dL = (kb1*LK) - (ka1*L*K) + (kcat5*LpAPLp) + (kb6*LpAKL) - ((VA/(2*sigma))*ka6*LpAK*L) + (kcat7*LpP)
		dLp = (kcat1*LK) + (kb2*LpA) - (ka2*Lp*A) + (kb5*LpAPLp) - ((VA/(2*sigma))*ka5*Lp*LpAP) + (kcat6*LpAKL) - (ka7*Lp*P) + (kb7*LpP)
		dK = (kb1*LK) - (ka1*L*K) + (kcat1*LK) + (kb3*LpAK) - (ka3*LpA*K)
		dP = (kb4*LpAP) - (ka4*LpA*P) - (ka7*Lp*P) + (kb7*LpP) + (kcat7*LpP)
		dLK = (ka1*L*K) - (kb1*LK) - (kcat1*LK)
		dA = (kb2*LpA) - (ka2*Lp*A)
		dLpA = (ka2*Lp*A) - (kb2*LpA) + (kb3*LpAK) - (ka3*LpA*K) + (kb4*LpAP) - (ka4*LpA*P)
		dLpAK = (ka3*LpA*K) - (kb3*LpAK) + (kb6*LpAKL) - ((VA/(2*sigma))*ka6*LpAK*L) + (kcat6*LpAKL)
		dLpAP = (ka4*LpA*P) - (kb4*LpAP) + (kb5*LpAPLp) - ((VA/(2*sigma))*ka5*LpAP*Lp) + (kcat5*LpAPLp)
		dLpAPLp = ((VA/(2*sigma))*ka5*LpAP*Lp) - (kb5*LpAPLp) - (kcat5*LpAPLp)
		dLpAKL = ((VA/(2*sigma))*ka6*LpAK*L) - (kb6*LpAKL) - (kcat6*LpAKL)
		dLpP = (ka7*Lp*P) - (kb7*LpP) - (kcat7*LpP)

		return([dL, dLp, dK, dP, dLK, dA, dLpA, dLpAK, dLpAP, dLpAPLp, dLpAKL, dLpP])	
	
	def getPerAmp(self, subject, mode="ode", indx=0): 
		if mode == "ode":
			ts = np.linspace(0, self.T, self.N) 
			Y = self.simulate(subject)    				
		#else:
			#ts,Y = self.represilatorStochastic(subject) 
		ts = np.array(ts) 
		Y = np.array(Y) 
		sig = Y[:, indx]
		indx_max, properties = signal.find_peaks(sig, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/100)      
		indx_min, properties = signal.find_peaks(sig*-1, prominence = (np.max(sig) - np.min(sig))/4, distance = len(ts)/100)     

		amps = [] 
		pers = []   
		for i in range(min(len(indx_max), len(indx_min))):
			amps.append((sig[indx_max[i]] - sig[indx_min[i]])/2) 			
			if i + 1 < len(indx_max):
				pers.append(ts[indx_max[i + 1]] - ts[indx_max[i]])
			if i + 1 < len(indx_min):
				pers.append(ts[indx_min[i + 1]] - ts[indx_min[i]])
		
		if len(amps) > 0 and len(pers) > 0:
			amps = np.array(amps)   	
			pers = np.array(pers)  
			
			#print(amps)
			amp = np.mean(amps)	
			#print(pers) 
			per = np.mean(pers) 
		else:
			amp = 0
			per = 0  
		
		print("amp" + str(amp)) 
		print("per" + str(per))   	
		
		return per, amp 

	def ssa(self, can):
		omega = self.omega
		self.y0 = [can[13],0,can[14],can[15],0,can[16],0,0,0,0,0,0]
		y0 = [i*omega for i in self.y0]
		y_conc = np.array(y0).astype(int)
		Y_total = []
		Y_total.append(y_conc)
		t = 0 
		t_end = 100
		T = []   
		T.append(t)
		
		#get kinetic rates 
		VA = can[12]
		#A = V/VA
		sigma = 0.001
		y = VA/(2*sigma)
		
		
		ka1 = can[0]
		kb1 = can[1]
		kcat1 = can[2]

		ka2 = can[3]
		kb2 = can[4]

		ka3 = can[5]
		kb3 = can[6]

		ka4 = can[7]
		kb4 = can[8]

		ka7 = can[9]
		kb7 = can[10]
		kcat7 = can[11]

		ka5 = can[9]*y
		kb5 = can[10]
		kcat5 = can[11]

		ka6 = can[0]*y
		kb6 = can[1]
		kcat6 = can[2]
		
		N = np.zeros((18,12)) #6 species, 15 reactions
		#L+K -> LK
		N[0,0] = -1 
		N[0,2] = -1 
		N[0,4] = 1

		#LK -> L+K
		N[1,0] = 1
		N[1,2] = 1
		N[1,4] = -1

		#LK -> Lp+K
		N[2,4] = -1
		N[2,1] = 1
		N[2,2] = 1

		#Lp+A -> LpA
		N[3,1] = -1
		N[3,5] = -1
		N[3,6] = 1

		#LpA -> Lp+A
		N[4,1] = 1
		N[4,5] = 1
		N[4,6] = -1
		
		#LpA + K -> LpAK
		N[5,6] = -1 
		N[5,2] = -1
		N[5,7] = 1

		#LpAK -> LpA + K
		N[6,6] = 1
		N[6,2] = 1
		N[6,7] = -1 
		
		#L + LpAK -> LpAKL
		N[7,0] = -1
		N[7,7] = -1
		N[7,10] = 1

		#LpAKL -> L + LpAK
		N[8,0] = 1
		N[8,7] = 1
		N[8,10] = -1

		#LpAKL -> Lp + LpAK
		N[9,10] = -1
		N[9,1] = 1
		N[9,7] = 1

		#Lp + P -> LpP
		N[10,1] = -1
		N[10,3] = -1
		N[10,11] = 1

		#LpP -> Lp + P
		N[11,1] = 1
		N[11,3] = 1
		N[11,11] = -1

		#LpP -> L + P
		N[12,11] = -1
		N[12,0] = 1
		N[12,3] = 1

		#LpA + P -> LpAP
		N[13,6] = -1
		N[13,3] = -1
		N[13,8] = 1

		#LpA + P <- LpAP
		N[14,6] = 1
		N[14,3] = 1
		N[14,8] = -1

		#Lp + LpAP -> LpAPLp
		N[15,1] = -1
		N[15,8] = -1
		N[15,9] = 1

		#Lp + LpAP <- LpAPLp
		N[16,1] = 1
		N[16,8] = 1
		N[16,9] = -1

		#LpAPLp -> L + LpAP
		N[17,9] = -1
		N[17,0] = 1
		N[17,8] = 1 
		
		while t < t_end:
			#choose two random numbers 
			r = np.random.uniform(size=2)
			r1 = r[0] 
			r2 = r[1] 					
			
			#get propensities
			a = np.zeros(18)
			a[0] = ka1*y_conc[0]*y_conc[2] 
			a[1] = kb1*y_conc[4]
			a[2] = kcat1*y_conc[4]  
			
			a[3] = ka2*y_conc[1]*y_conc[5] 
			a[4] = kb2*y_conc[6]

			a[5] = ka3*y_conc[6]*y_conc[2]
			a[6] = kb3*y_conc[7]

			a[7] = ka6*y_conc[0]*y_conc[7]
			a[8] = kb6*y_conc[10]
			a[9] = kcat6*y_conc[10]

			a[10] = ka7*y_conc[1]*y_conc[3]
			a[11] = kb7*y_conc[11]
			a[12] = kcat7*y_conc[11]

			a[13] = ka4*y_conc[6]*y_conc[3]
			a[14] = kb4*y_conc[8]

			a[15] = ka5*y_conc[1]*y_conc[8]
			a[16] = kb5*y_conc[9]
			a[17] = kcat5*y_conc[9]  
			
			asum = np.cumsum(a)
			a0 = np.sum(a)  
			#get tau
			tau = (1.0/a0)*np.log(1.0/r1)     
		
			#select reaction 
			reaction_number = np.argwhere(asum > r2*a0)[0,0] #get first element			
		
			#update concentrations
			y_conc = y_conc + N[reaction_number,:]   	
			Y_total.append(y_conc) 
			#update time
			t = t + tau  
			T.append(t)
		T = np.array(T) 
		Y_total = np.array(Y_total)
		return T, Y_total  
	

'''
Regions consist of cloud of points and principal component that govern the direction of exploration  
''' 
class Region: 
	def __init__(self, points, model, label, depth=1):    
		self.points = np.array(points)  
		self.model = model  
		self.pca = PCA(n_components=self.model.nParams)
		self.components = None
		self.prevComponents = None 
		self.cluster = False
		self.terminated = False  
		self.iter = 0      
		self.maxIter = 10            
		self.threshold = 0.001    
		self.label = label
		self.maxVarScale = 4
		self.minVarScale = 2   
		self.varScaleDt = (self.maxVarScale - self.minVarScale)/(float(self.maxIter))    		     		
		self.varScale = self.maxVarScale         
		self.depth = depth     
		
	def updateVariance(self): 
		self.varScale = self.varScale - self.varScaleDt

	def updateIter(self):
		self.iter = self.iter + 1
		self.updateVariance()          	
		
	def fitPCA(self): 
		self.prevComponents = self.components 
		self.pca.fit(self.points)
		self.components = self.pca.components_
	
	def transform(self, points):  
		return self.pca.transform(points)  
		
	def inverse_transform(self, points):
		return self.pca.inverse_transform(points)   
		
	def converged(self):
		if self.components is None or self.prevComponents is None: 
			return False		
		return np.linalg.norm(self.components - self.prevComponents) < self.threshold   
		
	def explored(self):    
		return self.terminated or self.iter > self.maxIter or self.converged()   

def write_parms(params, waterbox, iteration, name):
	with open(name + str(iteration) + 'parms.inp', 'w+') as f:
		f.write('start parameters' + '\n')
		f.write('\t' + 'nItr = 100000000' + '\n')
		f.write('\t' + 'timeStep = 1' + '\n')
		f.write('\t' + 'timeWrite = 500' + '\n')
		f.write('\t' + 'trajWrite = 1000000000' + '\n')
		f.write('\t' + 'restartWrite = 100000000' + '\n')
		f.write('\t' + 'checkPoint = 100000000' + '\n')
		f.write('\t' + 'pdbWrite = 100000000' + '\n')
		f.write('end parameters' + '\n' + '\n')
	   
		f.write('start boundaries' + '\n')
		f.write('\t' + 'WaterBox = ' + str(waterbox) + '\n')
		f.write('end boundaries' + '\n' + '\n')

		f.write('start molecules' + '\n')
		f.write('\t' + 'pip2 : ' + str(params[13]) + ' (head~U)' + '\n')
		f.write('\t' + 'ap2 : ' + str(params[16]) + '\n')
		f.write('\t' + 'kin : ' + str(params[14]) + '\n')
		f.write('\t' + 'syn : ' + str(params[15]) + '\n')
		f.write('end molecules' + '\n' + '\n')

		f.write('start reactions' + '\n')
		f.write('\t' + '### KIN - PIP2 ###' + '\n')
		f.write('\t' + 'kin(pi) + pip2(head~U) <-> kin(pi!1).pip2(head~U!1)' + '\n')
		f.write('\t' + 'onRate3Dka = ' + str(params[0]) + '\n')
		f.write('\t' + 'offRatekb = ' + str(params[1]) + '\n')
		f.write('\t' + 'kcat = ' + str(params[2]) + '\n')
		f.write('\t' + 'norm1 = [0,0,1]' + '\n')
		f.write('\t' + 'norm2 = [0,0,1]' + '\n')
		f.write('\t' + 'assocAngles = [1.5707963, 1.5707963, nan, nan, M_PI]' + '\n')
		f.write('\t' + 'sigma = ' + str(params[-1]) + '\n')
		f.write('\t' + 'coupledRxnLabel = autoP' + '\n' + '\n')

		f.write('\t' + '# KIN autoPhosphorylation #' + '\n')
		f.write('\t' + 'pip2(head~U) -> pip2(head~P)' + '\n')
		f.write('\t' + 'rate = 0.0' + '\n')
		f.write('\t' + 'rxnLabel = autoP' + '\n' + '\n')

		f.write('\t' + '#### PIP2 - AP2 ####' + '\n')
		f.write('\t' + 'pip2(head~P) + ap2(m2muh) <-> pip2(head~P!1).ap2(m2muh!1)' + '\n')
		f.write('\t' + 'onRate3Dka = ' + str(params[3]) + '\n')
		f.write('\t' + 'offRatekb = ' + str(params[4]) + '\n')
		f.write('\t' + 'norm1 = [0,0,1]' + '\n')
		f.write('\t' + 'norm2 = [0,0,1]' + '\n')
		f.write('\t' + 'assocAngles = [1.5707963, 1.5707963, nan, nan, M_PI]' + '\n')
		f.write('\t' + 'sigma = ' + str(params[-1]) + '\n' + '\n')

		f.write('\t' + '### KIN - AP2 ###' + '\n')
		f.write('\t' + 'kin(ap) + ap2(sy,m2muh!*) <-> kin(ap!1).ap2(sy!1,m2muh!*)' + '\n')
		f.write('\t' + 'onRate3Dka = ' + str(params[5]) + '\n')
		f.write('\t' + 'offRatekb = ' + str(params[6]) + '\n')
		f.write('\t' + 'norm1 = [0,0,1]' + '\n')
		f.write('\t' + 'norm2 = [0,0,1]' + '\n')
		f.write('\t' + 'assocAngles = [1.5707963, 2.35619, nan, M_PI, M_PI]' + '\n')
		f.write('\t' + 'sigma = ' + str(params[-1]) + '\n' + '\n')

		f.write('\t' + '### SYN - AP2 ###' + '\n')
		f.write('\t' + 'syn(ap) + ap2(sy,m2muh!*) <-> syn(ap!1).ap2(sy!1,m2muh!*)' + '\n')
		f.write('\t' + 'onRate3Dka = ' + str(params[7]) + '\n')
		f.write('\t' + 'offRatekb = ' + str(params[8]) + '\n')
		f.write('\t' + 'norm1 = [0,0,1]' + '\n')
		f.write('\t' + 'norm2 = [0,0,1]' + '\n')
		f.write('\t' + 'assocAngles = [1.5707963, 2.35619, nan, M_PI, M_PI]' + '\n')
		f.write('\t' + 'sigma = ' + str(params[-1]) + '\n' + '\n')

		f.write('\t' + '### SYN - PIP2 ###' + '\n')
		f.write('\t' + 'syn(pi) + pip2(head~P) <-> syn(pi!1).pip2(head~P!1)' + '\n')
		f.write('\t' + 'onRate3Dka = ' + str(params[9]) + '\n')
		f.write('\t' + 'offRatekb = ' + str(params[10]) + '\n')
		f.write('\t' + 'kcat = ' + str(params[11]) + '\n')
		f.write('\t' + 'norm1 = [0,0,1]' + '\n')
		f.write('\t' + 'norm2 = [0,0,1]' + '\n')
		f.write('\t' + 'assocAngles = [1.5707963, 1.5707963, nan, nan, M_PI]' + '\n')
		f.write('\t' + 'sigma = ' + str(params[-1]) + '\n')
		f.write('\t' + 'coupledRxnLabel = autoU' + '\n' + '\n')

		f.write('\t' + '# SYN autoDephosphorylation #' + '\n')
		f.write('\t' + 'pip2(head~P) -> pip2(head~U)' + '\n')
		f.write('\t' + 'rate = 0.0' + '\n')
		f.write('\t' + 'rxnLabel = autoU' + '\n' + '\n')

		f.write('end reactions')


class Solver:
	def __init__(self, model, name, populationSize=5000, NGEN = 10, nsamples = 1e5, volume = 0.5):                                                      
		self.model = model 
		self.name = name           
		self.populationSize = populationSize         
		self.NGEN = NGEN  
		self.nsamples = int(nsamples) 	
		self.indpb = 0.75     
		self.volume = volume	
		
		#GA operators
		creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
		creator.create("Candidate", list, fitness=creator.FitnessMax)  		
		self.toolbox = base.Toolbox()	 
		self.toolbox.register("candidate", self.generateCandidate) 
		self.toolbox.register("population", tools.initRepeat, list, self.toolbox.candidate)  
		self.toolbox.register("mate", tools.cxTwoPoint)
		self.toolbox.register("mutate", self.mutateCandidate, indpb=self.indpb, mult=0.5)      
		self.toolbox.register("select", tools.selTournament, tournsize=int(self.populationSize/10))     		
	
	#### functions for printing NERDSS parameters ####
	def waterbox(self,V,VA):
		V = V * 1e9 #convert um3 to nm3
		VA = VA * 1000 #convert um to nm 

		A = V/VA #units of nm2
		x = round(np.sqrt(A))
		y = round(np.sqrt(A))
		z = round(V/A) 

		dimensions = [x,y,z]
		#ztest = z/10
		#xtest = np.sqrt(V/ztest)
		#ytest = np.sqrt(V/ztest)

		#test = (ztest*xtest*ytest)/(xtest*ytest)

		return dimensions

	def copynumber(self, conc, volume, um = True):
		if um == True:
			volume = volume/1e15 #converts liters to um^3
			conc = conc/1e6 #converts umol to mol
			
			moles = conc * volume #volume must be passed in um^-3
			copies = moles * 6.023e23
			
			return copies
		elif um == False:
			volume = volume/1e15
			moles = conc / volume
			rate = moles / 6.023e23

			return rate

	def rate_converter(self, rate, V, ode = False):
		#convert ka in (uM*s)^-1 to nm^3/us
		if ode == True:
			rate1 = rate/0.602214076 ##conversion ratio from page 10 of NERDSS manual
			rate2 = rate1*1e6 ##convert from us to s
			volume = V*1e9 #convert volume to nm3
			return rate2/volume ##calculate copy numbers per second
			#new_rate = rate/1e6 #seconds to microseconds
			#new_rate = new_rate * 1e24 #convert liters to nm^3
			#new_rate = new_rate/(6.023e17) #convert micromoles to copies 
			#return new_rate
		else: #else kb or kcat in s^-1
			#new_rate = rate / 1e6 #convert per second to per microsecond
			return rate/0.602214076 




	#estimate initial values with GA
	def findNominalValues(self):    	 	
		nominalVals = []   
		
		for evalMode in self.model.modes: 
			nominalValsMode = []
			self.toolbox.register("evaluate", evalMode)   
			#initialize new random population
			self.popu = self.toolbox.population(self.populationSize)  	 
			
			for gen in range(self.NGEN):  
				print(gen)
				#generate offspprings with crossover and mutations
				offspring = algorithms.varAnd(self.popu, self.toolbox, cxpb=0.5, mutpb=0.75)  
				#evaluate individuals
				fits = self.toolbox.map(self.toolbox.evaluate, offspring) 
				for fit, ind in zip(fits, offspring): 
					if self.model.isViable(ind) and ind not in nominalValsMode:  
						nominalValsMode.append(ind)      
					ind.fitness.values = fit     
				#roulete wheel selection
				self.popu = self.toolbox.select(offspring, k=len(self.popu)) 
				
				rando = np.random.randint(0, len(self.popu))
				rdm_ind = self.popu[rando]
				rdm_ind_label = {}

				for parameter, value in zip(parameter_values.keys(), rdm_ind):
					rdm_ind_label[parameter] = value


				print(rdm_ind_label)

				kadic = {'ka1':rdm_ind[0],'ka2':rdm_ind[3],'ka3':rdm_ind[5],'ka4':rdm_ind[7],'ka7':rdm_ind[9]}
				kbdic = {'kb1':rdm_ind[1],'kb2':rdm_ind[4],'kb3':rdm_ind[6],'kb4':rdm_ind[8],'kb7':rdm_ind[10]}
				kcatdic = {'kcat1':rdm_ind[2],'kcat7':rdm_ind[11]}
				KDdic = {'Km1':(rdm_ind[1]+rdm_ind[2])/rdm_ind[0],'Kd2':rdm_ind[4]/rdm_ind[3],'Kd3':rdm_ind[6]/rdm_ind[5],'Kd4':rdm_ind[8]/rdm_ind[7],'Km7':(rdm_ind[10]+rdm_ind[11])/rdm_ind[9]}
				comparison_list = [kadic,kbdic,kcatdic,KDdic]

				for h in comparison_list:
					sorted_dict = {}
					sorted_keys = sorted(h, key=h.get)  # [1, 3, 2]

					string = str()
					for i in range(len(sorted_keys)):
						string += sorted_keys[i] + " < "
					print(string)
				
				#calculate waterbox dimensions for NERDSS
				waterbox = self.waterbox(self.volume, rdm_ind[12])

				#convert parameters for NERDSS
				NERDSSparamlist = []
				parmslist = []
				ODElist = []
				for param in rdm_ind_label.items():
					if 'ka' in param[0]:
						converted_rate = self.rate_converter(param[1],self.volume)
						NERDSSparamlist.append(param[0] + ': ' + str(converted_rate) + ' nm^3/us')
						parmslist.append(str(converted_rate))
						converted_rate_ode = self.rate_converter(param[1],self.volume,True)
						ODElist.append(converted_rate_ode)
					elif 'kb' in param[0] or 'kcat' in param[0]:
						#converted_rate = self.rate_converter(param[1], ka = False)
						NERDSSparamlist.append(param[0] + ': ' + str(param[1]) + ' s^-1')
						parmslist.append(str(param[1]))
						ODElist.append(param[1])
					elif 'VA' in param[0]:
						gamma = param[1]/(2*self.model.sigma)
						ODElist.append(gamma)
						ODElist.append(self.model.sigma * 1e3)
					#elif 'V' in param[0]:
					#	pass
					#elif 'sigma' in param[0]:
					#	sig = param[1] * 1e3
					#	NERDSSparamlist.append(param[0] + ': ' + str(sig/2) + ' nm') ##divide by 2 for some reason
					#	parmslist.append(str(sig/2))
					else:
						copynumber = self.copynumber(param[1],self.volume)
						NERDSSparamlist.append(param[0] + ': ' + str(round(copynumber)) + ' copies')
						parmslist.append(str(round(copynumber)))
						ODElist.append(copynumber)

				NERDSSparamlist.append('sigma: ' + str((self.model.sigma)*1e3/2))
				parmslist.append(str((self.model.sigma)*1e3/2))
				ODElist.extend([0,0,0,0,0,0,0,0])
				

				write_parms(parmslist, waterbox, gen, self.name)

				print('\n'+ 'NERDSS parameters' + '\n')
				print(NERDSSparamlist)

				print('\n'+ 'Waterbox dimensions' + '\n')
				print(waterbox)

				print('\n' + 'ODE test parameters' + '\n')
				print(ODElist)


				print("Number of viable points: " + str(len(nominalValsMode))) 

				print(self.model.isViable(rdm_ind))
				
				cost, penalty = self.model.eval(rdm_ind)
				print('Cost: ' + str(cost))
				try:
					print('Frequency penalty: ' + str(penalty))
				except IndexError:
					pass 
				self.model.plotModel(rdm_ind)



					 
			
			print("Number of viable points: " + str(len(nominalValsMode))) 
			nominalVals.extend(nominalValsMode)     
		return nominalVals        


		
	#creates an array of random candidates  
	def generateCandidate(self): 
		candidate = []
		for ind in range(self.model.nParams): 
			#try:
				#candidate.append(loguniform.rvs(self.model.parameter_values[self.model.params[ind]]["min"], self.model.parameter_values[self.model.params[ind]]["max"]))
			#except ValueError:
			candidate.append(random.uniform(self.model.parameter_values[self.model.params[ind]]["min"], self.model.parameter_values[self.model.params[ind]]["max"]))
		return creator.Candidate(candidate) 	
		
	def checkOutAllBounds(self, candidate):
		for idx, val in enumerate(candidate):
			if self.checkOutOfBounds(candidate, idx): 
				return True  
		return False      
				
	def checkOutOfBounds(self, candidate, idx): 
		#if out of bounds return True 
		if candidate[idx] < self.model.parameter_values[self.model.params[idx]]["min"] or candidate[idx] > self.model.parameter_values[self.model.params[idx]]["max"]: 
			return True
		return False    		
	
	#returns a tuple of mutated candidate	
	def mutateCandidate(self, candidate, indpb, mult): 	
		for idx, val in enumerate(candidate):	
			rnd = random.uniform(0, 1)
			if rnd >= indpb:
				rnd2 = random.uniform(1 - mult, 1 + mult)   
				candidate[idx] = val*rnd2	
				if candidate[idx] < self.model.parameter_values[self.model.params[idx]]["min"]: 
					candidate[idx] = self.model.parameter_values[self.model.params[idx]]["min"]  
				if candidate[idx] > self.model.parameter_values[self.model.params[idx]]["max"]:  
					candidate[idx] = self.model.parameter_values[self.model.params[idx]]["max"]    					
		return candidate,     
	
	def getViablePoints(self, points):
		viable = list() 
		i = 0
		for point in points:  
			i += 1
			if i % 1000 == 0:
				print(i)     
			
			#check if point is viable 
			if self.model.isViable(point): 
				viable.append(point)   		
		return viable          
	
	# gap statistic method
	# returns the optimal number of clusters 	
	def gapStatistic(self, region, number_ref = 10, max_clusters = 2, plot = False):        
		#sample size is equal to the number of samples in gaussian sampling  
		sample_size = self.nsamples    
		subjects = np.array(region.points)                 
		gaps = []
		deviations = []   
		references = [] 
		clusters_range = range(1, max_clusters + 1) 
		
		transformed = region.transform(subjects) 
		#get min and max parameter values in pca space 
		minP = np.min(transformed, axis=0)  
		maxP = np.max(transformed, axis=0)   
		
		for gap_clusters in clusters_range:
			print(gap_clusters) 
			reference_inertia = []	
			for index in range(number_ref): 

				#OBB ... orientated bounding box 
				#random sampling within the PCA bounding box			
				reference = minP + random.rand(sample_size, self.model.nParams)*(maxP - minP)
				reference = region.inverse_transform(reference) 
				
				kmeanModel = KMeans(gap_clusters) 
				kmeanModel.fit(reference) 
				reference_inertia.append(kmeanModel.inertia_)    
			
			kmeanModel = KMeans(gap_clusters)      
			kmeanModel.fit(subjects)     
			log_ref_inertia = np.log(reference_inertia)	 
			#calculate gap
			gap = np.mean(log_ref_inertia) - np.log(kmeanModel.inertia_)  
			sk = math.sqrt(1 + 1.0/number_ref)*np.std(log_ref_inertia)  
			gaps.append(gap)    
			deviations.append(sk)        			
			
		# Plot the gaps   		
		if plot:
			plt.clf() 
			ax = plt.gca() 
			ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))	 
			ax.xaxis.set_major_locator(ticker.MultipleLocator(2))	
			lines = plt.errorbar(clusters_range, gaps, ecolor='dodgerblue', yerr=deviations, fmt='-', color='dodgerblue') 
			plt.setp(lines[0], linewidth=1.5)  
			plt.ylabel('Gaps')
			plt.show()  
			
		#return optimal number of clusters
		for k in range(0, max_clusters - 1): 
			if gaps[k] >= gaps[k + 1] - deviations[k + 1]: 
				print("Optimal number of clusters: " + str(k + 1)) 
				return k + 1     
		print("Optimal number of clusters: " + str(max_clusters))    	
		return max_clusters   

	
	#returns the viable volume for 
	def getViableVolume(self, viableRegions, sample_size = int(1e4)):
		volume = 0 

		for region in viableRegions:		
			regPoints = region.points
			region.fitPCA() 						 						
			transformed = region.transform(regPoints) 
		  		
			minP = np.min(transformed, axis=0)   
			maxP = np.max(transformed, axis=0)   
		
			dP = maxP - minP
			volB = np.prod(dP)			

			mcRef = minP + random.rand(sample_size, self.model.nParams)*dP  
			mcRef = region.inverse_transform(mcRef)	 		
			
			viaPoints = self.getViablePoints(mcRef) 
			count = np.ma.size(viaPoints, axis=0) 
			
			#volume for region  
			ratio = count/sample_size   
			volume = volume + ratio*volB  			
	
		print("Bounding box volume " + str(volB)) 
		print("Volume " + str(volume))   
		print("Total volume " + str(self.model.getTotalVolume()))   		
		print("Volume ratio:" + str(volume/self.model.getTotalVolume())) 
		return volume 


	def setBoxColors(self, bp, nRegions, ax, colors = ["#0E74C8", "#15A357", "r", "k"]):
		colorLen = len(colors) 

		for i in range(nRegions): 
			col = colors[i % colorLen] 		 
			plt.setp(bp['boxes'][i], color=col, linewidth=1.5)    
			plt.setp(bp['caps'][2*i], color=col, linewidth=1.5)  
			plt.setp(bp['caps'][2*i + 1], color=col, linewidth=1.5) 
			plt.setp(bp['whiskers'][2*i], color=col, linewidth=1.5)  
			plt.setp(bp['whiskers'][2*i + 1], color=col, linewidth=1.5)   
			plt.setp(bp['fliers'][i], color=col) 
			plt.setp(bp['medians'][i], color=col, linewidth=1.5)   
		
	def plotParameterVariances(self, viableSets, names=None, units=None):      
		#go through all parameters  
		params = self.model.params    
		figure = plt.figure()     
		nRows = math.ceil(len(params)/3)    
		for pcount, param in enumerate(params):    
			ax1 = plt.subplot(nRows, 3, pcount+1)  
			#if names == None:
			#	ax1.set_title(str(param) + str(pcount))    
			#else:
			#	ax1.set_title(names[pcount])  
			if units != None:
				plt.ylabel(names[pcount] + " " + units[pcount])  
			allRegions = [] 	
			#go through all regions 
			numSets = len(viableSets) 
			allNames = []
			allBoxes = []
			for count, reg in enumerate(viableSets): 
				points = np.array(reg.points)    
				data = points[:,pcount]   
				allRegions.append(data)   
				allNames.append("Region " + str(count + 1))   				
			bp = ax1.boxplot(allRegions, positions=list(range(1, numSets + 1)), widths = 0.4) 
			self.setBoxColors(bp, numSets, ax1) 		
			allBoxes = bp['boxes'] 
			
		#draw legend 
		figure.legend(allBoxes, allNames, 'lower right')
		plt.show()     
		
	#Main method  
	def run(self, filename, maxDepth=0):    
		#filename is a file to which viable sets will be serialized    

		#estimate the inital viable set 
		viablePoints = self.findNominalValues()         		                 		
		
		if not viablePoints: 
			print("No viable points found!")  
			return 
		
		
		
		#dump viable points to file  
		pickle.dump(viablePoints, open(filename + "ViableSet_IterGA.p", "wb+"))   
		
		reg = Region(viablePoints, self.model, "0")   
		reg.fitPCA() 
		
		fpca = PCA(n_components=2)  		 		
		fpca.fit(reg.points)
				
		viableSets = list() 
		viableSets.append(reg)  		  		 
		converged = False 		
		iter = 0 
		
		while not converged: 
			converged = True 			 		 	 					  	 
			iter += 1 
			print("Iteration: " + str(iter))  	
			for set in viableSets:   				
				set.updateIter() 
				#if set not already explored  
				if not set.explored():
					setSize = len(set.points) 
					print("Label: " + set.label)   
					print("Iter: " + str(set.iter))  
					print("Variance scaling factor: " + str(set.varScale))    								
					converged = False   					
						  
					#sample with 0 mean and scaled variance of prinicpal components       
					candidateSet = random.multivariate_normal([0]*self.model.nParams, np.diag(set.pca.explained_variance_)*set.varScale, self.nsamples)				
					candidateSet = set.inverse_transform(candidateSet)      
								
					#check if parameter values are not out of range  		
					inBounds = list() 
					for cand in candidateSet: 				
						if not self.checkOutAllBounds(cand): 
							inBounds.append(cand)  
					inBounds = np.array(inBounds)   		
					candidateSet = inBounds   
					
					X = fpca.transform(set.points) 
					Y = fpca.transform(candidateSet) 	 				
					fig = plt.figure(iter)  
					plt.clf()          
					plt.scatter(Y[:, 0], Y[:, 1], c="red", alpha=0.1, edgecolor='k', rasterized=True, label = "Candidate")  
					plt.scatter(X[:, 0], X[:, 1], c="cornflowerblue", alpha=0.8, edgecolor='k', rasterized=True, label = "Viable") 

					plt.xlabel('PC 1') 
					plt.ylabel('PC 2')  
					plt.title("Iteration"+str(set.iter))  
					plt.legend()
					plt.savefig(filename + "Set" + set.label + "Iter" + str(set.iter) + ".pdf")        	 
					#identify viable points  
					viablePoints = np.array(self.getViablePoints(candidateSet)) 
					
					#if viable set is smaller than number of parameters do not accept it
					print("Number of viable points: " + str(len(viablePoints)))   
					
					if len(viablePoints) <= setSize/10:   						 
						#cluster if not enough points obtained with sampling    
						print("Clustering, insufficient number of points")   
						set.terminated = True      
						set.cluster = True           
					else:
						pickle.dump(candidateSet, open(filename + "_Region" + str(set.label) + "CandidateSet_Iter" + str(set.iter) +  ".p", "wb+"))     
						pickle.dump(viablePoints, open(filename + "_Region" + str(set.label) + "ViableSet_Iter" + str(set.iter) + ".p", "wb+"))  						
						set.points = viablePoints           
						set.fitPCA()     	 			
				#if set not already terminated, terminate it and cluster   
				elif not set.terminated:    
					set.terminated = True       
					set.cluster = True         					
						
			#clustering, check for new clusters        	            
			newViableSets = list()   
			for set in viableSets: 
				if set.cluster and (maxDepth == 0 or set.depth < maxDepth):   
					set.cluster = False    
					setLabel = set.label    
					setDepth = set.depth      
					#determine the optimal number of clusters
					print("Clustering set" + set.label)      
					k = self.gapStatistic(set)    
					if k > 1:     
						#cluster and divide sets based on clustering 
						#update the list of sets 
						converged = False     
						labels = KMeans(n_clusters = k).fit_predict(set.points)         
						for i in range(k): 
							ind = np.where(labels == i)[0]  
							points = set.points[ind]
							reg = Region(points, self.model, setLabel + str(i), depth=setDepth+1)         
							reg.fitPCA()      
							#append new region to new set  
							newViableSets.append(reg)       						 			
			#end of clustering					 
			viableSets.extend(newViableSets)     
		#end of while loop  

parameter_values = {  "ka1": {"min": 1000/1e6, "max": 1e6/1e6},  
			"kb1": {"min": 0, "max": 100},             				        
			"kcat1": {"min": 0, "max": 500},         
			"ka2": {"min": 1000/1e6, "max": 1e6/1e6},         
			"kb2": {"min": 0, "max": 1000}, 
			"ka3": {"min": 1e3/1e6, "max":1e8/1e6}, 
			"kb3": {"min": 0, "max":500},
			"ka4": {"min": 1e3/1e6, "max":1e6/1e6},
			"kb4": {"min": 0, "max": 100},  
			"ka7": {"min": 1e3/1e6, "max": 1e6/1e6}, 
			"kb7": {"min": 0, "max": 1000}, 
			"kcat7": {"min": 0, "max": 200},
			#"V": {"min": 0.3, "max": 0.7},
			"VA": {"min": 0.5, "max": 1.5},
			#"sigma": {"min": 0.001, "max": 0.001}, 
			"L": {"min": 1, "max": 15},  #max 20
			#"Lp": {"min": 0, "max": 0}, 
			"K": {"min": 0.1, "max": 1}, #min 0.01 max 0.1
			"P": {"min": 0.1, "max": 1}, #min 0.01 max 0.1
			#"LK": {"min": 0.0, "max": 0}, 
			"A": {"min": .1, "max": 3}, # max 10
			#"LpA": {"min": 0.0, "max": 0},
			#"LpAK": {"min": 0.0, "max": 0}, 
			#"LpAP": {"min": 0.0, "max": 0}, 	
			#"LpAPLp": {"min": 0.0, "max": 0},
			#"LpAKL": {"min": 0.0, "max": 0},
			#"LpP": {"min": 0.0, "max": 0},
			}      	

filename =  os.path.join(".", "6-27")  
  
model = Oscillator(parameter_values, np.array(["ka1", "kb1", "kcat1", "ka2", "kb2", "ka3", "kb3", "ka4","kb4","ka7","kb7","kcat7","VA","L","K","P","A"]), np.array([0,0,0,0,0,0,0,0,0,0,0,0]), mode=1)  
solver = Solver(model, name = '6-27', volume = 0.5)         







plt.rcParams.update({'font.size': 12}) 

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Candidate", list, fitness=creator.FitnessMax)		
toolbox = base.Toolbox()	 
toolbox.register("candidate", Solver.generateCandidate) 
				
	 
folder1 = os.path.join(".", "/home/local/WIN/jfisch27/Desktop/Oscillator/Python/2022_simple_system/2d_troubleshooting/PEPE/6-27-22")   
			
				
file1 =  os.path.join(folder1, "6-27_Region01ViableSet_Iter10.p")    
		   

viablePoints = pickle.load(open(file1, "rb"))  
   

allViablePoints = viablePoints
pca = PCA(n_components=2)
pca.fit(allViablePoints)

model = Oscillator(parameter_values, np.array(["ka1", "kb1", "kcat1", "ka2", "kb2", "ka3", "kb3", "ka4","kb4","ka7","kb7","kcat7","VA","L","K","P","A"]), np.array([0,0,0,0,0,0,0,0,0,0,0,0]), mode=1)  

###                   ###
###  SSA simulations  ###
###                   ###  

#sample few random points from viable regions for region 2     
readFromFile = False                 

region = viablePoints    
numSamples = 9
if readFromFile:
	sampleNumbers = pickle.load(open("stochastic_samples_numbers_repressilator", "rb"))
else:
	num2 = region.shape[0]  
	sampleNumbers = np.random.choice(num2, numSamples, replace=False)   
	pickle.dump(sampleNumbers, open("stochastic_samples_numbers_repressilator", "wb+"))    
	
#plot few simulations (ode + ssa)
t = np.linspace(0, model.T, model.N)
#t = np.linspace(0, 1000, model2.N)
if readFromFile:
	samples = pickle.load(open("stochastic_samples_repressilator","rb"))     
else:	
	samples = region[sampleNumbers,:]
	pickle.dump(samples, open("stochastic_samples_repressilator", "wb+"))   
	
notWorking = samples[numSamples-3:numSamples,:] 
notWorking = np.copy(notWorking) 
notWorking[0,2] = 1 #low hill coefficient
notWorking[1,0] = 1 #low transcription 
notWorking[2,5] = 10 #high protein degradation   

allSamples = np.vstack((samples, notWorking)) 

ts_total_stochastic = [] 
Y_total_stochastic = []
t_total_ode = []
Y_total_ode = []



if readFromFile:
	t_total_ode, Y_total_ode = pickle.load(open("stochastic_simulations_repressilator", "rb"))  

plt.figure(figsize=(25,25))
for i in range(numSamples + 3): 
	print(i)  
	
	ax1=plt.subplot(4, 3, i + 1)

	ax1.set_title("Sample " + str(i + 1), fontsize=12)	 
	
	sample = allSamples[i] 
	if readFromFile:
		ts = ts_total_stochastic[i]   
		Y = Y_total_stochastic[i]  
		Y_ode = Y_total_ode[i]  
		t = t_total_ode[i]  
	else:
		ts, Y = model.ssa(sample)
		Y = Y/model.omega 
		Y_ode = model.simulate(sample)  
		ts_total_stochastic.append(ts)
		Y_total_stochastic.append(Y) 
		t_total_ode.append(t)  
		Y_total_ode.append(Y_ode) 
	

	plt.plot(t, Y_ode[:,0], color="b", label = 'L', linestyle = "--")
	plt.plot(ts, Y[:,0], color="b")

	plt.plot(t,Y_ode[:,1], color="r", label = 'Lp', linestyle = "--")
	plt.plot(ts,Y[:,1], color="r")

	plt.plot(t,Y_ode[:,2], color="gold", label = 'K', linestyle = "--")
	plt.plot(ts,Y[:,2], color="gold")

	plt.plot(t,Y_ode[:,3], color="g", label = 'P', linestyle = "--")
	plt.plot(ts,Y[:,3], color="g")

	plt.plot(t,Y_ode[:,4], color="cyan", label = 'LK', linestyle = "--")
	plt.plot(ts,Y[:,4], color="cyan")

	plt.plot(t,Y_ode[:,5], color="black", label = 'A', linestyle = "--")
	plt.plot(ts,Y[:,5], color="black")

	plt.plot(t,Y_ode[:,6], color="indigo", label = 'LpA', linestyle = "--")
	plt.plot(ts,Y[:,6], color="indigo")

	plt.plot(t,Y_ode[:,7], color="yellow", label = 'LpAK', linestyle = "--")
	plt.plot(ts,Y[:,7], color="yellow")

	plt.plot(t,Y_ode[:,8], color="magenta", label = 'LpAP', linestyle = "--")
	plt.plot(ts,Y[:,8], color="magenta")

	plt.plot(t,Y_ode[:,9], color="deeppink", label = 'LpAPLp', linestyle = "--")
	plt.plot(ts,Y[:,9], color="deeppink")

	plt.plot(t,Y_ode[:,10], color="peru", label = 'LpAKL', linestyle = "--")
	plt.plot(ts,Y[:,10], color="peru")
	#lines = plt.plot(t, Y_ode[:,0], t, Y_ode[:,1], t, Y_ode[:,2], t, Y_ode[:,3], t, Y_ode[:,4], t, Y_ode[:,5], t, Y_ode[:,6], t, Y_ode[:,7], t, Y_ode[:,8]) #ts, Y[:,1])  
	#plt.setp(lines[0], linestyle = "-", linewidth=1.5, alpha=1, c="#15A357")
	


		
	if i == 9 or i == 10 or i == 11:
		plt.xlabel(r"Time [$h$]")    
	if i == 0 or i == 3 or i == 6 or i == 9:
		plt.ylabel('Concentration [$nM$]') 
	if i == 2:
		plt.legend(loc = 1) 
plt.savefig("6-27_9samples.png")  	
#plt.show()


if not readFromFile:
	pickle.dump((t_total_ode, Y_total_ode) ,open("stochastic_simulations_repressilator", "wb+"))       	
	
#plot boxplots
repetitions = 100    
amps = []
pers = []      
ode_amps = []
ode_pers = []

if readFromFile:
	pers, amps = pickle.load(open("stochastic_persamps_repressilator","rb")) 
for i in range(numSamples): 
	print(i) 
	sample = samples[i]
	per, amp = model2.getPerAmp(sample, mode="ode", indx=1)   
	ode_amps.append(amp)   
	ode_pers.append(per)    
	
	#if not readFromFile:	
		#curra = []
		#currp = []
		#for j in range(repetitions):
			#per, amp = model2.getPerAmp(sample, mode="ssa", indx=1)  
			#amp = amp/model2.omega    
			#curra.append(amp)  
			#currp.append(per)   		
		#amps.append(curra) 
		#pers.append(currp) 

if not readFromFile:	
	pickle.dump((pers, amps), open("stochastic_persamps_repressilator", "wb+"))  

plt.subplot(1,2,1)	
bp = plt.boxplot(amps)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_amps[i], ode_amps[i]], linewidth=3, color="black")
	
plt.ylabel('Amplitude [$nM$]') 
plt.xlabel('Sample \n \n $\mathbf{(a)}$')  


print(ode_amps)   

plt.subplot(1,2,2) 	
bp = plt.boxplot(pers)   
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_pers[i], ode_pers[i]], linewidth=3, color="black")
	
plt.ylabel('Period [$h$]') 
plt.xlabel('Sample \n \n $\mathbf{(b)}$')  

print(ode_pers)   	
plt.show() 



# %%
#plot boxplots
repetitions = 100    
amps = []
pers = []      
ode_amps = []
ode_pers = []

if readFromFile:
	ode_pers, ode_amps = pickle.load(open("ode_persamps_repressilator","rb")) 
for i in range(numSamples): 
	print(i) 
	sample = samples[i]
	per, amp = model.getPerAmp(sample, mode="ode", indx=1)   
	ode_amps.append(amp)   
	ode_pers.append(per)    
	
	#if not readFromFile:	
		#curra = []
		#currp = []
		#for j in range(repetitions):
			#per, amp = model2.getPerAmp(sample, mode="ssa", indx=1)  
			#amp = amp/model2.omega    
			#curra.append(amp)  
			#currp.append(per)   		
		#amps.append(curra) 
		#pers.append(currp) 

if not readFromFile:	
	pickle.dump((ode_pers, ode_amps), open("ode_persamps_repressilator", "wb+"))  



plt.subplot(1,2,1)	
bp = plt.boxplot(ode_amps)     
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_amps[i], ode_amps[i]], linewidth=3, color="black")
	
plt.ylabel('Amplitude [$nM$]') 
plt.xlabel('Sample \n \n $\mathbf{(a)}$')  


print(ode_amps)   

plt.subplot(1,2,2) 	
bp = plt.boxplot(ode_pers)   
for i in range(numSamples):  	
	plt.setp(bp['boxes'][i], color="#0E74C8", linewidth=1.5)    
	plt.setp(bp['caps'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['caps'][2*i + 1], color="#0E74C8", linewidth=1.5) 
	plt.setp(bp['whiskers'][2*i], color="#0E74C8", linewidth=1.5)  
	plt.setp(bp['whiskers'][2*i + 1], color="#0E74C8", linewidth=1.5)   
	plt.setp(bp['fliers'][i], color="#0E74C8") 
	plt.setp(bp['medians'][i], color="#0E74C8", linewidth=1.5)   

	plt.plot([i + 1 - 0.25, i + 1 + 0.25], [ode_pers[i], ode_pers[i]], linewidth=3, color="black")
	
plt.ylabel('Period [$h$]') 
plt.xlabel('Sample \n \n $\mathbf{(b)}$')  

print(ode_pers)   	
plt.show()    

# %%
#plot dterministic simulation    	
#simulationPoint = [49.61, 1.43, 4.4, 21.83, 1.72, 0.78, 123.12] 
#model1.getPerAmp(simulationPoint, mode="ode", indx=1)   

#region1 = solver.Region(viablePoints1, model1, "region1")     
regionTwo = Region(viablePoints2, model2, "region2")     

#region1PCA = pca.transform(viablePoints1)
region2PCA = pca.transform(viablePoints2)  

#plot overlap of both regions 
plt.scatter(region2PCA[:, 0], region2PCA[:, 1], c="#15A357", alpha=1, edgecolor="#158257",  rasterized=True)  
#plt.scatter(region1PCA[:, 0], region1PCA[:, 1], c="#0E74C8", alpha=0.5, edgecolor="#0E56C8",  rasterized=True)
plt.xlabel('PC 1')
plt.ylabel('PC 2') 
plt.show()
#plot points on pca projections  
#filesA = ["represViableSet_IterGA.p", "repres_Region0ViableSet_Iter1.p", "repres_Region0ViableSet_Iter9.p"] 
#filesB = ["repres_Region0CandidateSet_Iter1.p", "repres_Region0CandidateSet_Iter2.p", "repres_Region0CandidateSet_Iter10.p"] 

#pca1 = PCA(n_components=2) 
pca2 = PCA(n_components=2) 

pointsa = pickle.load(open('/home/local/WIN/jfisch27/Desktop/Oscillator/Python/2022_simple_system/2d troubleshooting/FreeKinaseSIMPLE2_oscillatorCostTwo_Region0100101ViableSet_Iter10.p', "rb"))
#pointsb = pickle.load(open(path.join(folder2, filesA[0]) , "rb")) 

#pca1.fit(pointsb) 
pca2.fit(pointsa)  

# %%
#plot viable regions in PCA space     
plt.rcParams.update({'font.size': 12})
i = 1
iters = [1, 2, 10] 
for filea, fileb in zip(filesA, filesB):	

	file1a =  path.join(folder1, filea)     
	file2a =  path.join(folder2, filea)    
	file1b =  path.join(folder1, fileb)     
	file2b =  path.join(folder2, fileb)   	

	viablePoints1 = pickle.load(open(file1a, "rb"))    
	viablePoints2 = pickle.load(open(file2a, "rb"))  
	
	candidatePoints1 = pickle.load(open(file1b, "rb"))  
	candidatePoints2 = pickle.load(open(file2b, "rb"))

	region1PCA = pca1.transform(viablePoints1) 
	region2PCA = pca2.transform(viablePoints2) 
	
	region1PCAcandidates = pca1.transform(candidatePoints1) 
	region2PCAcandidates = pca2.transform(candidatePoints2)   

	ax1=plt.subplot(2, 3, i) 
	ax1.title.set_text("Iteration " + str(iters[i-1]))

	plt.scatter(region1PCAcandidates[:, 0], region1PCAcandidates[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True)  
	plt.scatter(region1PCA[:, 0], region1PCA[:, 1], c="#0E8DC8", alpha=1, edgecolor="#0E5AC8", rasterized=True)    
	if i == 1:
		plt.ylabel('PC 2')  
	if i == 2:
		plt.xlabel(r"$\bf{(a)}}$")   	 	

	ax1=plt.subplot(2, 3, i + 3) 
	plt.scatter(region2PCAcandidates[:, 0], region2PCAcandidates[:, 1], c="#A62B21", alpha=1, edgecolor='#922B21', rasterized=True)   	
	plt.scatter(region2PCA[:, 0], region2PCA[:, 1], c="#0E8DC8", alpha=1, edgecolor="#0E5AC8", rasterized=True)  
	if i == 2:
		plt.xlabel("PC 1 \n"+ r"$\bf{(c)}$") 
	else:
		plt.xlabel("PC 1")   
	if i == 1:
		plt.ylabel('PC 2')    
	
	i += 1
plt.show()	 

# %%
solver_object = solver.Solver(model1)
viableSets = [region1, region2]   		
paramNames = [r"$\alpha$", r"$\alpha_0$",r"$n$", r"$\beta$",r"$\delta_m$", r"$\delta_p$",r"$Kd$"] 
units = [r"[$h^{-1}$]", r"[$h^{-1}$]", "", r"[$h^{-1}$]", r"[$h^{-1}$]", r"[$h^{-1}]$",r"[$nM$]"]    
solver_object.plotParameterVariances(viableSets, names=paramNames, units=units)         


#plot few repressilator responses 	  
print("plotting model 2") 	
number = np.size(viablePoints1, 0)   
rndPoints = np.random.randint(number, size=10)  
for pnt in rndPoints:     
	print(viablePoints1[pnt,:])     
	model1.plotModel(viablePoints1[pnt,:]) 
	model1.plotModel(viablePoints1[pnt,:], mode="ssa")      	
print("plotting model 2")  	
number = np.size(viablePoints2, 0)     
rndPoints = np.random.randint(number, size=10)   
for pnt in rndPoints:    
	model2.plotModel(viablePoints2[pnt,:])   
	model2.plotModel(viablePoints2[pnt,:], mode="ssa")    	


#calculate viable volume for both cases and determine deviations 
solver_object = solver.Solver(model1) 
vol1 = solver_object.getViableVolume([region1], 100000) 
solver_object = solver.Solver(model2)    
vol2 = solver_object.getViableVolume([region2], 100000)      

print("Region 2 is " + str(vol2/vol1) + "greater than Region 1.")      


# %%
viablePoints2 = pickle.load(open('/home/local/WIN/jfisch27/Desktop/Oscillator/Python/2022_simple_system/2d_troubleshooting/PEPE/unrealFreeKinaseSIMPLE2_oscillatorCostTwoViableSet_IterGA.p', "rb")) 

# %%
regionTwo = Region(viablePoints2, model, "region2") 

# %%
paramNames = ["ka1", "kb1", "kcat1", "ka2", "kb2", "ka3", "kb3", "ka4","kb4","ka7","kb7","kcat7","V","VA","sigma","L","Lp","K","P","LK","A","LpA","LpAK","LpAP","LpAPLp","LpAKL","LpP"] 
units = [r"[$h^{-1}$]", r"[$h^{-1}$]", "", r"[$h^{-1}$]", r"[$h^{-1}$]", r"[$h^{-1}]$",r"[$nM$]"]    
solver.plotParameterVariances(regionTwo, names=paramNames, units=units) 

# %%
def plotParameterVariances(self, viableSets, names=None, units=None):      
	#go through all parameters  
	params = self.model.params    
	figure = plt.figure(figsize= (10,25))     
	nRows = math.ceil((len(params)-15)/3)    
	for pcount, param in enumerate(params):
		if pcount < 12:    
			ax1 = plt.subplot(nRows, 3, pcount+1)  
			if names == None:
				ax1.set_title(str(param) + str(pcount))    
			else:
				ax1.set_title(names[pcount])  
			if units != None:
				plt.ylabel(names[pcount] + " " + units[pcount])  
			allRegions = [] 	
			#go through all regions 
			#numSets = len(viableSets) 
			allNames = []
			allBoxes = []
			
			points = np.array(viableSets.points)    
			data = points[:,pcount]   
			allRegions.append(data)   
			#allNames.append("Region " + str(count + 1))   				
			bp = ax1.boxplot(allRegions, sym = '') 
			#self.setBoxColors(bp, numSets, ax1) 		
			allBoxes = bp['boxes'] 
		
	#draw legend 
	#figure.legend(allBoxes, allNames, 'lower right')
	plt.show()     

# %%
paramNames = ["ka1", "kb1", "kcat1", "ka2", "kb2", "ka3", "kb3", "ka4","kb4","ka7","kb7","kcat7","V","VA","sigma","L","Lp","K","P","LK","A","LpA","LpAK","LpAP","LpAPLp","LpAKL","LpP"] 
units = [r"[$h^{-1}$]", r"[$h^{-1}$]", "", r"[$h^{-1}$]", r"[$h^{-1}$]", r"[$h^{-1}]$",r"[$nM$]"]    
plotParameterVariances(solver, regionTwo, names=paramNames, units=None) 

# %%
points = np.array(regionTwo.points) 

# %%
only_params = points[:,:11]
only_params[1].size

# %%
only_params = points[:,:12]
#Km1 = only_params[:,1]+only_params[:,2]/only_params[:,0]
#only_params = np.insert(only_params, 1, Km1, axis=1)
Km1 = only_params[:,1]+only_params[:,2]/only_params[:,0]
Kd2 = only_params[:,4]/only_params[:,3]
Kd3 = only_params[:,6]/only_params[:,5]
Kd4 = only_params[:,8]/only_params[:,7]
Km7 = only_params[:,10]+only_params[:,11]/only_params[:,9]
only_params = np.column_stack((only_params,Km1,Kd2,Kd3,Kd4,Km7))
only_params.shape

# %%
np.column_stack((only_params, Km1,Kd2,Kd3,Kd4,Km7)).shape

# %%
kalist = [only_params[:,0], only_params[:,3], only_params[:,7], only_params[:,9]]
plt.figure(figsize=(15,10))
plt.violinplot(kalist)
plt.xticks([1,2,3,4], ['ka1','ka2','ka4','ka5'])
plt.ylabel(r'($\mu$'+'M'+'$*s)^{-1}$')


# %%
kblist = [only_params[:,1], only_params[:,4], only_params[:,6], only_params[:,8], only_params[:,10]]
plt.figure(figsize=(15,10))
plt.violinplot(kblist)
plt.xticks([1,2,3,4,5], ['kb1','kb2','kb3','kb4', 'kb7'])
plt.ylabel('$s^{-1}$')


# %%
kcatlist = [only_params[:,2], only_params[:,11]]
plt.figure(figsize=(15,10))
plt.violinplot(kcatlist)
plt.xticks([1,2], ['kcat1','kcat7'])
plt.ylabel('$s^{-1}$')

# %%
Kdlist = [(only_params[:,1]+only_params[:,2])/only_params[:,0], only_params[:,4]/only_params[:,3],only_params[:,6]/only_params[:,5], only_params[:,8]/only_params[:,7], (only_params[:,10]+only_params[:,11])/only_params[:,9]]
plt.figure(figsize=(15,10))
plt.violinplot(kblist)
plt.xticks([1,2,3,4,5], ['Km1','Kd2','Kd3','Kd4', 'Km7'])
plt.ylabel(r'$\mu$'+'M')

# %%
def prep_boxplot(region):
	points = np.array(region.points)
	only_params = points[:,:11]
	Km1 = only_params[:,1]+only_params[:,2]/only_params[:,0]
	Kd2 = only_params[:,4]/only_params[:,3]
	Kd3 = only_params[:,6]/only_params[:,5]
	Kd4 = only_params[:,8]/only_params[:,7]
	Km7 = only_params[:,10]+only_params[:,11]/only_params[:,9]
	only_params = np.column_stack((only_params,Km1,Kd2,Kd3,Kd4,Km7))

	return only_params


