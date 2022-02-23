#!/usr/bin/python

import numpy as np
from scipy.integrate import odeint
from scipy.signal import find_peaks
from sympy import *



def calculate_current_population_change_rates(current_counts, t, params):

    #xn = 100
    xn = current_counts[0]
    xc = current_counts[1]
    
    yc = current_counts[2]
    yn = current_counts[3]


    sig = params[0]
    p = params[1]
    kdxn = params[2]
    kexport = params[3]
    kdxc = params[4]
    eps = params[5]
    kdyn = params[6]
    kdyc = params[7]
    Km = params[8]
    kimport = params[9]

    dxn_dt = kdxn*(sig/(1 + yn**p) - xn) - kexport*xn 
    dxc_dt = eps*kexport*xn - kdxc*xc
    dyc_dt = kdyc*(xc - yc) - eps*kimport*yc 
    dyn_dt = (kimport*yc) - (kdyn*yn/(Km + yn))




    return([dxn_dt, dxc_dt, dyc_dt, dyn_dt])


def sweeper():
    #lists of parameter ranges
    sig_sweep = np.arange(100, 1100, 10).tolist()
    p_sweep = np.arange(1, 4).tolist()
    kdxn_sweep = np.arange(1, 50).tolist()
    kexport_sweep = np.arange(0.1, 1).tolist()
    kdxc_sweep = np.arange(0.1,1).tolist()
    eps_sweep = np.arange(0.1, 2).tolist()
    kdyn_sweep = np.arange(1,10).tolist()
    kdyc_sweep = np.arange(0.1, 2, 0.1).tolist()
    Km_sweep = np.arange(0.1, 2, 0.1).tolist()
    kimport_sweep = np.arange(0.1, 2, 0.1).tolist()

    param_range = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    for a in sig_sweep:
        for b in p_sweep:
            for c in kdxn_sweep:
                for d in kexport_sweep:
                    for e in kdxc_sweep:
                        for f in eps_sweep:
                            for g in kdyn_sweep:
                                for h in kdyc_sweep:
                                    for i in Km_sweep:
                                        for j in kimport_sweep:
                                            initial_populations = [0,40,100,40] # [prey, predators] units in hundreds

                                            # Create a timeline from 0 to 50 divided into a thousand steps
                                            t = np.linspace(0,50,num=1000)

                                
                                            sig=a
                                            p=b
                                            kdxn=c
                                            kexport=d
                                            kdxc=e
                                            eps=f
                                            kdyn=g
                                            kdyc=h
                                            Km=i
                                            kimport=j

                                            params = [sig, p, kdxn, kexport, kdxc, eps, kdyn, kdyc, Km, kimport]

                                        
                                            solutions = odeint(calculate_current_population_change_rates, initial_populations, t, args=(params,))

                                            xn_peaks = len(find_peaks(solutions[:,0])[0])
                                            xc_peaks = len(find_peaks(solutions[:,1])[0])
                                            yc_peaks = len(find_peaks(solutions[:,2])[0])
                                            yn_peaks = len(find_peaks(solutions[:,3])[0])
                    
                                        
                                            param_range = np.append(param_range, [[xn_peaks,xc_peaks,yc_peaks,yn_peaks,sig,p,kdxn,kexport,kdxc,eps,kdyn,kdyc,Km,kimport]], axis = 0)
                                            
    return np.delete(param_range,0,0)

#sys.stdout=open("sweep.txt","w")
sweep = sweeper()
#sys.stdout.close()
np.savetxt('sweep.csv', sweep, delimiter=',')

