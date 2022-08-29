# %%
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import pandas as pd
#import seaborn as sns
#import re
import os.path
import glob

# %% [markdown]
# ## Conversion functions

# %%
def copynumber(conc, volume, um = True):
    if um == True:
        volume = volume/1e15
        conc1 = conc/1e6
        
        moles = conc1 * volume
        copies = moles * 6.023e23
        
        return conc
    elif um == False:
        volume = volume/1e15
        moles = conc / volume
        rate = moles / 6.023e23

        return rate


# %%
##converts to copies/s
##multiply by 10^6, then divide by volume in nm^3

def rate_converter(conc,V):
    #conc1 = conc #/0.602214076 ## convert from uM-1 s-1 to nm^3/us
    conc2 = conc * 1e6 ##convert to seconds from microseconds
    nanoV = V #*1e9 ##convert volume from um3 to nm3
    return conc2/nanoV ##divide by volume to get copies per second

# %%
# from copies to uM
def concentration(copy, V):
    umoles = copy/6.023e17
    liters = V/1e24

    uM = umoles/liters

    return uM

def main():
    V = int
    VA = int

    file = glob.glob('*.inp')
    parm = open(file[0], 'r')

    for line in parm:
        
        if 'WaterBox' in line:
            linesplit = line.split(' ')
        
            x = linesplit[2][1:4]
            
            y = linesplit[3][:-1]
            
            z = linesplit[4][:-2]

            V = int(x)*int(y)*int(z)
            VA = int(z)
            #V = int(dim[0][1:])*int(dim[1])*int(dim[2][:-2]) 
            #VA = int(dim[2][:-2])


    #print(V) 
    #print(VA)



    # %% [markdown]
    # ## Calculate concentrations from parms file

    # %%
    file = glob.glob('*.inp')
    parm = open(file[0], 'r')

    pip2 = int 
    ap2 = int
    kin = int
    syn = int

    for line in parm:

        if 'pip2 :' in line:
            linesplit = line.split(' ')
            pip2 = int(linesplit[2])
        
        if 'ap2 :' in line:
            linesplit = line.split(' ')
            ap2 = int(linesplit[2])

        if 'kin :' in line:
            linesplit = line.split(' ')
            kin = int(linesplit[2])

        if 'syn :' in line:
            linesplit = line.split(' ')
            syn = int(linesplit[2])

    L = concentration(pip2, V)
    A = concentration(ap2, V)
    K = concentration(kin, V)
    P = concentration(syn, V)



    # %% [markdown]
    # ## Load copy number data into dataframe

    # %%
    df = pd.read_csv('copy_numbers_time.dat')


    # %%
    #df.plot.line(x='Time (s)', figsize=(20,10))

    # %% [markdown]
    # ## Plot for copynumbers

    # %%
    #ka_ids = [0,3,5,7,9]

    #HERE#
    input = [
    0.00040370279168983574 , #ka1 #change 
    57.11277101514493 , #kb1 #change 
    362.1374668043775 , #kcat1 #change 
    0.003213372917272041 , #ka2 #change 
    1.5425586046848487 , #kb2 #change 
    0.25456803713943227 , #ka3 #change 
    266.6560866362819 , #kb3 #change 
    6.488748769693232e-05 , #ka4 #change 
    0.16285136809161127 , #kb4 #change 
    0.004385696809106526 , #ka7 #change 
    31.04091356562089 , #kb7 #change 
    31.5155642816625 , #kcat7 #change 
    734.834057877303 , #y #change 
    0.0 , #L #change 
    903.45 , #Lp #change 
    60.23000000000001 , #K #change 
    90.345 , #P #change 
    0.0 , #LK #change 
    271.035 , #A #change 
    0.0 , #Lpa #change 
    0.0 , #LpAK #change 
    0.0 , #LpAP #change 
    0.0 , #LpAPLp #change 
    0.0 , #LpAKL #change 
    0.0 ] #LpP #change 

    # Create a timeline from 0 to 50 divided into a thousand steps
    t = np.linspace(0,100,num=1000)

    #convert NERDSS ka rates to copynumber rates
    #can = [rate_converter(x,V) if input[:13].index(x) in ka_ids else x for x in input[:13]]

    can = input[:13]

    initial_populations = input[13:]

    def calculate_current_population_change_rates(current_counts, t, can):

        #V = can[12]
        #VA = can[14]
        #A = V/VA
        #sigma = can[13]
        y1 = can[12]
        
        
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

        
        #initial conditions
        L = current_counts[0]   # make sure in uM
        Lp = current_counts[1] 
        K = current_counts[2]
        P = current_counts[3]
        LK = current_counts[4] 
        A = current_counts[5]
        LpA = current_counts[6]
        LpAK = current_counts[7]
        LpAP = current_counts[8]
        LpAPLp = current_counts[9]
        LpAKL = current_counts[10]
        LpP = current_counts[11]



        dL = (kb1*LK) - (ka1*L*K) + (kcat5*LpAPLp) + (kb6*LpAKL) - ((y1)*ka6*LpAK*L) + (kcat7*LpP)
        dLp = (kcat1*LK) + (kb2*LpA) - (ka2*Lp*A) + (kb5*LpAPLp) - ((y1)*ka5*Lp*LpAP) + (kcat6*LpAKL) - (ka7*Lp*P) + (kb7*LpP)
        dK = (kb1*LK) - (ka1*L*K) + (kcat1*LK) + (kb3*LpAK) - (ka3*LpA*K)
        dP = (kb4*LpAP) - (ka4*LpA*P) - (ka7*Lp*P) + (kb7*LpP) + (kcat7*LpP)
        dLK = (ka1*L*K) - (kb1*LK) - (kcat1*LK)
        dA = (kb2*LpA) - (ka2*Lp*A)
        dLpA = (ka2*Lp*A) - (kb2*LpA) + (kb3*LpAK) - (ka3*LpA*K) + (kb4*LpAP) - (ka4*LpA*P)
        dLpAK = (ka3*LpA*K) - (kb3*LpAK) + (kb6*LpAKL) - ((y1)*ka6*LpAK*L) + (kcat6*LpAKL)
        dLpAP = (ka4*LpA*P) - (kb4*LpAP) + (kb5*LpAPLp) - ((y1)*ka5*LpAP*Lp) + (kcat5*LpAPLp)
        dLpAPLp = (y1*ka5*LpAP*Lp) - (kb5*LpAPLp) - (kcat5*LpAPLp)
        dLpAKL = ((y1)*ka6*LpAK*L) - (kb6*LpAKL) - (kcat6*LpAKL)
        dLpP = (ka7*Lp*P) - (kb7*LpP) - (kcat7*LpP)

        return([dL, dLp, dK, dP, dLK, dA, dLpA, dLpAK, dLpAP, dLpAPLp, dLpAKL, dLpP])	

    # Repeatedly calls 'calculate_current_population_change_rates' for every time step and solves numerically to get the population numbers
    solutions = odeint(calculate_current_population_change_rates, initial_populations, t, args=(can,))

    #solutions = solve_ivp(fun=calculate_current_population_change_rates,method='BDF', t_span=(0, 100), 
    #            y0=initial_populations, t_eval=t, args=(params,),rtol=1e-6, atol=1e-9)


    # %%
    fig = plt.figure(figsize=(8,4),dpi=200)

    plt.plot(t,solutions[:,0], color="b", label = 'L')
    plt.plot(t,solutions[:,1], color="g", label = 'Lp')
    plt.plot(t,solutions[:,2], color="gold", label = 'K')
    plt.plot(t,solutions[:,3], color="r", label = 'P')
    plt.plot(t,solutions[:,4], color="cyan", label = 'LK')
    plt.plot(t,solutions[:,5], color="black", label = 'A')

    plt.plot(df['Time (s)'], df['pip2(head~U)'], color = 'b', linestyle = 'dashed', alpha = 0.7)
    plt.plot(df['Time (s)'], df['pip2(head~P)'], color = 'g', linestyle = 'dashed', alpha = 0.7)
    plt.plot(df['Time (s)'], df['kin(ap)'], color = 'gold', linestyle = 'dashed', alpha = 0.7)
    plt.plot(df['Time (s)'], df['syn(ap)'], color = 'r', linestyle = 'dashed', alpha = 0.7)
    plt.plot(df['Time (s)'], df['kin(pi!1).pip2(head~U!1)'], color = 'cyan', linestyle = 'dashed', alpha = 0.7)
    plt.plot(df['Time (s)'], df['ap2(m2muh)'], color = 'black', linestyle = 'dashed', alpha = 0.7)

    plt.text(0, -1100, 'L: ' + str(round(L)) + ' uM' + '\n' + 
    'A: ' + str(round(A)) + ' uM' + '\n' +
    'K: ' + str(round(K)) + ' uM' + '\n' +
    'P: ' + str(round(P)) + ' uM')

    plt.text(70, -900, 'Volume: ' + str(V*1e-9) + ' $um^3$' + '\n' + 'V/A: ' + str(VA*1e-3) + ' um')

    plt.xlabel('Time (s)')
    plt.ylabel('Copies')
    plt.legend(loc = 'upper right')
    plt.save('compare.png')


if __name__ == '__main__':
    main()